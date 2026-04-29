#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import EMBEDDING_MODEL_DIR, QueryConfig, RERANKER_MODEL_DIR  # noqa: E402
from goldenglow.data.sft_teacher import (  # noqa: E402
    TeacherApiConfig,
    build_normalized_sample_id,
    call_teacher_api,
    load_generation_config,
    load_story_documents,
    make_sample_fingerprint,
    parse_teacher_json,
    sample_evidence_documents,
    split_samples,
    weighted_choice,
)
from scripts.merge_sft_datasets import merge_datasets  # noqa: E402


SUPPORTED_SUPPLEMENT_TASK_TYPES = {
    "user_question_hypothesis_generation",
    "follow_up_hypothesis_generation",
    "conclusion_generation",
}

ACTION_CHOICES = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
HYPOTHESIS_INTENTS = {
    "plot_fact",
    "plot_reasoning",
    "timeline",
    "character_relation",
    "event_summary",
    "compare",
    "persona_chat",
    "out_of_scope",
}
INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"
INITIAL_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "intent",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
CONCLUSION_SCHEMA_FIELDS = (
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
)
LEGACY_INTENT_MAP = {
    "plot_inference": "plot_reasoning",
    "plot_motivation": "plot_reasoning",
    "character_motivation": "plot_reasoning",
    "identity_relationship": "character_relation",
    "character_relationship": "character_relation",
    "relationship_inference": "character_relation",
    "character_identity": "plot_fact",
    "role_identification": "plot_fact",
    "plot_item": "plot_fact",
    "plot_explanation": "plot_reasoning",
    "plot_qa": "plot_fact",
    "follow_up": "plot_fact",
    "clarification_needed": "out_of_scope",
}
LEGACY_PROMPT_HYPOTHESIS_MARKERS = (
    '"character_name"',
    '"appearances"',
    '"known_info"',
    '"relationship_hints"',
    '"bridging_objects"',
    '"bridge_objects"',
    '"upstream_titles"',
    '"upper_titles"',
    '"relationship_clues"',
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PRIMARY_ENTITY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
CODE_NAME_RE = re.compile(r"【代号】\s*([\u4e00-\u9fff]{2,8})")
REAL_NAME_RE = re.compile(r"本名([\u4e00-\u9fff]{2,8})")
OPERATOR_NAME_RE = re.compile(r"干员([\u4e00-\u9fff]{2,8})")
PRIMARY_ENTITY_STOP_WORDS = {
    "用户问题",
    "多轮上下文",
    "当前假设",
    "当前证据",
    "历史生成",
    "历史检索",
    "检索轮次",
    "行动前",
    "行动后",
    "幕间",
    "档案",
    "语音",
    "剧情",
    "故事",
    "问题",
    "关系",
    "身份",
    "来历",
    "真相",
    "原因",
    "动机",
}


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def default_merged_output_dir(base_dir: Path, supplement_dir: Path) -> Path:
    return base_dir.parent / f"{base_dir.name}_plus_{supplement_dir.name}"


def is_fatal_teacher_api_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.lower()
    fatal_markers = (
        "http error 401",
        "http error 403",
        "httperror 401",
        "httperror 403",
        "subscription_not_found",
        "invalid api key",
        "unauthorized",
        "forbidden",
        "insufficient_permissions",
        "permission denied",
    )
    return any(marker in normalized for marker in fatal_markers)


def build_retrieval_seed_query(document: dict[str, Any], *, max_chars: int) -> str:
    parts: list[str] = []
    for key in ("activity_name", "story_name", "stage_code", "avg_tag"):
        value = document.get(key)
        if value:
            parts.append(str(value))

    speakers: list[str] = []
    for segment in document.get("segments") or []:
        speaker = segment.get("speaker") if isinstance(segment, dict) else None
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) >= 4:
            break
    if speakers:
        parts.append(" ".join(speakers))

    clean_text = str(document.get("clean_text") or "")
    if clean_text:
        parts.append(clean_text[:max_chars])
    return "\n".join(part for part in parts if part).strip()


def load_retriever(*, device: str):
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever

    return ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=RERANKER_MODEL_DIR,
        device=device,
    )


def retrieve_evidence_documents(
    retriever,
    *,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    results = retriever.search(
        query,
        config=QueryConfig(
            dense_top_k=max(40, top_k * 8),
            sparse_top_k=max(40, top_k * 8),
            fusion_top_k=max(30, top_k * 6),
            rerank_top_k=top_k,
        ),
    )
    return [item["document"] for item in results]


def compute_target_counts(target_total: int, task_mix: dict[str, float]) -> dict[str, int]:
    total_weight = sum(task_mix.values())
    targets = {
        task_type: int(target_total * (weight / total_weight))
        for task_type, weight in task_mix.items()
    }
    assigned = sum(targets.values())
    remainders = sorted(
        (
            (
                target_total * (weight / total_weight) - targets[task_type],
                task_type,
            )
            for task_type, weight in task_mix.items()
        ),
        reverse=True,
    )
    for _, task_type in remainders[: max(0, target_total - assigned)]:
        targets[task_type] += 1
    return targets


def choose_task_type(
    rng: random.Random,
    task_mix: dict[str, float],
    target_counts: dict[str, int],
    current_counts: Counter,
) -> str:
    remaining = {
        task_type: max(0, target_counts[task_type] - current_counts.get(task_type, 0))
        for task_type in task_mix
    }
    deficit_weights = {task_type: count for task_type, count in remaining.items() if count > 0}
    if deficit_weights:
        return weighted_choice(rng, deficit_weights)
    return weighted_choice(rng, task_mix)


def quotas_satisfied(current_counts: Counter, target_counts: dict[str, int]) -> bool:
    return all(
        current_counts.get(task_type, 0) >= target
        for task_type, target in target_counts.items()
    )


def format_evidence_pack(evidence_docs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(evidence_docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or ''}",
                    f"stage_code: {doc.get('stage_code') or ''}",
                    f"avg_tag: {doc.get('avg_tag') or ''}",
                    f"source_path: {doc.get('source_path') or ''}",
                    "clean_text:",
                    doc.get("clean_text", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def extract_primary_entity_candidates(
    evidence_docs: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[str]:
    counts: Counter[str] = Counter()
    for doc in evidence_docs:
        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            if speaker and speaker not in PRIMARY_ENTITY_STOP_WORDS:
                counts[speaker] += 4

        clean_text = str(doc.get("clean_text") or "")[:200]
        for pattern in (CODE_NAME_RE, REAL_NAME_RE, OPERATOR_NAME_RE):
            for match in pattern.findall(clean_text):
                token = str(match).strip()
                if token and token not in PRIMARY_ENTITY_STOP_WORDS:
                    counts[token] += 2

    return [name for name, _ in counts.most_common(limit)]


def build_latest_hypothesis_schema_example(
    *,
    question: str = "烛煌的真实身份是什么？",
    intent: str = "plot_fact",
    entities: list[str] | None = None,
    keywords: list[str] | None = None,
    expected_answer_type: str = "身份关系",
    dialogue_context: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "intent": intent,
        "entities": entities or ["烛煌"],
        "keywords": keywords or ["烛煌", "真实身份", "身世", "来历"],
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def build_latest_hypothesis_schema_block() -> str:
    return json.dumps(
        build_latest_hypothesis_schema_example(),
        ensure_ascii=False,
        indent=2,
    )


def build_follow_up_hypothesis_schema_example(
    *,
    question: str = "烛煌的真实身份是什么？",
    entities: list[str] | None = None,
    keywords: list[str] | None = None,
    expected_answer_type: str = "身份关系",
    dialogue_context: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "entities": entities or ["烛煌", "太师"],
        "keywords": keywords or ["烛煌", "太师", "身世", "太师是谁", "烛煌 太师 什么关系"],
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def build_conclusion_schema_example(
    *,
    question: str = "烛煌的真实身份是什么？",
    next_action: str = "retrieve_more",
    answer: str = "",
    missing_slots: list[str] | None = None,
    clarification_question: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots or ["太师是谁", "烛煌与太师的关系"],
        "clarification_question": clarification_question,
    }


def build_follow_up_hypothesis_schema_block() -> str:
    return json.dumps(
        build_follow_up_hypothesis_schema_example(),
        ensure_ascii=False,
        indent=2,
    )


def build_conclusion_schema_block() -> str:
    return json.dumps(
        build_conclusion_schema_example(),
        ensure_ascii=False,
        indent=2,
    )


def build_initial_hypothesis_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题，不要改写成别的问题。",
            "- `intent`: 问题的语义类型，只表示问题类型，不表示流程状态。可选值仅限 plot_fact、plot_reasoning、timeline、character_relation、event_summary、compare、persona_chat、out_of_scope。",
            "- `entities`: 当前问题里最核心、最值得用于检索的实体，一般是角色、组织、地点、事件名。第一个元素必须是主实体。",
            "- `keywords`: 比 entities 更宽的检索词，可以包含原词、同义改写、关系词、短语化检索扩展；必须包含主实体。",
            "- `expected_answer_type`: 期望最终答案的形态，例如事实问答、身份关系、原因/动机、时间线、过程解释。",
            "- `dialogue_context`: 多轮对话上下文，用于补指代和追问背景；如果没有多轮上下文，可以省略，系统会按空字符串处理。",
        ]
    )


def build_follow_up_hypothesis_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题，保持不变。",
            "- `entities`: 在上一轮实体基础上，补充本轮证据里出现的关键桥接对象；第一个元素必须保留主实体。",
            "- `keywords`: 面向下一轮检索的缩小范围关键词，优先加入关系词、称谓、桥接短语；必须包含主实体。",
            "- `expected_answer_type`: 继续沿用当前问题所需的答案形态，例如身份关系、原因/动机、时间线、过程解释。",
            "- `dialogue_context`: 多轮对话上下文；通常沿用上一轮上下文，没有则可为空字符串。",
            "- `intent`: 不在 assistant 输出中出现，默认继承上一轮 hypothesis 的 intent。",
        ]
    )


def build_conclusion_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题。",
            "- `next_action`: 当前证据下的下一步动作，只能是 answer_directly、retrieve_more、clarify_user、abstain。",
            "- `answer`: 当前阶段结论文本。answer_directly 或 abstain 时必须非空；retrieve_more 时必须为空字符串。",
            "- `missing_slots`: 当前证据还缺哪些具体可检索的信息缺口，主要在 retrieve_more 时使用。",
            "- `clarification_question`: 当问题本身有歧义时，向用户发出的澄清问题；仅 clarify_user 时必须非空。",
        ]
    )


def build_initial_hypothesis_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    primary_entity_candidates = extract_primary_entity_candidates(evidence_docs)
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 中间推理步骤的高质量 SFT 样本。"
        "当前只生成“假设文档 prompt”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )

    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 hypothesis_builder。"},
        {
            "role": "user",
            "content": (
                "用户问题: 烛煌的真实身份是什么？\n"
                "多轮上下文: 无\n"
                "请生成初始假设文档 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_latest_hypothesis_schema_example(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]

    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }

    requirements = [
        f"1. 只生成 `{INITIAL_HYPOTHESIS_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用初始 hypothesis schema：question、intent、entities、keywords、expected_answer_type、dialogue_context。",
        "5. assistant JSON 不允许出现任何额外字段。",
        "6. intent 只能从以下集合中选择：" + "、".join(sorted(HYPOTHESIS_INTENTS)) + "。",
        "7. user prompt 应围绕“用户问题 + 多轮上下文 -> 初始假设文档 JSON”。",
        "8. assistant JSON 目标是服务检索，不是直接回答问题；不得把最终结论当作既定事实写死。",
        "9. 不要编造英文别名、Dr. 前缀、代号扩写、罗德岛职位推断或跨角色别名污染。",
        "10. 所有文本使用中文。",
        "11. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "12. 每条样本都必须围绕 1 到 2 个主实体构造问题；assistant JSON 的 `entities` 第一个元素必须是主实体。",
        "13. `keywords` 必须包含主实体，不要生成不带锚点实体的宽泛检索词。",
    ]

    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“初始假设文档生成”训练样本。\n\n"
        "建议主实体候选（优先围绕这些角色/称谓出题）:\n"
        + ("、".join(primary_entity_candidates) if primary_entity_candidates else "无")
        + "\n\n"
        "当前项目唯一合法的初始 hypothesis schema:\n"
        + build_latest_hypothesis_schema_block()
        + "\n\n"
        "字段含义说明:\n"
        + build_initial_hypothesis_field_explanations()
        + "\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_follow_up_hypothesis_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    primary_entity_candidates = extract_primary_entity_candidates(evidence_docs)
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 多轮补充检索步骤的高质量 SFT 样本。"
        "当前只生成“多轮补充假设文档生成”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )

    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 follow_up_hypothesis_builder。"},
        {
            "role": "user",
            "content": (
                "用户问题: 烛煌的真实身份是什么？\n多轮上下文: 无\n"
                "当前假设文档(JSON): "
                + json.dumps(
                    build_latest_hypothesis_schema_example(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                '上一轮结论生成结果(JSON): {"question":"烛煌的真实身份是什么？","next_action":"retrieve_more","answer":"","missing_slots":["太师是谁","烛煌与太师的关系"],"clarification_question":""}\n'
                "历史生成结果: [第1轮 hypothesis 已定位到烛煌身份问题，但尚未补出太师桥接线索]\n"
                "历史检索上下文: [第1轮检索已使用“烛煌 身份 来历”等查询，但仍缺太师相关桥接信息]\n"
                "当前检索轮次: 第2轮 / 最多3轮\n"
                "当前证据: [...]\n当前未解点: 还不知道太师是谁，也不知道烛煌和太师的关系。\n请生成补充检索假设文档 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_follow_up_hypothesis_schema_example(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]

    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }

    requirements = [
        f"1. 只生成 `{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用补充 hypothesis schema：question、entities、keywords、expected_answer_type、dialogue_context。",
        "5. assistant JSON 不允许出现 intent，follow-up hypothesis 必须继承上一轮 intent。",
        "6. user prompt 必须包含：用户问题、多轮上下文、当前假设文档(JSON)、上一轮结论生成结果(JSON)、历史生成结果、历史检索上下文、当前证据、当前检索轮次。",
        "7. user prompt 里的 `当前假设文档(JSON)` 必须严格使用初始 hypothesis schema。",
        '8. 不要在 user prompt 或 assistant JSON 中使用 `character_name`、`appearances`、`known_info`、`relationship_hints`、`bridging_objects`、`constraints`、`aliases` 等旧字段或衍生字段。',
        "9. assistant JSON 应只生成更强的检索线索，不直接回答问题。",
        "10. assistant JSON 的 keywords 应体现缩小范围后的二次检索查询。",
        "11. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "12. follow-up hypothesis 必须保留上一轮主实体；assistant JSON 的 `entities` 第一个元素必须仍然是主实体。",
        "13. `keywords` 必须保留主实体，并在其基础上补充桥接词，不要丢失锚点实体。",
    ]

    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“多轮补充假设文档生成”训练样本。\n\n"
        "建议主实体候选（优先围绕这些角色/称谓缩小检索范围）:\n"
        + ("、".join(primary_entity_candidates) if primary_entity_candidates else "无")
        + "\n\n"
        "当前项目唯一合法的初始 hypothesis schema:\n"
        + build_latest_hypothesis_schema_block()
        + "\n\n"
        "当前项目唯一合法的 follow-up hypothesis schema:\n"
        + build_follow_up_hypothesis_schema_block()
        + "\n\n"
        "字段含义说明:\n"
        + build_follow_up_hypothesis_field_explanations()
        + "\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_conclusion_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 结论生成步骤的高质量 SFT 样本。"
        "当前只生成“结论生成”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )

    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 conclusion_generator。"},
        {
            "role": "user",
            "content": (
                "用户问题: 烛煌的真实身份是什么？\n多轮上下文: 无\n"
                "当前假设文档(JSON): "
                + json.dumps(
                    build_latest_hypothesis_schema_example(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                "历史检索上下文: [第1轮已检索烛煌身世相关片段，但仍缺太师桥接信息]\n"
                "当前检索轮次: 第2轮 / 最多3轮\n"
                "当前证据: [...]\n请基于证据生成当前阶段结论 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_conclusion_schema_example(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]

    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": CONCLUSION_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "decision_case": "answer_directly|retrieve_more|clarify_user|abstain",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }

    requirements = [
        f"1. 只生成 `{CONCLUSION_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用：question、next_action、answer、missing_slots、clarification_question。",
        "5. next_action 只能是 `answer_directly`、`retrieve_more`、`clarify_user`、`abstain`。",
        "6. user prompt 必须包含：用户问题、多轮上下文、当前假设文档(JSON)、当前证据；当前假设文档不能是空对象 {}。",
        "7. conclusion prompt 必须显式带出当前检索轮次与历史检索上下文。",
        "8. `answer_directly` 或 `abstain` 时，answer 必须非空。",
        "9. `clarify_user` 时 clarification_question 必须非空。",
        "10. `retrieve_more` 时 answer 必须为空字符串，missing_slots 必须为具体可检索缺口。",
        "11. 结论生成样本必须显式体现“基于当前证据是否足够作答”的判断。",
        "12. 所有文本使用中文，不要编造英文别名或 Dr. 前缀。",
        "13. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
    ]

    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“结论生成”训练样本。\n\n"
        "当前项目唯一合法的初始 hypothesis schema:\n"
        + build_latest_hypothesis_schema_block()
        + "\n\n"
        "当前项目唯一合法的 conclusion schema:\n"
        + build_conclusion_schema_block()
        + "\n\n"
        "字段含义说明:\n"
        + build_conclusion_field_explanations()
        + "\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_teacher_prompts(
    *,
    task_type: str,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        return build_initial_hypothesis_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )
    if task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
        return build_follow_up_hypothesis_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )
    if task_type == CONCLUSION_TASK_TYPE:
        return build_conclusion_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )
    raise ValueError(f"Unsupported supplement task type: {task_type}")


def _parse_json_content(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    match = JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _is_noisy_term(value: str) -> bool:
    lowered = value.lower()
    return (
        "{@" in value
        or "{" in value
        or "}" in value
        or "@nickname" in lowered
        or lowered.startswith("dr.")
        or lowered.startswith("doctor ")
    )


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [item for item in value if isinstance(item, (str, int, float))]
    else:
        return []
    items = [
        str(item).strip()
        for item in raw_items
        if str(item).strip() and not _is_noisy_term(str(item).strip())
    ]
    return _dedupe_keep_order(items)[:limit]


def _normalize_intent(value: Any) -> str:
    intent = str(value or "").strip()
    intent = LEGACY_INTENT_MAP.get(intent, intent)
    return intent if intent in HYPOTHESIS_INTENTS else ""


def _extract_expected_answer_type(payload: dict[str, Any]) -> str:
    return str(payload.get("expected_answer_type") or "").strip()


def _contains_empty_current_hypothesis(user_text: str) -> bool:
    normalized = re.sub(r"\s+", "", user_text)
    return "当前假设文档(JSON):{}" in normalized or "当前假设文档:{}" in normalized


def _contains_legacy_prompt_hypothesis_schema(user_text: str) -> bool:
    return any(marker in user_text for marker in LEGACY_PROMPT_HYPOTHESIS_MARKERS)


def _normalize_initial_hypothesis_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    required_fields = tuple(field for field in INITIAL_HYPOTHESIS_SCHEMA_FIELDS if field != "dialogue_context")
    if any(field not in payload for field in required_fields):
        return None
    extra_keys = set(payload) - set(INITIAL_HYPOTHESIS_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    intent = _normalize_intent(payload.get("intent"))
    entities = _normalize_string_list(payload.get("entities"), limit=12)
    keywords = _normalize_string_list(payload.get("keywords"), limit=20)
    expected_answer_type = _extract_expected_answer_type(payload)
    dialogue_context = str(payload.get("dialogue_context") or "").strip()

    if not question or not intent or not entities or not keywords or not expected_answer_type:
        return None

    normalized = {
        "question": question,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }
    return normalized


def _normalize_follow_up_hypothesis_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if any(field not in payload for field in FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS):
        return None
    extra_keys = set(payload) - set(FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    entities = _normalize_string_list(payload.get("entities"), limit=12)
    keywords = _normalize_string_list(payload.get("keywords"), limit=20)
    expected_answer_type = _extract_expected_answer_type(payload)
    dialogue_context = str(payload.get("dialogue_context") or "").strip()
    if not question or not entities or not keywords or not expected_answer_type:
        return None
    return {
        "question": question,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def _normalize_conclusion_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if any(field not in payload for field in CONCLUSION_SCHEMA_FIELDS):
        return None
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    next_action = str(payload.get("next_action") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    missing_slots = _normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()
    if not question or next_action not in ACTION_CHOICES:
        return None
    if next_action in {"answer_directly", "abstain"} and not answer:
        return None
    if next_action == "clarify_user" and not clarification_question:
        return None
    if next_action == "retrieve_more":
        if answer or not missing_slots:
            return None
    else:
        if next_action != "clarify_user":
            clarification_question = ""
    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
    }


def validate_and_normalize_samples(
    payload: dict[str, Any],
    *,
    expected_task_type: str,
    evidence_docs: list[dict[str, Any]],
    request_id: str,
) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Teacher payload must contain a list field named 'samples'")

    source_story_ids = [doc.get("story_id") for doc in evidence_docs if doc.get("story_id")]
    source_stage_codes = [doc.get("stage_code") for doc in evidence_docs if doc.get("stage_code")]
    source_activity_names = [doc.get("activity_name") for doc in evidence_docs if doc.get("activity_name")]

    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        task_type = sample.get("task_type") or expected_task_type
        if task_type != expected_task_type or task_type not in SUPPORTED_SUPPLEMENT_TASK_TYPES:
            continue

        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            continue

        clean_messages: list[dict[str, Any]] = []
        valid = True
        for message in messages:
            if not isinstance(message, dict):
                valid = False
                break
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                valid = False
                break
            if message.get("tool_calls") or role == "tool":
                valid = False
                break
            clean_messages.append({"role": role, "content": content if isinstance(content, str) else ""})
        if not valid:
            continue

        if not any(message["role"] == "system" for message in clean_messages):
            continue
        if not any(message["role"] == "user" for message in clean_messages):
            continue
        user_message = next(
            (message for message in clean_messages if message["role"] == "user"),
            None,
        )
        if user_message is None:
            continue
        if task_type in {
            FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
            CONCLUSION_TASK_TYPE,
        } and _contains_empty_current_hypothesis(user_message["content"]):
            continue
        if _contains_legacy_prompt_hypothesis_schema(user_message["content"]):
            continue

        final_assistant = next(
            (
                message
                for message in reversed(clean_messages)
                if message["role"] == "assistant" and str(message.get("content") or "").strip()
            ),
            None,
        )
        if final_assistant is None:
            continue

        assistant_payload = _parse_json_content(final_assistant.get("content"))
        if assistant_payload is None:
            continue

        normalized_assistant_payload: dict[str, Any] | None = None
        if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
            normalized_assistant_payload = _normalize_initial_hypothesis_assistant_payload(assistant_payload)
        elif task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
            normalized_assistant_payload = _normalize_follow_up_hypothesis_assistant_payload(assistant_payload)
        elif task_type == CONCLUSION_TASK_TYPE:
            normalized_assistant_payload = _normalize_conclusion_assistant_payload(assistant_payload)
        if normalized_assistant_payload is None:
            continue
        final_assistant["content"] = json.dumps(
            normalized_assistant_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
        normalized.append(
            {
                "id": build_normalized_sample_id(
                    request_id=request_id,
                    task_type=task_type,
                    index=index,
                ),
                "task_type": task_type,
                "bucket": "tool",
                "messages": clean_messages,
                "tools": [],
                "meta": {
                    "category": "tool",
                    "grounded": True,
                    "difficulty": meta.get("difficulty") or "medium",
                    "notes": meta.get("notes") or "",
                    "source_story_ids": meta.get("source_story_ids") or source_story_ids,
                    "source_stage_codes": meta.get("source_stage_codes") or source_stage_codes,
                    "source_activity_names": meta.get("source_activity_names") or source_activity_names,
                    "task_family": "hypothesis_generation"
                    if task_type in {INITIAL_HYPOTHESIS_TASK_TYPE, FOLLOW_UP_HYPOTHESIS_TASK_TYPE}
                    else "conclusion_generation",
                    "decision_case": normalized_assistant_payload.get("next_action")
                    if task_type == CONCLUSION_TASK_TYPE
                    else None,
                    "generation_mode": "evidence_grounded_prompt_supplement",
                    "request_id": request_id,
                },
            }
        )
    return normalized


def dedupe_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for sample in samples:
        fingerprint = make_sample_fingerprint(sample)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        sample["fingerprint"] = fingerprint
        unique.append(sample)
    return unique


def save_tool_splits(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    tool_dir = output_dir / "tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    all_records = splits["train"] + splits["val"] + splits["test"]
    save_jsonl(tool_dir / "all.jsonl", all_records)
    save_jsonl(tool_dir / "train.jsonl", splits["train"])
    save_jsonl(tool_dir / "val.jsonl", splits["val"])
    save_jsonl(tool_dir / "test.jsonl", splits["test"])


def build_request_record(
    *,
    request_id: str,
    task_type: str,
    evidence_docs: list[dict[str, Any]],
    evidence_mode: str | None,
    retrieval_query: str | None,
    retrieval_seed_doc_id: str | None,
    system_prompt: str,
    user_prompt: str,
    raw_text: str | None,
    parsed_ok: bool,
    accepted_samples: int,
    latency_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "task_type": task_type,
        "bucket": "tool",
        "evidence_doc_ids": [doc["id"] for doc in evidence_docs],
        "evidence_mode": evidence_mode,
        "retrieval_query": retrieval_query,
        "retrieval_seed_doc_id": retrieval_seed_doc_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_text": raw_text,
        "parsed_ok": parsed_ok,
        "accepted_samples": accepted_samples,
        "latency_seconds": latency_seconds,
        "error": error,
        "created_at": int(time.time()),
    }


def execute_generation_request(
    *,
    teacher_api_cfg: TeacherApiConfig,
    task_type: str,
    evidence_docs: list[dict[str, Any]],
    request_id: str,
    system_prompt: str,
    user_prompt: str,
    source_cfg: dict[str, Any],
    retrieval_query: str | None,
    retrieval_seed_doc_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed_ok = False
    raw_text: str | None = None
    accepted_for_request: list[dict[str, Any]] = []
    error: str | None = None
    started = time.time()

    try:
        raw_text, _ = call_teacher_api(
            teacher_api_cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        payload = parse_teacher_json(raw_text)
        accepted_for_request = validate_and_normalize_samples(
            payload,
            expected_task_type=task_type,
            evidence_docs=evidence_docs,
            request_id=request_id,
        )
        parsed_ok = True
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency_seconds = time.time() - started
    request_record = build_request_record(
        request_id=request_id,
        task_type=task_type,
        evidence_docs=evidence_docs,
        evidence_mode=source_cfg.get("evidence_mode"),
        retrieval_query=retrieval_query,
        retrieval_seed_doc_id=retrieval_seed_doc_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_text=raw_text,
        parsed_ok=parsed_ok,
        accepted_samples=len(accepted_for_request),
        latency_seconds=latency_seconds,
        error=error,
    )
    return request_record, accepted_for_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supplementary SFT data and optionally merge it into the main training dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sft_teacher_prompt_supplement.json",
    )
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument(
        "--only-task-type",
        type=str,
        choices=sorted(SUPPORTED_SUPPLEMENT_TASK_TYPES),
        default=None,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override supplement dataset output directory from config.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only generate supplement data; do not merge with the base teacher dataset.",
    )
    parser.add_argument(
        "--merge-base-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "sft_data" / "teacher_v2",
        help="Base dataset directory used when auto-merging after supplement generation.",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=Path,
        default=None,
        help="Merged dataset output directory. Default: <merge-base-dir>_plus_<supplement-dir-name>.",
    )
    parser.add_argument("--merge-train-ratio", type=float, default=0.9)
    parser.add_argument("--merge-val-ratio", type=float, default=0.05)
    parser.add_argument("--merge-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_generation_config(args.config)
    teacher_api_cfg = TeacherApiConfig(**config["teacher_api"])
    source_cfg = config["source"]
    dataset_cfg = config["dataset"]
    task_mix = config["task_mix"]

    if args.only_task_type:
        task_mix = {args.only_task_type: 1.0}

    target_total = args.target_total or int(dataset_cfg["target_total"])
    max_requests = args.max_requests or int(dataset_cfg["max_requests"])
    samples_per_request = int(dataset_cfg["samples_per_request"])
    seed = int(dataset_cfg["seed"])
    train_ratio = float(dataset_cfg["train_ratio"])
    val_ratio = float(dataset_cfg["val_ratio"])
    concurrency = max(1, int(args.concurrency or dataset_cfg.get("concurrency", 1)))

    if not os.environ.get(teacher_api_cfg.api_key_env):
        raise SystemExit(
            f"Missing teacher API key env var: {teacher_api_cfg.api_key_env}\n"
            f"Please export it first, for example:\n"
            f"  export {teacher_api_cfg.api_key_env}=<your_key>"
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / dataset_cfg["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "requests"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    requests_log_path = output_dir / "requests.jsonl"

    documents = load_story_documents(PROJECT_ROOT / source_cfg["documents_path"])
    rng = random.Random(seed)
    retriever = None
    if source_cfg.get("evidence_mode") == "retrieval":
        print(
            f"[startup] loading retrieval stack on device={args.device} ...",
            file=sys.stderr,
            flush=True,
        )
        try:
            retriever = load_retriever(device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(
                "Warning: failed to load retrieval stack; falling back to random evidence sampling.\n"
                f"Reason: {exc}",
                file=sys.stderr,
            )
            retriever = None
        else:
            print("[startup] retrieval stack ready.", file=sys.stderr, flush=True)

    accepted_samples = load_jsonl_if_exists(output_dir / "all.jsonl")
    request_records = load_jsonl_if_exists(requests_log_path)
    current_counts = Counter(sample.get("task_type") for sample in accepted_samples)
    target_counts = compute_target_counts(target_total, task_mix)
    next_request_number = len(request_records)
    requests_this_run = 0
    pending_sample_counts: Counter[str] = Counter()

    print(
        (
            f"[startup] output_dir={output_dir} target_total={target_total} "
            f"max_requests={max_requests} samples_per_request={samples_per_request} "
            f"concurrency={concurrency} existing_samples={len(accepted_samples)}"
        ),
        file=sys.stderr,
        flush=True,
    )

    progress = tqdm(total=target_total, initial=min(len(accepted_samples), target_total), desc="supplement")
    in_flight: dict[Future[tuple[dict[str, Any], list[dict[str, Any]]]], dict[str, Any]] = {}
    fatal_api_error: str | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while True:
            if fatal_api_error is not None:
                break
            while len(in_flight) < concurrency and requests_this_run < max_requests:
                projected_counts = current_counts + pending_sample_counts
                projected_total = len(accepted_samples) + sum(pending_sample_counts.values())
                if quotas_satisfied(projected_counts, target_counts) and projected_total >= target_total:
                    break

                task_type = choose_task_type(rng, task_mix, target_counts, projected_counts)
                retrieval_query = None
                retrieval_seed_doc_id = None

                if retriever is not None:
                    seed_doc = rng.choice(documents)
                    retrieval_seed_doc_id = seed_doc["id"]
                    retrieval_query = build_retrieval_seed_query(
                        seed_doc,
                        max_chars=int(source_cfg["seed_query_max_chars"]),
                    )
                    evidence_docs = retrieve_evidence_documents(
                        retriever,
                        query=retrieval_query,
                        top_k=int(source_cfg["retrieval_top_k"]),
                    )
                else:
                    evidence_docs = sample_evidence_documents(
                        documents,
                        rng,
                        max_docs=int(source_cfg["max_evidence_docs_per_request"]),
                    )

                request_id = f"req-{next_request_number:04d}"
                system_prompt, user_prompt = build_teacher_prompts(
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    samples_per_request=samples_per_request,
                )
                future = executor.submit(
                    execute_generation_request,
                    teacher_api_cfg=teacher_api_cfg,
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    request_id=request_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    source_cfg=source_cfg,
                    retrieval_query=retrieval_query,
                    retrieval_seed_doc_id=retrieval_seed_doc_id,
                )
                in_flight[future] = {
                    "task_type": task_type,
                    "request_id": request_id,
                }
                pending_sample_counts[task_type] += samples_per_request
                next_request_number += 1
                requests_this_run += 1
                progress.set_postfix(
                    submitted=requests_this_run,
                    in_flight=len(in_flight),
                    accepted=len(accepted_samples),
                )
                progress.refresh()
                print(
                    f"[submit] {request_id} task={task_type} evidence_docs={len(evidence_docs)}",
                    file=sys.stderr,
                    flush=True,
                )

            if not in_flight:
                break

            if len(accepted_samples) == 0 and request_records == []:
                print(
                    "[waiting] first request is running; progress bar will move after the first accepted sample is written.",
                    file=sys.stderr,
                    flush=True,
                )

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                request_meta = in_flight.pop(future)
                task_type = str(request_meta["task_type"])
                pending_sample_counts[task_type] -= samples_per_request
                if pending_sample_counts[task_type] <= 0:
                    pending_sample_counts.pop(task_type, None)

                try:
                    request_record, accepted_for_request = future.result()
                except Exception as exc:  # noqa: BLE001
                    request_id = str(request_meta["request_id"])
                    request_record = {
                        "request_id": request_id,
                        "task_type": task_type,
                        "bucket": "tool",
                        "evidence_doc_ids": [],
                        "evidence_mode": source_cfg.get("evidence_mode"),
                        "retrieval_query": None,
                        "retrieval_seed_doc_id": None,
                        "system_prompt": "",
                        "user_prompt": "",
                        "raw_text": None,
                        "parsed_ok": False,
                        "accepted_samples": 0,
                        "latency_seconds": 0.0,
                        "error": f"Unhandled worker error: {exc}",
                        "created_at": int(time.time()),
                    }
                    accepted_for_request = []

                request_records.append(request_record)
                save_jsonl(requests_log_path, request_records)
                (prompt_dir / f"{request_record['request_id']}.json").write_text(
                    json.dumps(request_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    (
                        f"[done] {request_record['request_id']} task={task_type} "
                        f"accepted={request_record['accepted_samples']} "
                        f"parsed_ok={request_record['parsed_ok']} "
                        f"latency={request_record['latency_seconds']:.2f}s"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if is_fatal_teacher_api_error(request_record.get("error")):
                    fatal_api_error = str(request_record["error"])
                    print(
                        (
                            "[fatal] teacher API returned an authorization/subscription error; "
                            "stop submitting more requests.\n"
                            f"reason: {fatal_api_error}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

                if accepted_for_request:
                    accepted_samples.extend(accepted_for_request)
                    accepted_samples = dedupe_samples(accepted_samples)
                    current_counts = Counter(sample.get("task_type") for sample in accepted_samples)
                    save_jsonl(output_dir / "all.jsonl", accepted_samples)
                    progress.n = min(len(accepted_samples), target_total)
                    progress.set_postfix(
                        submitted=requests_this_run,
                        in_flight=len(in_flight),
                        accepted=len(accepted_samples),
                    )
                    progress.refresh()

    progress.close()

    final_samples = accepted_samples[:target_total]
    final_samples = dedupe_samples(final_samples)
    save_jsonl(output_dir / "all.jsonl", final_samples)
    splits = split_samples(
        final_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    save_tool_splits(output_dir, splits)

    summary = {
        "output_dir": str(output_dir),
        "samples": len(final_samples),
        "task_type_distribution": dict(Counter(sample.get("task_type") for sample in final_samples)),
        "splits": {name: len(records) for name, records in splits.items()},
        "request_count_total": len(request_records),
        "request_count_this_run": requests_this_run,
        "concurrency": concurrency,
    }

    if not args.skip_merge:
        merge_base_dir = args.merge_base_dir.resolve()
        merged_output_dir = (
            args.merged_output_dir.resolve()
            if args.merged_output_dir is not None
            else default_merged_output_dir(merge_base_dir, output_dir)
        )
        merge_manifest = merge_datasets(
            base_dir=merge_base_dir,
            supplement_dir=output_dir,
            output_dir=merged_output_dir,
            train_ratio=args.merge_train_ratio,
            val_ratio=args.merge_val_ratio,
            seed=args.merge_seed,
        )
        summary["merge"] = {
            "base_dir": str(merge_base_dir),
            "output_dir": str(merged_output_dir),
            "merged_total_after_dedupe": merge_manifest["stats"]["merged_total_after_dedupe"],
            "split_sizes": merge_manifest["stats"]["split_sizes"],
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
