#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.generate_prompt_supplement_from_teacher as legacy  # noqa: E402
from goldenglow.data.sft_teacher import (  # noqa: E402
    TeacherApiConfig,
    build_normalized_sample_id,
    call_teacher_api,
    dedupe_samples,
    load_generation_config,
    load_story_documents,
    make_sample_fingerprint,
    parse_teacher_json,
    split_samples,
)
from scripts.merge_sft_datasets import (  # noqa: E402
    _normalize_record,
    bucket_of,
    load_jsonl as load_merged_jsonl,
    save_bucket_splits,
    save_json,
    save_jsonl,
    with_dataset_source,
)


SUPPORTED_SUPPLEMENT_TASK_TYPES = {
    legacy.INITIAL_HYPOTHESIS_TASK_TYPE,
    legacy.CONCLUSION_TASK_TYPE,
}
INITIAL_HYPOTHESIS_TASK_TYPE = legacy.INITIAL_HYPOTHESIS_TASK_TYPE
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = legacy.FOLLOW_UP_HYPOTHESIS_TASK_TYPE
CONCLUSION_TASK_TYPE = legacy.CONCLUSION_TASK_TYPE
ACTION_CHOICES = legacy.ACTION_CHOICES
CONCLUSION_SCHEMA_FIELDS = (
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sft_teacher_prompt_supplement_merged_v1.json"
DEFAULT_MERGE_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "sft_data" / "teacher_v2"
ROUND_INFO_RE = re.compile(r"当前检索轮次:\s*第(\d+)轮\s*/\s*最多(\d+)轮")
SAFE_ENTITY_PATTERN_RE = re.compile(
    r"(?:代号|本名|名为|名字是|叫做|叫我|我是)\s*[，,:： ]*\s*([\u4e00-\u9fff]{2,8})"
)
SAFE_ENTITY_TEXT_RE = re.compile(r"[\u4e00-\u9fff]{2,10}")
ENTITY_STOP_WORDS = {
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
    "干员语音",
    "干员档案",
    "干员报到",
    "信赖触摸",
    "信赖提升后交谈",
    "剧情",
    "故事",
    "问题",
    "干员",
    "这位干员",
    "角色",
    "人物",
    "关系",
    "身份",
    "来历",
    "真相",
    "原因",
    "动机",
    "页面触发",
    "引导",
    "页面",
    "拼装",
    "测试了",
    "通过者",
    "追寻之旅",
    "这艘船",
    "战地指挥官",
}
ENTITY_STOP_SUBSTRINGS = (
    "页面触发",
    "引导",
    "教程",
    "测试",
    "交谈",
    "信赖",
    "触摸",
    "问候",
    "报到",
    "语音",
    "档案",
    "这位",
)
ENTITY_BAD_SUFFIXES = ("了", "的", "吗", "呢", "吧", "啊")
GENERIC_ENTITY_WORDS = {
    "干员",
    "这位干员",
    "角色",
    "人物",
    "对方",
    "此人",
    "那个人",
}
ABSTRACT_ENTITY_HINTS = (
    "干员",
    "身份",
    "关系",
    "角色",
    "经历",
    "过去",
    "背景",
    "来历",
    "原因",
    "动机",
    "影子",
    "东西",
    "目标",
    "物种",
    "类型",
)


def _expected_answer_type_is_compatible(question: str, intent: str, expected_answer_type: str) -> bool:
    q = str(question or "").strip()
    i = str(intent or "").strip()
    a = str(expected_answer_type or "").strip()
    if not q or not a:
        return False
    if "为什么" in q or "为了什么" in q:
        return a == "原因/动机"
    if "什么关系" in q or i == "character_relation":
        return a in {"关系问答", "身份关系"}
    if "说了什么" in q or "会说什么" in q:
        return a in {"角色台词", "事实问答"}
    if "提到了哪些内容" in q or "讲了什么" in q or "有什么经历" in q:
        return a in {"事件概要", "事实问答", "事件行为"}
    if "寻找什么" in q or "扮演什么角色" in q:
        return a in {"事件行为", "事实问答"}
    if "是什么身份" in q or "真实身份是什么" in q or "族裔背景是什么" in q or "是什么生物" in q:
        return a in {"身份关系", "事实问答"}
    return True


def _contains_abstract_entity_hint(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return True
    if token.startswith(("这位", "那个", "某个", "某位")):
        return True
    return any(hint in token for hint in ABSTRACT_ENTITY_HINTS)


def _collect_grounding_texts(
    *,
    evidence_docs: list[dict[str, Any]],
    user_text: str,
    current_hypothesis: dict[str, Any] | None,
) -> list[str]:
    texts = [str(user_text or "")]
    if isinstance(current_hypothesis, dict):
        texts.append(json.dumps(current_hypothesis, ensure_ascii=False))
        for entity in current_hypothesis.get("entities") or []:
            if isinstance(entity, str):
                texts.append(entity)
    for doc in evidence_docs:
        for key in ("activity_name", "story_name", "stage_name", "chapter_name", "zone_name", "clean_text"):
            value = doc.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            for key in ("speaker", "text"):
                value = segment.get(key)
                if isinstance(value, str) and value:
                    texts.append(value)
    return texts


def _is_grounded_follow_up_entity(
    entity: str,
    *,
    evidence_docs: list[dict[str, Any]],
    user_text: str,
    current_hypothesis: dict[str, Any] | None,
    candidate_entities: list[str],
) -> bool:
    token = str(entity or "").strip()
    if not _is_safe_entity_candidate(token) or _is_generic_entity(token):
        return False
    if token in candidate_entities:
        return True
    grounding_texts = _collect_grounding_texts(
        evidence_docs=evidence_docs,
        user_text=user_text,
        current_hypothesis=current_hypothesis,
    )
    if any(token in text for text in grounding_texts):
        return True
    if _contains_abstract_entity_hint(token):
        return False
    return False


def _is_safe_entity_candidate(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if token in ENTITY_STOP_WORDS:
        return False
    if any(token.endswith(suffix) for suffix in ENTITY_BAD_SUFFIXES):
        return False
    if any(marker in token for marker in ENTITY_STOP_SUBSTRINGS):
        return False
    if len(token) < 2 or len(token) > 12:
        return False
    if not SAFE_ENTITY_TEXT_RE.fullmatch(token):
        return False
    return True


def _is_generic_entity(value: str) -> bool:
    token = str(value or "").strip()
    return not token or token in GENERIC_ENTITY_WORDS


def extract_primary_entity_candidates(
    evidence_docs: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[str]:
    counts: Counter[str] = Counter()

    def add_candidate(token: str, weight: int) -> None:
        if _is_safe_entity_candidate(token):
            counts[token] += weight

    for doc in evidence_docs:
        for key, weight in (
            ("story_name", 4),
            ("activity_name", 3),
            ("stage_name", 3),
            ("chapter_name", 2),
            ("zone_name", 2),
        ):
            value = doc.get(key)
            if isinstance(value, str):
                add_candidate(value, weight)

        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            add_candidate(speaker, 6)

        clean_text = str(doc.get("clean_text") or "")[:240]
        for match in SAFE_ENTITY_PATTERN_RE.findall(clean_text):
            add_candidate(match, 4)

    return [name for name, _ in counts.most_common(limit)]


def build_initial_hypothesis_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
    avoid_questions: list[str] | None = None,
) -> tuple[str, str]:
    primary_entity_candidates = extract_primary_entity_candidates(evidence_docs)
    example_question, example_entities, example_keywords, example_answer_type = _derive_example_hypothesis_fields(
        evidence_docs
    )
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
                f"用户问题: {example_question}\n"
                "多轮上下文: 无\n"
                "请生成初始假设文档 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                legacy.build_latest_hypothesis_schema_example(
                    question=example_question,
                    entities=example_entities,
                    keywords=example_keywords,
                    expected_answer_type=example_answer_type,
                ),
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
        "6. intent 只能从以下集合中选择：" + "、".join(sorted(legacy.HYPOTHESIS_INTENTS)) + "。",
        "7. user prompt 应围绕“用户问题 + 多轮上下文 -> 初始假设文档 JSON”。",
        "8. assistant JSON 目标是服务检索，不是直接回答问题；不得把最终结论当作既定事实写死。",
        "9. 不要编造英文别名、Dr. 前缀、代号扩写、罗德岛职位推断或跨角色别名污染。",
        "10. 所有文本使用中文。",
        "11. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "12. 每条样本都必须围绕 1 到 2 个主实体构造问题；assistant JSON 的 `entities` 第一个元素必须是主实体。",
        "13. `keywords` 必须包含主实体，不要生成不带锚点实体的宽泛检索词。",
        "14. assistant JSON 的主实体必须来自“主实体候选”或当前证据文本中已经出现的实体，不得借用 schema 示例里的“烛煌”等占位实体。",
        "15. 如果当前证据缺少可靠实体锚点，只能围绕证据中明确出现的活动名、故事名、角色名提问，不得凭空创造角色。",
        "16. 不得直接照搬 schema 示例里的 question 文本，必须基于当前证据重新构造问题。",
        "17. assistant JSON 的 `entities` 不允许使用“干员”“这位干员”“角色”“人物”等泛指词作为实体。",
        "18. 如果问题是关系类问题（例如包含“X和Y是什么关系”或 intent = character_relation），`entities` 必须至少包含两个明确实体，且问题中的双方都要保留。",
        "19. `expected_answer_type` 必须和问题形式匹配：身份类问题用“身份关系”；原因类问题用“原因/动机”；关系类问题用“关系问答”或“身份关系”；台词类问题用“角色台词”；内容概括类问题用“事件概要”或“事实问答”。",
        "20. 特别注意：像“在事件中扮演什么角色/负责什么/做了什么/起了什么作用”这类问题，不要把 `expected_answer_type` 写成“身份关系”。",
    ]

    if avoid_questions:
        requirements.append("21. 不要重复生成下列近期高频问题：" + "；".join(avoid_questions) + "。")

    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“初始假设文档生成”训练样本。\n\n"
        "建议主实体候选（优先围绕这些角色/称谓出题）:\n"
        + ("、".join(primary_entity_candidates) if primary_entity_candidates else "无")
        + "\n\n"
        "当前项目唯一合法的初始 hypothesis schema:\n"
        + legacy.build_latest_hypothesis_schema_block()
        + "\n\n"
        "字段含义说明:\n"
        + legacy.build_initial_hypothesis_field_explanations()
        + "\n\nexpected_answer_type 选择规则:\n"
        + build_expected_answer_type_guidelines()
        + "\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + legacy.format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def _extract_round_info(user_text: str) -> tuple[int | None, int | None]:
    match = ROUND_INFO_RE.search(user_text)
    if not match:
        return None, None
    try:
        current_round = int(match.group(1))
        max_rounds = int(match.group(2))
    except ValueError:
        return None, None
    return current_round, max_rounds


def _primary_entity_is_grounded(
    primary_entity: str,
    *,
    evidence_docs: list[dict[str, Any]],
    candidate_entities: list[str],
) -> bool:
    if primary_entity in candidate_entities:
        return True
    for doc in evidence_docs:
        for key in ("activity_name", "story_name", "stage_name", "chapter_name", "zone_name", "clean_text"):
            value = doc.get(key)
            if isinstance(value, str) and primary_entity in value:
                return True
    return False


def _extract_current_hypothesis_payload(user_text: str) -> dict[str, Any] | None:
    marker = "当前假设文档(JSON):"
    start = user_text.find(marker)
    if start < 0:
        return None
    tail = user_text[start + len(marker) :].lstrip()
    if not tail.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(tail):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(tail[: index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _question_requires_two_entities(question: str, intent: str) -> bool:
    normalized_question = str(question or "").strip()
    normalized_intent = str(intent or "").strip()
    return normalized_intent == "character_relation" or (
        "和" in normalized_question and "关系" in normalized_question
    )


def _derive_example_hypothesis_fields(
    evidence_docs: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], str]:
    candidate_entities = extract_primary_entity_candidates(evidence_docs)
    if candidate_entities:
        primary = candidate_entities[0]
        return (
            f"{primary}是什么身份？",
            [primary],
            [primary, "身份", "来历", "背景"],
            "身份关系",
        )

    for doc in evidence_docs:
        for key in ("story_name", "activity_name", "stage_name"):
            value = str(doc.get(key) or "").strip()
            if _is_safe_entity_candidate(value):
                return (
                    f"{value}讲了什么？",
                    [value],
                    [value, "剧情", "内容"],
                    "事实问答",
                )

    return (
        "这段剧情讲了什么？",
        ["当前剧情"],
        ["当前剧情", "内容", "经过"],
        "事实问答",
    )


def _build_question_frequency_map(samples: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    question_counts: dict[str, Counter[str]] = {
        INITIAL_HYPOTHESIS_TASK_TYPE: Counter(),
        CONCLUSION_TASK_TYPE: Counter(),
    }
    for sample in samples:
        task_type = str(sample.get("task_type") or "")
        if task_type not in question_counts:
            continue
        messages = sample.get("messages") or []
        if not isinstance(messages, list):
            continue
        final_assistant = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and str(message.get("content") or "").strip()
            ),
            None,
        )
        if final_assistant is None:
            continue
        payload = legacy._parse_json_content(final_assistant.get("content"))
        if not isinstance(payload, dict):
            continue
        question = str(payload.get("question") or "").strip()
        if question:
            question_counts[task_type][question] += 1
    return question_counts


def _build_avoid_questions(
    question_counts: Counter[str],
    *,
    max_question_repeats: int,
    limit: int = 12,
) -> list[str]:
    blocked = [
        question
        for question, count in question_counts.most_common()
        if count >= max_question_repeats
    ]
    return blocked[:limit]


def build_expected_answer_type_guidelines() -> str:
    return "\n".join(
        [
            "- `expected_answer_type` 只描述答案形态，不等同于 `intent`。",
            "- 即使 `intent = plot_fact`，`expected_answer_type` 也不一定是“身份关系”；要按问题本身决定。",
            "- “X是什么身份 / 真实身份是什么 / 族裔背景是什么 / 是什么生物” -> 优先用 `身份关系`。",
            "- “X为什么 / 为了什么 / 动机是什么” -> 用 `原因/动机`。",
            "- “X和Y是什么关系” -> 用 `关系问答`，必要时也可用 `身份关系`。",
            "- “说了什么 / 会说什么 / 台词是什么” -> 用 `角色台词` 或 `事实问答`。",
            "- “提到了哪些内容 / 讲了什么 / 有什么经历” -> 用 `事件概要`、`事实问答` 或 `事件行为`。",
            "- “在事件中扮演什么角色 / 负责什么 / 做了什么 / 起了什么作用” -> 优先用 `事件行为` 或 `事实问答`，不要写成 `身份关系`。",
        ]
    )


def build_merged_conclusion_schema_example(
    *,
    question: str,
    entities: list[str],
    keywords: list[str],
    expected_answer_type: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "next_action": "retrieve_more",
        "answer": "",
        "missing_slots": ["更直接的人物身份证据", "与主实体直接相关的桥接信息"],
        "clarification_question": "",
        "follow_up_hypothesis": legacy.build_follow_up_hypothesis_schema_example(
            question=question,
            entities=entities,
            keywords=keywords,
            expected_answer_type=expected_answer_type,
        ),
    }


def build_merged_conclusion_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题。",
            "- `next_action`: 当前证据下的下一步动作，只能是 answer_directly、retrieve_more、clarify_user、abstain。",
            "- `answer`: 当前阶段结论文本。answer_directly 或 abstain 时必须非空；retrieve_more 时必须为空字符串。",
            "- `missing_slots`: 当前证据还缺哪些具体可检索的信息缺口，主要在 retrieve_more 时使用。",
            "- `clarification_question`: 当问题本身有歧义时，向用户发出的澄清问题；仅 clarify_user 时必须非空。",
            "- `follow_up_hypothesis`: 当 next_action = retrieve_more 时必须给出下一轮检索 hypothesis；否则必须为 null。",
        ]
    )


def build_merged_conclusion_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
    avoid_questions: list[str] | None = None,
) -> tuple[str, str]:
    example_question, example_entities, example_keywords, example_answer_type = _derive_example_hypothesis_fields(
        evidence_docs
    )
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 合并版结论步骤的高质量 SFT 样本。"
        "当前只生成“结论生成（内嵌 follow_up hypothesis）”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )

    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 conclusion_generator。"},
        {
            "role": "user",
            "content": (
                f"用户问题: {example_question}\n多轮上下文: 无\n"
                "当前假设文档(JSON): "
                + json.dumps(
                    legacy.build_latest_hypothesis_schema_example(
                        question=example_question,
                        entities=example_entities,
                        keywords=example_keywords,
                        expected_answer_type=example_answer_type,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                "历史生成结果: [第1轮 hypothesis 已生成，但当前证据还不足以直接回答]\n"
                "历史检索上下文: [第1轮已围绕主实体检索，但仍缺更直接证据]\n"
                "当前检索轮次: 第2轮 / 最多3轮\n"
                "当前证据: [...]\n请基于证据生成当前阶段结论 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_merged_conclusion_schema_example(
                    question=example_question,
                    entities=example_entities,
                    keywords=example_keywords,
                    expected_answer_type=example_answer_type,
                ),
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
        "4. assistant JSON 必须严格使用：question、next_action、answer、missing_slots、clarification_question、follow_up_hypothesis。",
        "5. next_action 只能是 `answer_directly`、`retrieve_more`、`clarify_user`、`abstain`。",
        "6. user prompt 必须包含：用户问题、多轮上下文、当前假设文档(JSON)、历史生成结果、历史检索上下文、当前检索轮次、当前证据；当前假设文档不能是空对象 {}。",
        "7. `answer_directly` 或 `abstain` 时，answer 必须非空，follow_up_hypothesis 必须为 null。",
        "8. `clarify_user` 时 clarification_question 必须非空，follow_up_hypothesis 必须为 null。",
        "9. `retrieve_more` 时 answer 必须为空字符串，missing_slots 必须为具体可检索缺口，follow_up_hypothesis 必须非空。",
        "10. follow_up_hypothesis 只能包含：question、entities、keywords、expected_answer_type、dialogue_context。",
        "11. follow_up_hypothesis 不能包含 intent；它必须继承上一轮 intent。",
        "12. retrieve_more 样本里 follow_up_hypothesis 的 `entities` 第一个元素必须保留主实体，不允许丢锚点。",
        "13. 结论生成样本必须显式体现“基于当前证据是否足够作答”的判断。",
        "14. 所有文本使用中文，不要编造英文别名或 Dr. 前缀。",
        "15. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "16. 如果当前检索轮次还未达到最后一轮，且证据不足，优先选择 retrieve_more，不允许过早 abstain。",
        "17. 只有在已达到最大检索轮次、问题明确超出证据覆盖范围、或确认无法继续缩小检索方向时，才允许 abstain。",
        "18. 如果当前检索轮次已经达到最后一轮，则不允许再输出 retrieve_more。",
        "19. 不得直接照搬 schema 示例里的 question 文本，必须基于当前证据重新构造问题。",
        "20. 当前假设文档中的主实体不得是“干员”“这位干员”“角色”“人物”等泛指词；如果是这类泛指，应判定为坏 hypothesis，不要围绕它生成样本。",
        "21. 如果问题本身是关系类问题，retrieve_more 时生成的 follow_up_hypothesis 必须保留双方实体，不允许只保留其中一方。",
        "22. retrieve_more 时，follow_up_hypothesis 新增的实体或桥接词必须已经出现在当前证据、历史检索上下文或当前假设文档中；不得发明“医疗干员”“某个影子”这类抽象标签充当实体。",
        "23. `expected_answer_type` 必须和问题形式匹配：身份类问题用“身份关系”；原因类问题用“原因/动机”；关系类问题用“关系问答”或“身份关系”；台词类问题用“角色台词”；内容概括类问题用“事件概要”或“事实问答”。",
        "24. 特别注意：像“在事件中扮演什么角色/负责什么/做了什么/起了什么作用”这类问题，当前 hypothesis 和 follow_up_hypothesis 的 `expected_answer_type` 都不要写成“身份关系”。",
    ]

    if avoid_questions:
        requirements.append("25. 不要重复生成下列近期高频问题：" + "；".join(avoid_questions) + "。")

    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“合并版结论生成”训练样本。\n\n"
        "当前项目唯一合法的初始 hypothesis schema:\n"
        + legacy.build_latest_hypothesis_schema_block()
        + "\n\n"
        "当前项目唯一合法的 follow-up hypothesis schema:\n"
        + legacy.build_follow_up_hypothesis_schema_block()
        + "\n\n"
        "当前项目唯一合法的 merged conclusion schema:\n"
        + json.dumps(
            build_merged_conclusion_schema_example(
                question=example_question,
                entities=example_entities,
                keywords=example_keywords,
                expected_answer_type=example_answer_type,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n字段含义说明:\n"
        + build_merged_conclusion_field_explanations()
        + "\n\nexpected_answer_type 选择规则:\n"
        + build_expected_answer_type_guidelines()
        + "\n\n要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + legacy.format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_teacher_prompts(
    *,
    task_type: str,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
    avoid_questions: list[str] | None = None,
) -> tuple[str, str]:
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        return build_initial_hypothesis_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
            avoid_questions=avoid_questions,
        )
    if task_type == CONCLUSION_TASK_TYPE:
        return build_merged_conclusion_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
            avoid_questions=avoid_questions,
        )
    raise ValueError(f"Unsupported merged supplement task type: {task_type}")


def _normalize_merged_conclusion_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    required_fields = (
        "question",
        "next_action",
        "answer",
        "missing_slots",
        "clarification_question",
    )
    if any(field not in payload for field in required_fields):
        return None
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    next_action = str(payload.get("next_action") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    missing_slots = legacy._normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()
    follow_up_hypothesis = payload.get("follow_up_hypothesis")

    if not question or next_action not in ACTION_CHOICES:
        return None
    if next_action in {"answer_directly", "abstain"} and not answer:
        return None
    if next_action == "clarify_user" and not clarification_question:
        return None
    if next_action == "retrieve_more":
        if answer or not missing_slots:
            return None
        if not isinstance(follow_up_hypothesis, dict):
            return None
        normalized_follow_up = legacy._normalize_follow_up_hypothesis_assistant_payload(
            follow_up_hypothesis
        )
        if normalized_follow_up is None:
            return None
        follow_up_hypothesis = normalized_follow_up
    else:
        missing_slots = []
        if next_action != "clarify_user":
            clarification_question = ""
        follow_up_hypothesis = None

    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
        "follow_up_hypothesis": follow_up_hypothesis,
    }


def _is_valid_initial_hypothesis_sample(
    normalized_payload: dict[str, Any],
    *,
    evidence_docs: list[dict[str, Any]],
    user_text: str,
) -> bool:
    candidate_entities = extract_primary_entity_candidates(evidence_docs)
    if not candidate_entities:
        return False

    entities = normalized_payload.get("entities") or []
    if not entities:
        return False
    primary_entity = str(entities[0]).strip()
    if not _is_safe_entity_candidate(primary_entity) or _is_generic_entity(primary_entity):
        return False
    if any(_is_generic_entity(entity) for entity in entities):
        return False
    if not _primary_entity_is_grounded(
        primary_entity,
        evidence_docs=evidence_docs,
        candidate_entities=candidate_entities,
    ):
        return False
    if _question_requires_two_entities(
        str(normalized_payload.get("question") or ""),
        str(normalized_payload.get("intent") or ""),
    ):
        if len(entities) < 2:
            return False
        secondary_entity = str(entities[1]).strip()
        if not _is_safe_entity_candidate(secondary_entity) or _is_generic_entity(secondary_entity):
            return False
        if secondary_entity not in str(normalized_payload.get("question") or ""):
            return False

    question = str(normalized_payload.get("question") or "").strip()
    if not question or "烛煌" in question and not _primary_entity_is_grounded(
        "烛煌",
        evidence_docs=evidence_docs,
        candidate_entities=candidate_entities,
    ):
        return False
    if primary_entity not in question and primary_entity not in user_text:
        return False
    if not _expected_answer_type_is_compatible(
        question,
        str(normalized_payload.get("intent") or ""),
        str(normalized_payload.get("expected_answer_type") or ""),
    ):
        return False
    return True


def _is_valid_conclusion_decision(
    normalized_payload: dict[str, Any],
    *,
    evidence_docs: list[dict[str, Any]],
    user_text: str,
) -> bool:
    current_hypothesis = _extract_current_hypothesis_payload(user_text)
    if isinstance(current_hypothesis, dict):
        entities = current_hypothesis.get("entities")
        if isinstance(entities, list) and entities:
            primary_entity = str(entities[0]).strip()
            if _is_generic_entity(primary_entity):
                return False
    current_round, max_rounds = _extract_round_info(user_text)
    next_action = str(normalized_payload.get("next_action") or "").strip()
    if current_round is not None and max_rounds is not None:
        if next_action == "abstain" and current_round < max_rounds:
            return False
        if next_action == "retrieve_more" and current_round >= max_rounds:
            return False
    if next_action == "retrieve_more":
        follow_up_hypothesis = normalized_payload.get("follow_up_hypothesis")
        if not isinstance(follow_up_hypothesis, dict):
            return False
        follow_up_entities = follow_up_hypothesis.get("entities") or []
        if not follow_up_entities:
            return False
        candidate_entities = extract_primary_entity_candidates(evidence_docs)
        if isinstance(current_hypothesis, dict):
            current_entities = current_hypothesis.get("entities") or []
            if current_entities:
                current_primary = str(current_entities[0]).strip()
                if str(follow_up_entities[0]).strip() != current_primary:
                    return False
            current_question = str(current_hypothesis.get("question") or normalized_payload.get("question") or "").strip()
            current_intent = str(current_hypothesis.get("intent") or "").strip()
        else:
            current_question = str(normalized_payload.get("question") or "").strip()
            current_intent = ""
        for entity in follow_up_entities:
            if not _is_grounded_follow_up_entity(
                str(entity),
                evidence_docs=evidence_docs,
                user_text=user_text,
                current_hypothesis=current_hypothesis,
                candidate_entities=candidate_entities,
            ):
                return False
        if _question_requires_two_entities(current_question, current_intent) and len(follow_up_entities) < 2:
            return False
        if not _expected_answer_type_is_compatible(
            current_question,
            current_intent,
            str(follow_up_hypothesis.get("expected_answer_type") or ""),
        ):
            return False
    return True


def validate_and_normalize_samples(
    payload: dict[str, Any],
    *,
    expected_task_type: str,
    evidence_docs: list[dict[str, Any]],
    request_id: str,
    existing_question_counts: Counter[str] | None = None,
    max_question_repeats: int = 3,
) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Teacher payload must contain a list field named 'samples'")

    source_story_ids = [doc.get("story_id") for doc in evidence_docs if doc.get("story_id")]
    source_stage_codes = [doc.get("stage_code") for doc in evidence_docs if doc.get("stage_code")]
    source_activity_names = [doc.get("activity_name") for doc in evidence_docs if doc.get("activity_name")]
    question_counts = Counter(existing_question_counts or {})

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

        user_message = next((message for message in clean_messages if message["role"] == "user"), None)
        if user_message is None:
            continue
        if task_type == CONCLUSION_TASK_TYPE and legacy._contains_empty_current_hypothesis(user_message["content"]):
            continue
        if legacy._contains_legacy_prompt_hypothesis_schema(user_message["content"]):
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

        assistant_payload = legacy._parse_json_content(final_assistant.get("content"))
        if assistant_payload is None:
            continue

        normalized_assistant_payload: dict[str, Any] | None = None
        if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
            normalized_assistant_payload = legacy._normalize_initial_hypothesis_assistant_payload(
                assistant_payload
            )
        elif task_type == CONCLUSION_TASK_TYPE:
            normalized_assistant_payload = _normalize_merged_conclusion_assistant_payload(
                assistant_payload
            )
        if normalized_assistant_payload is None:
            continue
        if task_type == INITIAL_HYPOTHESIS_TASK_TYPE and not _is_valid_initial_hypothesis_sample(
            normalized_assistant_payload,
            evidence_docs=evidence_docs,
            user_text=user_message["content"],
        ):
            continue
        if task_type == CONCLUSION_TASK_TYPE and not _is_valid_conclusion_decision(
            normalized_assistant_payload,
            evidence_docs=evidence_docs,
            user_text=user_message["content"],
        ):
            continue
        question = str(normalized_assistant_payload.get("question") or "").strip()
        if question and question_counts.get(question, 0) >= max_question_repeats:
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
                    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE
                    else "conclusion_generation_merged",
                    "decision_case": normalized_assistant_payload.get("next_action")
                    if task_type == CONCLUSION_TASK_TYPE
                    else None,
                    "generation_mode": "evidence_grounded_prompt_supplement_merged",
                    "request_id": request_id,
                },
            }
        )
        if question:
            question_counts[question] += 1
    return normalized


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
    return legacy.build_request_record(
        request_id=request_id,
        task_type=task_type,
        evidence_docs=evidence_docs,
        evidence_mode=evidence_mode,
        retrieval_query=retrieval_query,
        retrieval_seed_doc_id=retrieval_seed_doc_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_text=raw_text,
        parsed_ok=parsed_ok,
        accepted_samples=accepted_samples,
        latency_seconds=latency_seconds,
        error=error,
    )


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
    existing_question_counts: Counter[str] | None,
    max_question_repeats: int,
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
            existing_question_counts=existing_question_counts,
            max_question_repeats=max_question_repeats,
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


def merge_datasets_for_merged_flow(
    *,
    base_dir: Path,
    supplement_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    base_dir = base_dir.resolve()
    supplement_dir = supplement_dir.resolve()
    output_dir = output_dir.resolve()

    base_records = with_dataset_source(load_merged_jsonl(base_dir / "all.jsonl"), base_dir.name)
    supplement_records = with_dataset_source(load_merged_jsonl(supplement_dir / "all.jsonl"), supplement_dir.name)

    def is_merged_flow_compatible(record: dict[str, Any]) -> bool:
        task_type = str(record.get("task_type") or "")
        if task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
            return False
        if task_type != CONCLUSION_TASK_TYPE:
            return True
        messages = record.get("messages") or []
        if not isinstance(messages, list):
            return False
        final_assistant = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and str(message.get("content") or "").strip()
            ),
            None,
        )
        if final_assistant is None:
            return False
        assistant_payload = legacy._parse_json_content(final_assistant.get("content"))
        if not isinstance(assistant_payload, dict):
            return False
        if str(assistant_payload.get("next_action") or "").strip() != "retrieve_more":
            return True
        return isinstance(assistant_payload.get("follow_up_hypothesis"), dict)

    filtered_base_records = [record for record in base_records if is_merged_flow_compatible(record)]
    filtered_supplement_records = [record for record in supplement_records if is_merged_flow_compatible(record)]

    normalization_stats: Counter[str] = Counter()
    cleaned_base_records = [
        normalized
        for record in filtered_base_records
        if (normalized := _normalize_record(record, source_name=base_dir.name, stats=normalization_stats))
        is not None
    ]
    cleaned_supplement_records = [
        normalized
        for record in filtered_supplement_records
        if (
            normalized := _normalize_record(
                record,
                source_name=supplement_dir.name,
                stats=normalization_stats,
            )
        )
        is not None
    ]

    merged_records = dedupe_samples(cleaned_base_records + cleaned_supplement_records)
    splits = split_samples(
        merged_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    save_jsonl(output_dir / "all.jsonl", merged_records)
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    save_bucket_splits(output_dir, splits)

    task_distribution = Counter(record.get("task_type") or "unknown" for record in merged_records)
    category_distribution = Counter(bucket_of(record) for record in merged_records)
    manifest = {
        "generator": "generate_prompt_supplement_from_teacher_merged",
        "base_dir": str(base_dir),
        "supplement_dir": str(supplement_dir),
        "output_dir": str(output_dir),
        "stats": {
            "base_total": len(base_records),
            "supplement_total": len(supplement_records),
            "base_total_after_flow_filter": len(filtered_base_records),
            "supplement_total_after_flow_filter": len(filtered_supplement_records),
            "base_incompatible_filtered": len(base_records) - len(filtered_base_records),
            "supplement_incompatible_filtered": len(supplement_records) - len(filtered_supplement_records),
            "base_total_after_cleaning": len(cleaned_base_records),
            "supplement_total_after_cleaning": len(cleaned_supplement_records),
            "merged_total_before_dedupe": len(cleaned_base_records) + len(cleaned_supplement_records),
            "merged_total_after_dedupe": len(merged_records),
            "split_sizes": {name: len(records) for name, records in splits.items()},
            "task_type_distribution": dict(task_distribution),
            "category_distribution": dict(category_distribution),
            "normalization": dict(normalization_stats),
        },
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "stats.json", manifest["stats"])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate merged-flow supplementary SFT data and optionally merge it into the main training dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--merge-base-dir", type=Path, default=DEFAULT_MERGE_BASE_DIR)
    parser.add_argument("--merged-output-dir", type=Path, default=None)
    parser.add_argument("--merge-train-ratio", type=float, default=0.9)
    parser.add_argument("--merge-val-ratio", type=float, default=0.05)
    parser.add_argument("--merge-seed", type=int, default=42)
    parser.add_argument("--max-question-repeats", type=int, default=3)
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
    rng = legacy.random.Random(seed)
    retriever = None
    if source_cfg.get("evidence_mode") == "retrieval":
        print(
            f"[startup] loading retrieval stack on device={args.device} ...",
            file=sys.stderr,
            flush=True,
        )
        try:
            retriever = legacy.load_retriever(device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(
                "Warning: failed to load retrieval stack; falling back to random evidence sampling.\n"
                f"Reason: {exc}",
                file=sys.stderr,
            )
            retriever = None
        else:
            print("[startup] retrieval stack ready.", file=sys.stderr, flush=True)

    accepted_samples = legacy.load_jsonl_if_exists(output_dir / "all.jsonl")
    request_records = legacy.load_jsonl_if_exists(requests_log_path)
    current_counts = Counter(sample.get("task_type") for sample in accepted_samples)
    question_counts_by_task = _build_question_frequency_map(accepted_samples)
    target_counts = legacy.compute_target_counts(target_total, task_mix)
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

    progress = tqdm(total=target_total, initial=min(len(accepted_samples), target_total), desc="supplement_merged")
    in_flight: dict[Future[tuple[dict[str, Any], list[dict[str, Any]]]], dict[str, Any]] = {}
    fatal_api_error: str | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while True:
            if fatal_api_error is not None:
                break
            while len(in_flight) < concurrency and requests_this_run < max_requests:
                projected_counts = current_counts + pending_sample_counts
                projected_total = len(accepted_samples) + sum(pending_sample_counts.values())
                if legacy.quotas_satisfied(projected_counts, target_counts) and projected_total >= target_total:
                    break

                task_type = legacy.choose_task_type(rng, task_mix, target_counts, projected_counts)
                retrieval_query = None
                retrieval_seed_doc_id = None

                if retriever is not None:
                    seed_doc = rng.choice(documents)
                    retrieval_seed_doc_id = seed_doc["id"]
                    retrieval_query = legacy.build_retrieval_seed_query(
                        seed_doc,
                        max_chars=int(source_cfg["seed_query_max_chars"]),
                    )
                    evidence_docs = legacy.retrieve_evidence_documents(
                        retriever,
                        query=retrieval_query,
                        top_k=int(source_cfg["retrieval_top_k"]),
                    )
                else:
                    evidence_docs = legacy.sample_evidence_documents(
                        documents,
                        rng,
                        max_docs=int(source_cfg["max_evidence_docs_per_request"]),
                    )

                if (
                    task_type == INITIAL_HYPOTHESIS_TASK_TYPE
                    and not extract_primary_entity_candidates(evidence_docs)
                ):
                    continue

                avoid_questions = _build_avoid_questions(
                    question_counts_by_task.get(task_type, Counter()),
                    max_question_repeats=max(1, args.max_question_repeats),
                )

                request_id = f"req-{next_request_number:04d}"
                system_prompt, user_prompt = build_teacher_prompts(
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    samples_per_request=samples_per_request,
                    avoid_questions=avoid_questions,
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
                    existing_question_counts=question_counts_by_task.get(task_type, Counter()).copy(),
                    max_question_repeats=max(1, args.max_question_repeats),
                )
                in_flight[future] = {"task_type": task_type, "request_id": request_id}
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
                if legacy.is_fatal_teacher_api_error(request_record.get("error")):
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
                    question_counts_by_task = _build_question_frequency_map(accepted_samples)
                    save_jsonl(output_dir / "all.jsonl", accepted_samples)
                    progress.n = min(len(accepted_samples), target_total)
                    progress.set_postfix(
                        submitted=requests_this_run,
                        in_flight=len(in_flight),
                        accepted=len(accepted_samples),
                    )
                    progress.refresh()

    progress.close()

    final_samples = dedupe_samples(accepted_samples[:target_total])
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
    legacy.save_tool_splits(output_dir, splits)

    summary = {
        "generator": "generate_prompt_supplement_from_teacher_merged",
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
            else legacy.default_merged_output_dir(merge_base_dir, output_dir)
        )
        merge_manifest = merge_datasets_for_merged_flow(
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
            "base_incompatible_filtered": merge_manifest["stats"]["base_incompatible_filtered"],
            "supplement_incompatible_filtered": merge_manifest["stats"]["supplement_incompatible_filtered"],
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
