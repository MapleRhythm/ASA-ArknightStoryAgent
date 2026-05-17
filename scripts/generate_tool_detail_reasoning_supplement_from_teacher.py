#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib import error, request

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import (  # noqa: E402
    dedupe_samples,
    load_generation_config,
    make_sample_fingerprint,
    parse_teacher_json,
    split_samples,
)
from scripts.generate_prompt_supplement_from_teacher_merged import (  # noqa: E402
    merge_datasets_for_merged_flow,
)
from scripts.llama_factory.prepare_sft_dataset import main as prepare_llama_factory_main  # noqa: E402
import scripts.generate_tool_detail_reasoning_supplement as seed_data  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sft_teacher_prompt_supplement_merged_v1.json"
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "indexes" / "arknights_story" / "documents.jsonl"
DEFAULT_BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sft_data"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sft_data"
    / "tool_detail_reasoning_supplement_teacher_v1"
)
DEFAULT_MERGED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sft_data"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1"
)
DEFAULT_LLAMA_FACTORY_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llama_factory"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1"
)

INITIAL_HYPOTHESIS_TASK_TYPE = seed_data.INITIAL_HYPOTHESIS_TASK_TYPE
CONCLUSION_TASK_TYPE = seed_data.CONCLUSION_TASK_TYPE
HYPOTHESIS_SYSTEM = seed_data.HYPOTHESIS_SYSTEM
CONCLUSION_SYSTEM = seed_data.CONCLUSION_SYSTEM
HYPOTHESIS_INTENTS = seed_data.HYPOTHESIS_INTENTS
RETRIEVAL_ACTIONS = {"answer_directly", "retrieve_more", "clarify_user", "abstain"}

BADCASE_TRIGGER_GROUPS = {
    "causal_reasoning": ("为什么", "为何", "为了", "原因", "导致", "目的", "动机"),
    "plot_inference": ("真相", "真正", "其实", "原来", "背后", "幕后", "计划", "阴谋"),
    "fact_detail": ("发现", "得知", "身份", "本名", "关系", "主使", "暗杀", "刺杀", "关闭", "启动"),
}
BADCASE_GENERIC_ENTITY_MARKERS = (
    "？？？",
    "旁白",
    "众人",
    "士兵",
    "市民",
    "路人",
    "男人",
    "女人",
    "女孩",
    "男孩",
    "孩子",
    "老人",
    "青年",
    "感染者",
    "工作人员",
    "研究员",
    "警备",
    "声音",
    "歌声",
    "员工",
    "商人",
    "技术员",
    "银行",
    "工人",
    "村民",
    "村长",
    "镇民",
    "店员",
    "记者",
    "主持人",
    "主持",
    "那",
    "这",
    "每一",
    "一些",
    "？？",
)

CATEGORY_TARGETS = {
    "overview_too_shallow": 80,
    "causal_reasoning": 60,
    "plot_inference": 50,
    "fact_detail": 55,
    "user_correction_context": 55,
}

CATEGORY_DESCRIPTIONS = {
    "overview_too_shallow": (
        "当前证据只有概述、上位结论或局部片段，不能直接给浅答案；"
        "结论样本应选择 retrieve_more，并生成更具体的 follow_up_hypothesis。"
    ),
    "causal_reasoning": (
        "训练因果/动机问题：证据足够时直接给出完整因果链；证据不足时明确缺口并继续检索。"
    ),
    "plot_inference": (
        "训练剧情发展推理：回答要串联多个证据点，避免只复述一个片段。"
    ),
    "fact_detail": (
        "训练事实细节检索假设和细节回答：关键词必须包含实体、别名、事件物件和结果。"
    ),
    "user_correction_context": (
        "训练多轮追问/用户纠正：必须继承 dialogue_context 中的上轮主题，"
        "解析“这个/该计划/不是和X有关吗”等指代，并把新增线索加入检索假设。"
    ),
}

TASK_SCHEMA_TEXT = {
    "samples": [
        {
            "task_type": "user_question_hypothesis_generation 或 conclusion_generation",
            "category": "overview_too_shallow|causal_reasoning|plot_inference|fact_detail|user_correction_context",
            "difficulty": "medium|hard",
            "question": "用户问题",
            "dialogue_context": "多轮上下文；没有则为空字符串",
            "intent": "plot_fact|plot_reasoning|timeline|character_relation|event_summary|compare|persona_chat|out_of_scope",
            "entities": ["核心实体", "别名或关联实体"],
            "keywords": ["用于检索的关键词"],
            "expected_answer_type": "答案类型",
            "current_round": 1,
            "next_action": "answer_directly|retrieve_more|clarify_user|abstain；仅 conclusion_generation 必填",
            "answer": "仅 answer_directly/abstain 时填写；retrieve_more 时必须为空字符串",
            "missing_slots": ["仅 retrieve_more 时填写的具体信息缺口"],
            "clarification_question": "仅 clarify_user 时填写，否则为空字符串",
            "follow_up_hypothesis": {
                "question": "原问题",
                "entities": ["下一轮检索实体"],
                "keywords": ["下一轮检索关键词"],
                "expected_answer_type": "答案类型",
                "dialogue_context": "沿用/整理后的上下文",
            },
            "notes": "简短说明该样本训练的行为",
        }
    ]
}


@dataclass(frozen=True)
class TeacherJob:
    request_id: str
    category: str
    task_type: str
    case_slug: str
    question_topic: str
    evidence_docs: list[dict[str, Any]]
    reference_answer: str
    shallow_answer: str
    missing_slots: list[str]
    entities: list[str]
    keywords: list[str]
    expected_answer_type: str
    dialogue_context: str
    samples_per_request: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate targeted tool SFT supplement samples through a teacher-model API. "
            "The teacher generates semantic labels; this script only formats and validates runtime messages."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-output-dir", type=Path, default=DEFAULT_MERGED_OUTPUT_DIR)
    parser.add_argument("--llama-factory-output-dir", type=Path, default=DEFAULT_LLAMA_FACTORY_OUTPUT_DIR)
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--max-requests", type=int, default=160)
    parser.add_argument("--samples-per-request", type=int, default=3)
    parser.add_argument("--badcase-candidate-limit", type=int, default=240)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--endpoint-path", default="/v1/chat/completions")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--export-llama-factory", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_text(text: str, *, limit: int = 1100) -> str:
    return seed_data.clean_text(text, limit=limit)


def dedupe_keep_order(items: list[Any], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def scale_category_targets(target_total: int) -> dict[str, int]:
    total = sum(CATEGORY_TARGETS.values())
    if target_total == total:
        return dict(CATEGORY_TARGETS)
    scaled: dict[str, int] = {}
    running = 0
    items = list(CATEGORY_TARGETS.items())
    for index, (category, target) in enumerate(items):
        if index == len(items) - 1:
            value = target_total - running
        else:
            value = max(1, int(round(target * target_total / total)))
            running += value
        scaled[category] = value
    return scaled


def render_evidence_docs(docs: list[dict[str, Any]], *, limit_per_doc: int = 1050) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据{index}]",
                    f"id: {doc.get('id') or ''}",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or doc.get('story_id') or ''}",
                    f"stage_code: {doc.get('stage_code') or doc.get('stage_id') or ''}",
                    f"avg_tag: {doc.get('avg_tag') or ''}",
                    "clean_text:",
                    clean_text(str(doc.get("clean_text") or ""), limit=limit_per_doc),
                ]
            )
        )
    return "\n\n".join(blocks)


def current_hypothesis_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": str(spec["question"]).strip(),
        "intent": str(spec["intent"]).strip(),
        "entities": dedupe_keep_order(list(spec["entities"]), limit=12),
        "keywords": dedupe_keep_order(list(spec["keywords"]), limit=20),
        "expected_answer_type": str(spec["expected_answer_type"]).strip(),
        "dialogue_context": str(spec.get("dialogue_context") or "").strip(),
    }


def build_initial_user_message(spec: dict[str, Any]) -> str:
    context = str(spec.get("dialogue_context") or "").strip() or "无"
    return "\n".join(
        [
            f"用户问题: {spec['question']}",
            f"多轮上下文: {context}",
            "请生成初始假设文档 JSON。",
        ]
    )


def build_conclusion_user_message(spec: dict[str, Any], evidence_docs: list[dict[str, Any]]) -> str:
    context = str(spec.get("dialogue_context") or "").strip() or "无"
    current_round = max(1, int(spec.get("current_round") or 1))
    max_rounds = 4
    hypothesis = current_hypothesis_from_spec(spec)
    return "\n".join(
        [
            f"用户问题: {spec['question']}",
            f"多轮上下文: {context}",
            f"当前假设文档(JSON): {compact_json(hypothesis)}",
            "历史生成结果: [第1轮 hypothesis 已生成]",
            "历史检索上下文: [已围绕当前假设进行检索，获得当前证据]",
            f"当前检索轮次: 第{current_round}轮 / 最多{max_rounds}轮",
            "当前证据:",
            render_evidence_docs(evidence_docs, limit_per_doc=900),
            "请基于证据生成当前阶段结论 JSON。",
        ]
    )


def normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    return dedupe_keep_order(raw, limit=limit)


def normalize_follow_up(value: Any, *, fallback_question: str, fallback_context: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    question = str(value.get("question") or fallback_question).strip()
    entities = normalize_string_list(value.get("entities"), limit=12)
    keywords = normalize_string_list(value.get("keywords"), limit=20)
    expected_answer_type = str(value.get("expected_answer_type") or "").strip()
    dialogue_context = str(value.get("dialogue_context") or fallback_context or "").strip()
    if not question or not entities or not keywords or not expected_answer_type:
        return None
    return {
        "question": question,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def normalize_teacher_spec(raw: Any, *, expected_category: str, expected_task_type: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    task_type = str(raw.get("task_type") or expected_task_type).strip()
    category = str(raw.get("category") or expected_category).strip()
    if task_type != expected_task_type:
        return None
    if category != expected_category:
        return None

    question = str(raw.get("question") or "").strip()
    if len(question) < 6:
        return None
    intent = str(raw.get("intent") or "").strip()
    if intent not in HYPOTHESIS_INTENTS:
        return None
    entities = normalize_string_list(raw.get("entities"), limit=12)
    keywords = normalize_string_list(raw.get("keywords"), limit=20)
    expected_answer_type = str(raw.get("expected_answer_type") or "").strip()
    dialogue_context = str(raw.get("dialogue_context") or "").strip()
    if not question or not entities or not keywords or not expected_answer_type:
        return None

    normalized = {
        "task_type": task_type,
        "category": category,
        "difficulty": str(raw.get("difficulty") or "hard").strip() or "hard",
        "question": question,
        "dialogue_context": dialogue_context,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "notes": str(raw.get("notes") or "").strip(),
    }

    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        return normalized

    next_action = str(raw.get("next_action") or "").strip()
    if next_action not in RETRIEVAL_ACTIONS:
        return None
    answer = str(raw.get("answer") or "").strip()
    missing_slots = normalize_string_list(raw.get("missing_slots"), limit=8)
    clarification_question = str(raw.get("clarification_question") or "").strip()
    follow_up = normalize_follow_up(
        raw.get("follow_up_hypothesis"),
        fallback_question=question,
        fallback_context=dialogue_context,
    )

    if next_action == "retrieve_more":
        if answer or not missing_slots or follow_up is None:
            return None
        current_round = int(raw.get("current_round") or 1)
        if current_round >= 4:
            return None
    elif next_action == "clarify_user":
        if not clarification_question:
            return None
        missing_slots = []
        follow_up = None
    else:
        if not answer:
            return None
        missing_slots = []
        clarification_question = ""
        follow_up = None

    normalized.update(
        {
            "current_round": max(1, min(4, int(raw.get("current_round") or 2))),
            "next_action": next_action,
            "answer": answer,
            "missing_slots": missing_slots,
            "clarification_question": clarification_question,
            "follow_up_hypothesis": follow_up,
        }
    )
    return normalized


def assistant_payload_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["task_type"] == INITIAL_HYPOTHESIS_TASK_TYPE:
        return current_hypothesis_from_spec(spec)
    return {
        "question": spec["question"],
        "next_action": spec["next_action"],
        "answer": spec["answer"],
        "missing_slots": spec["missing_slots"],
        "clarification_question": spec["clarification_question"],
        "follow_up_hypothesis": spec["follow_up_hypothesis"],
    }


def metadata_from_docs(
    docs: list[dict[str, Any]],
    *,
    category: str,
    request_id: str,
    task_type: str,
    next_action: str | None,
    difficulty: str,
    notes: str,
) -> dict[str, Any]:
    story_ids = dedupe_keep_order(
        [doc.get("story_name") or doc.get("story_id") for doc in docs],
        limit=12,
    )
    stage_codes = dedupe_keep_order(
        [doc.get("stage_code") or doc.get("stage_id") for doc in docs],
        limit=12,
    )
    activity_names = dedupe_keep_order([doc.get("activity_name") for doc in docs], limit=12)
    return {
        "category": "tool",
        "grounded": True,
        "difficulty": difficulty,
        "notes": notes,
        "source_story_ids": story_ids,
        "source_stage_codes": stage_codes,
        "source_activity_names": activity_names,
        "task_family": "hypothesis_generation"
        if task_type == INITIAL_HYPOTHESIS_TASK_TYPE
        else "conclusion_generation_merged",
        "decision_case": next_action,
        "generation_mode": "teacher_api_tool_detail_reasoning_supplement_v1",
        "supplement_category": category,
        "request_id": request_id,
    }


def record_from_teacher_spec(
    spec: dict[str, Any],
    *,
    request_id: str,
    index: int,
    evidence_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    task_type = spec["task_type"]
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        system = HYPOTHESIS_SYSTEM
        user = build_initial_user_message(spec)
    else:
        system = CONCLUSION_SYSTEM
        user = build_conclusion_user_message(spec, evidence_docs)

    payload = assistant_payload_from_spec(spec)
    record = {
        "id": f"{request_id}-{task_type}-{index:04d}",
        "task_type": task_type,
        "bucket": "tool",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": compact_json(payload)},
        ],
        "tools": [],
        "meta": metadata_from_docs(
            evidence_docs,
            category=spec["category"],
            request_id=request_id,
            task_type=task_type,
            next_action=spec.get("next_action"),
            difficulty=spec["difficulty"],
            notes=spec.get("notes") or CATEGORY_DESCRIPTIONS.get(spec["category"], ""),
        ),
    }
    seed_data.validate_record(record)
    record["fingerprint"] = make_sample_fingerprint(record)
    return record


def existing_record_is_usable(record: dict[str, Any]) -> bool:
    try:
        seed_data.validate_record(record)
        payload = json.loads(record["messages"][-1]["content"])
    except Exception:
        return False
    question = str(payload.get("question") or "").strip()
    if len(question) < 6:
        return False
    return True


def build_system_prompt() -> str:
    return (
        "你是《明日方舟》剧情问答 Agent 的教师模型数据生成器。"
        "你只输出严格 JSON，不输出 markdown、解释或多余文本。"
        "你的输出将被直接转换成 SFT 数据，因此字段必须完整、可解析、无额外字段污染。"
    )


def build_user_prompt(job: TeacherJob) -> str:
    action_guidance = {
        INITIAL_HYPOTHESIS_TASK_TYPE: (
            "当前任务是生成初始 hypothesis 样本。只生成检索假设，不要直接回答问题。"
            "entities/keywords 要能帮助召回正文细节，不能只写宽泛词。"
        ),
        CONCLUSION_TASK_TYPE: (
            "当前任务是生成 conclusion 样本。必须判断证据是否足够："
            "如果证据只有概述或缺关键因果，next_action=retrieve_more，answer 必须为空，并生成 follow_up_hypothesis；"
            "如果证据足够，next_action=answer_directly，answer 必须完整回答本质/因果/细节，follow_up_hypothesis 必须为 null。"
        ),
    }[job.task_type]

    return "\n".join(
        [
            f"请生成 {job.samples_per_request} 条教师 API SFT 样本规格。",
            "",
            "输出顶层 JSON 只有 samples 字段，结构如下：",
            json.dumps(TASK_SCHEMA_TEXT, ensure_ascii=False, indent=2),
            "",
            "硬性规则：",
            f"1. 每条样本 task_type 必须严格等于 `{job.task_type}`。",
            f"2. 每条样本 category 必须严格等于 `{job.category}`。",
            "3. 不要输出训练 messages，本地脚本会包装 messages；你只输出语义规格字段。",
            "4. conclusion_generation 的 assistant 最终 JSON 只能包含 question、next_action、answer、missing_slots、clarification_question、follow_up_hypothesis。",
            "5. retrieve_more 时 answer 必须是空字符串，missing_slots 必须具体，follow_up_hypothesis 必须非空。",
            "6. answer_directly/abstain/clarify_user 时 follow_up_hypothesis 必须为 null。",
            "7. follow_up_hypothesis 只能包含 question、entities、keywords、expected_answer_type、dialogue_context。",
            "8. 用户纠正或追问场景必须继承 dialogue_context，不允许把“这个/该计划/这件事”当成实体。",
            "9. 不能生成泛泛答案，例如“证据不足以确认”“与某议员有关”这类没有本质细节的 answer_directly。",
            "10. 用户问题必须是完整自然问句，禁止从证据原文截取残句当问题，例如“眼下能做的”“接下来要做的”。",
            "11. 不要照抄 schema 示例；必须围绕下面 case 和证据生成。",
            "",
            f"类别说明: {CATEGORY_DESCRIPTIONS[job.category]}",
            f"任务说明: {action_guidance}",
            "",
            "Case 信息：",
            f"- case_slug: {job.case_slug}",
            f"- question_topic: {job.question_topic}",
            f"- 推荐实体: {'、'.join(job.entities)}",
            f"- 推荐关键词: {'、'.join(job.keywords)}",
            f"- expected_answer_type: {job.expected_answer_type}",
            f"- dialogue_context 建议: {job.dialogue_context or '无'}",
            f"- shallow_answer 示例（不要当充分答案）: {job.shallow_answer or '无'}",
            f"- reference_answer（证据足够时答案应达到的信息密度）: {job.reference_answer or '无'}",
            f"- missing_slots 建议: {'；'.join(job.missing_slots) if job.missing_slots else '无'}",
            "",
            "当前证据包：",
            render_evidence_docs(job.evidence_docs),
        ]
    )


def join_api_url(api_base: str, endpoint_path: str) -> str:
    return f"{api_base.rstrip('/')}/{endpoint_path.lstrip('/')}"


def call_teacher_chat_api(
    *,
    api_base: str,
    endpoint_path: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
    extra_headers: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    http_request = request.Request(
        join_api_url(api_base, endpoint_path),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Teacher API HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Teacher API URL error: {exc}") from exc

    response_payload = json.loads(raw_response)
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("Teacher API response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content, response_payload
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get("text") or item.get("content") or ""))
        joined = "\n".join(text for text in texts if text).strip()
        if joined:
            return joined, response_payload
    raise RuntimeError("Unsupported teacher API response content shape")


def execute_job(
    *,
    job: TeacherJob,
    api_base: str,
    endpoint_path: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
    extra_headers: dict[str, str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(job)
    started = time.time()
    raw_text: str | None = None
    raw_response: dict[str, Any] | None = None
    accepted: list[dict[str, Any]] = []
    parsed_ok = False
    error_text: str | None = None

    try:
        raw_text, raw_response = call_teacher_chat_api(
            api_base=api_base,
            endpoint_path=endpoint_path,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
            extra_headers=extra_headers,
        )
        payload = parse_teacher_json(raw_text)
        specs = payload.get("samples")
        if not isinstance(specs, list):
            raise ValueError("Teacher JSON must contain list field `samples`")
        parsed_ok = True
        for index, raw_spec in enumerate(specs):
            spec = normalize_teacher_spec(
                raw_spec,
                expected_category=job.category,
                expected_task_type=job.task_type,
            )
            if spec is None:
                continue
            try:
                accepted.append(
                    record_from_teacher_spec(
                        spec,
                        request_id=job.request_id,
                        index=index,
                        evidence_docs=job.evidence_docs,
                    )
                )
            except Exception:
                continue
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)

    request_record = {
        "request_id": job.request_id,
        "category": job.category,
        "task_type": job.task_type,
        "case_slug": job.case_slug,
        "question_topic": job.question_topic,
        "evidence_doc_ids": [doc.get("id") for doc in job.evidence_docs],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_text": raw_text,
        "raw_response": raw_response,
        "parsed_ok": parsed_ok,
        "accepted_samples": len(accepted),
        "latency_seconds": time.time() - started,
        "error": error_text,
        "created_at": int(time.time()),
    }
    return request_record, accepted


def docs_from_ids(by_id: dict[str, dict[str, Any]], ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    return [by_id[doc_id] for doc_id in ids if doc_id in by_id]


def make_manual_jobs(
    cases: list[Any],
    by_id: dict[str, dict[str, Any]],
    *,
    samples_per_request: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case in cases:
        shallow_docs = docs_from_ids(by_id, list(case.shallow_doc_ids))
        rich_docs = docs_from_ids(by_id, list(case.rich_doc_ids))
        all_docs = docs_from_ids(by_id, list(case.source_doc_ids))
        if not all_docs:
            continue
        context = f"user: {case.question_topic}是什么\nassistant: {case.shallow_answer}"
        base = {
            "case_slug": case.slug,
            "question_topic": case.question_topic,
            "reference_answer": case.detailed_answer,
            "shallow_answer": case.shallow_answer,
            "missing_slots": list(case.missing_slots),
            "entities": list(case.entities),
            "keywords": list(case.keywords),
            "expected_answer_type": case.expected_answer_type,
            "samples_per_request": samples_per_request,
        }
        jobs.extend(
            [
                {
                    **base,
                    "category": "overview_too_shallow",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": shallow_docs or all_docs[:2],
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "causal_reasoning",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": rich_docs or all_docs,
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "plot_inference",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": rich_docs or all_docs,
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "fact_detail",
                    "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                    "evidence_docs": all_docs,
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "user_correction_context",
                    "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                    "evidence_docs": all_docs,
                    "dialogue_context": context,
                },
                {
                    **base,
                    "category": "user_correction_context",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": rich_docs or all_docs,
                    "dialogue_context": context,
                },
            ]
        )
    return jobs


def make_auto_jobs(
    candidates: list[dict[str, Any]],
    *,
    samples_per_request: int,
    limit: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        summary_doc = candidate["summary_doc"]
        detail_docs = candidate["detail_docs"][:4]
        entities = list(candidate["entities"])
        topic = str(candidate["topic"])
        summary = clean_text(str(summary_doc.get("clean_text") or ""), limit=320)
        details = " ".join(clean_text(str(doc.get("clean_text") or ""), limit=220) for doc in detail_docs[:2])
        extra_terms = seed_data.extract_terms(details, limit=8)
        keywords = dedupe_keep_order(
            entities + extra_terms + ["具体经过", "前因后果", "主使者", "目的", "动机", "结果"],
            limit=20,
        )
        reference_answer = clean_text(details, limit=520)
        missing = ["具体经过", "关键人物", "动机或目的", "事件结果"]
        context = f"user: {topic}发生了什么\nassistant: {summary}"
        base = {
            "case_slug": f"auto_{summary_doc.get('id')}",
            "question_topic": topic,
            "reference_answer": reference_answer,
            "shallow_answer": summary,
            "missing_slots": missing,
            "entities": entities,
            "keywords": keywords,
            "expected_answer_type": "事件前因后果",
            "samples_per_request": samples_per_request,
        }
        jobs.extend(
            [
                {
                    **base,
                    "category": "overview_too_shallow",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": [summary_doc],
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "causal_reasoning",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": detail_docs,
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "fact_detail",
                    "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                    "evidence_docs": [summary_doc] + detail_docs,
                    "dialogue_context": "",
                },
                {
                    **base,
                    "category": "user_correction_context",
                    "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                    "evidence_docs": [summary_doc] + detail_docs,
                    "dialogue_context": context,
                },
                {
                    **base,
                    "category": "plot_inference",
                    "task_type": CONCLUSION_TASK_TYPE,
                    "evidence_docs": detail_docs,
                    "dialogue_context": "",
                },
            ]
        )
    return jobs


def _badcase_trigger_categories(text: str) -> list[str]:
    categories: list[str] = []
    for category, triggers in BADCASE_TRIGGER_GROUPS.items():
        if any(trigger in text for trigger in triggers):
            categories.append(category)
    return categories


def _badcase_question_topic(category: str, entities: list[str], doc: dict[str, Any]) -> str:
    primary = entities[0] if entities else str(doc.get("story_name") or doc.get("activity_name") or "这段剧情")
    secondary = entities[1] if len(entities) > 1 else ""
    if category == "causal_reasoning":
        return f"{primary}相关事件的原因和动机"
    if category == "plot_inference":
        return f"{primary}相关剧情的真相和后续影响"
    if category == "fact_detail":
        if secondary:
            return f"{primary}和{secondary}相关细节"
        return f"{primary}相关事实细节"
    return f"{primary}相关剧情"


def _filter_badcase_entities(entities: list[str], doc: dict[str, Any]) -> list[str]:
    filtered = [
        entity
        for entity in entities
        if not any(marker in entity for marker in BADCASE_GENERIC_ENTITY_MARKERS)
    ]
    if filtered:
        return filtered
    fallback = [
        str(doc.get("story_name") or "").strip(),
        str(doc.get("activity_name") or "").strip(),
        str(doc.get("stage_code") or "").strip(),
    ]
    return dedupe_keep_order(fallback, limit=5)


def _badcase_missing_slots(category: str) -> list[str]:
    if category == "causal_reasoning":
        return ["事件起因", "关键人物动机", "中间转折", "最终结果"]
    if category == "plot_inference":
        return ["真相或隐藏目的", "桥接证据", "人物选择", "后续影响"]
    if category == "fact_detail":
        return ["具体事实", "涉及人物", "关键行动", "证据出处"]
    return ["具体经过", "关键人物", "前因后果", "结果"]


def _source_path_is_story_text(doc: dict[str, Any]) -> bool:
    source_path = str(doc.get("source_path") or "")
    if "[uc]info" in source_path:
        return False
    if "gamedata/story" not in source_path and "/story/" not in source_path:
        return False
    return bool(str(doc.get("clean_text") or "").strip())


def make_corpus_badcase_jobs(
    docs: list[dict[str, Any]],
    *,
    rng: random.Random,
    samples_per_request: int,
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        if not _source_path_is_story_text(doc):
            continue
        source_path = str(doc.get("source_path") or doc.get("id") or "")
        grouped[source_path].append(doc)

    candidates: list[dict[str, Any]] = []
    seen_story_stage: set[tuple[str, str, str]] = set()
    for source_path, group in grouped.items():
        ordered = sorted(group, key=lambda item: str(item.get("id") or ""))
        if len(ordered) < 2:
            continue
        for index, doc in enumerate(ordered):
            text = str(doc.get("clean_text") or "")
            if len(text) < 120:
                continue
            categories = _badcase_trigger_categories(text)
            if not categories:
                continue
            story_key = (
                source_path,
                str(doc.get("story_name") or doc.get("story_id") or ""),
                str(doc.get("stage_code") or doc.get("stage_id") or ""),
            )
            if story_key in seen_story_stage:
                continue
            seen_story_stage.add(story_key)
            start = max(0, index - 1)
            end = min(len(ordered), index + 4)
            evidence_docs = ordered[start:end]
            entities = seed_data.extract_doc_entities(doc, evidence_docs)
            if not entities:
                entities = seed_data.extract_terms(text, limit=8)
            entities = _filter_badcase_entities(entities, doc)
            if not entities:
                continue
            extra_terms = seed_data.extract_terms(" ".join(str(item.get("clean_text") or "") for item in evidence_docs), limit=10)
            entity_list = _filter_badcase_entities(
                dedupe_keep_order(entities + extra_terms[:3], limit=10),
                doc,
            )
            candidates.append(
                {
                    "doc": doc,
                    "evidence_docs": evidence_docs,
                    "entities": entity_list,
                    "keywords": dedupe_keep_order(
                        entity_list
                        + extra_terms
                        + ["具体经过", "前因后果", "隐藏目的", "关键证据", "后续影响"],
                        limit=20,
                    ),
                    "categories": categories,
                }
            )

    rng.shuffle(candidates)
    jobs: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for candidate in candidates:
        if len(jobs) >= limit:
            break
        categories = sorted(
            candidate["categories"],
            key=lambda category: (category_counts[category], category),
        )
        category = categories[0]
        category_counts[category] += 1
        doc = candidate["doc"]
        evidence_docs = candidate["evidence_docs"]
        entities = candidate["entities"]
        keywords = candidate["keywords"]
        topic = _badcase_question_topic(category, entities, doc)
        joined_evidence = " ".join(
            clean_text(str(item.get("clean_text") or ""), limit=260) for item in evidence_docs[:3]
        )
        shallow = clean_text(str(doc.get("clean_text") or ""), limit=320)
        context = f"user: {topic}是什么\nassistant: {shallow}"
        base = {
            "case_slug": f"badcase_{doc.get('id')}",
            "question_topic": topic,
            "reference_answer": clean_text(joined_evidence, limit=650),
            "shallow_answer": shallow,
            "missing_slots": _badcase_missing_slots(category),
            "entities": entities,
            "keywords": keywords,
            "expected_answer_type": "事件前因后果" if category != "fact_detail" else "事实问答",
            "samples_per_request": samples_per_request,
        }
        jobs.append(
            {
                **base,
                "category": category,
                "task_type": CONCLUSION_TASK_TYPE
                if category in {"causal_reasoning", "plot_inference"}
                else INITIAL_HYPOTHESIS_TASK_TYPE,
                "evidence_docs": evidence_docs,
                "dialogue_context": "",
            }
        )
        if len(jobs) >= limit:
            break
        jobs.append(
            {
                **base,
                "category": "user_correction_context",
                "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                "evidence_docs": evidence_docs,
                "dialogue_context": context,
            }
        )
    return jobs


def instantiate_jobs(
    job_templates: list[dict[str, Any]],
    *,
    seed: int,
    max_requests: int,
    request_id_offset: int = 0,
) -> list[TeacherJob]:
    rng = random.Random(seed)
    templates = list(job_templates)
    rng.shuffle(templates)
    output: list[TeacherJob] = []
    for index, template in enumerate(templates[:max_requests]):
        output.append(
            TeacherJob(
                request_id=f"detail-teacher-{index + request_id_offset:04d}",
                category=str(template["category"]),
                task_type=str(template["task_type"]),
                case_slug=str(template["case_slug"]),
                question_topic=str(template["question_topic"]),
                evidence_docs=list(template["evidence_docs"]),
                reference_answer=str(template["reference_answer"]),
                shallow_answer=str(template["shallow_answer"]),
                missing_slots=list(template["missing_slots"]),
                entities=list(template["entities"]),
                keywords=list(template["keywords"]),
                expected_answer_type=str(template["expected_answer_type"]),
                dialogue_context=str(template.get("dialogue_context") or ""),
                samples_per_request=int(template["samples_per_request"]),
            )
        )
    return output


def write_dataset(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    splits = split_samples(records, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    save_jsonl(output_dir / "all.jsonl", records)
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    seed_data.save_bucket_splits(output_dir, splits)
    return splits


def export_llama_factory(source_dir: Path, output_dir: Path) -> None:
    old_argv = list(sys.argv)
    try:
        sys.argv = [
            "prepare_sft_dataset.py",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ]
        prepare_llama_factory_main()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = parse_args()
    config = load_generation_config(resolve_path(args.config))
    teacher_cfg = config.get("teacher_api") or {}
    api_base = args.api_base or str(teacher_cfg.get("base_url") or "https://api.svips.org")
    endpoint_path = args.endpoint_path
    api_key_env = args.api_key_env or str(teacher_cfg.get("api_key_env") or "TEACHER_API_KEY")
    model = args.model or str(teacher_cfg.get("model") or os.environ.get("MINIMAX_MODEL") or "")
    temperature = float(args.temperature if args.temperature is not None else teacher_cfg.get("temperature", 0.3))
    max_tokens = int(args.max_tokens if args.max_tokens is not None else max(6000, int(teacher_cfg.get("max_output_tokens", 4000))))
    timeout = float(args.timeout if args.timeout is not None else max(180, float(teacher_cfg.get("timeout_seconds", 120))))
    json_mode = not args.no_json_mode
    extra_headers = teacher_cfg.get("extra_headers") if isinstance(teacher_cfg.get("extra_headers"), dict) else None

    if not model:
        raise SystemExit("Missing teacher model. Pass --model or set it in config.")
    api_key = os.environ.get(api_key_env)
    if not args.dry_run and not api_key:
        raise SystemExit(
            f"Missing teacher API key env var: {api_key_env}\n"
            f"Set it first, for example:\n"
            f"  export {api_key_env}=<your_key>"
        )

    documents_path = resolve_path(args.documents)
    base_dir = resolve_path(args.base_dir)
    output_dir = resolve_path(args.output_dir)
    merged_output_dir = resolve_path(args.merged_output_dir)
    llama_factory_output_dir = resolve_path(args.llama_factory_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = output_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    requests_log_path = output_dir / "requests.jsonl"

    docs, by_id = seed_data.load_documents(documents_path)
    base_records = load_jsonl(base_dir / "all.jsonl")
    existing_fingerprints = {make_sample_fingerprint(record) for record in base_records}
    existing_samples = seed_data.load_jsonl(output_dir / "all.jsonl") if (output_dir / "all.jsonl").exists() else []
    existing_request_records = (
        seed_data.load_jsonl(requests_log_path) if requests_log_path.exists() else []
    )

    rng = random.Random(args.seed)
    manual_cases = seed_data.build_manual_cases(docs, by_id)
    auto_candidates = seed_data.build_overview_candidates(docs, by_id, rng=rng)
    job_templates = make_manual_jobs(
        manual_cases,
        by_id,
        samples_per_request=args.samples_per_request,
    )
    job_templates.extend(
        make_auto_jobs(
            auto_candidates,
            samples_per_request=args.samples_per_request,
            limit=max(20, args.max_requests),
        )
    )
    job_templates.extend(
        make_corpus_badcase_jobs(
            docs,
            rng=rng,
            samples_per_request=args.samples_per_request,
            limit=max(20, args.badcase_candidate_limit),
        )
    )
    jobs = instantiate_jobs(
        job_templates,
        seed=args.seed,
        max_requests=args.max_requests,
        request_id_offset=len(existing_request_records),
    )

    if args.dry_run:
        preview = {
            "dry_run": True,
            "jobs": len(jobs),
            "first_job": {
                "request_id": jobs[0].request_id,
                "category": jobs[0].category,
                "task_type": jobs[0].task_type,
                "case_slug": jobs[0].case_slug,
                "prompt_chars": len(build_user_prompt(jobs[0])),
            }
            if jobs
            else None,
            "api_base": api_base,
            "endpoint_path": endpoint_path,
            "model": model,
            "api_key_env": api_key_env,
        }
        save_json(output_dir / "dry_run_manifest.json", preview)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    accepted_samples = [
        record
        for record in existing_samples
        if existing_record_is_usable(record)
        and make_sample_fingerprint(record) not in existing_fingerprints
    ]
    accepted_samples = dedupe_samples(accepted_samples)
    current_seen = existing_fingerprints | {make_sample_fingerprint(record) for record in accepted_samples}
    category_targets = scale_category_targets(args.target_total)
    current_category_counts = Counter(
        str(record.get("meta", {}).get("supplement_category") or "unknown")
        for record in accepted_samples
    )

    def target_satisfied() -> bool:
        if len(accepted_samples) >= args.target_total:
            return True
        return all(current_category_counts.get(cat, 0) >= count for cat, count in category_targets.items())

    request_records = list(existing_request_records)
    completed_job_signatures = {
        (
            str(record.get("category") or ""),
            str(record.get("task_type") or ""),
            str(record.get("case_slug") or ""),
            str(record.get("question_topic") or ""),
        )
        for record in existing_request_records
        if record.get("parsed_ok") and int(record.get("accepted_samples") or 0) > 0
    }
    print(
        (
            f"[startup] target={args.target_total} existing={len(accepted_samples)} "
            f"jobs={len(jobs)} concurrency={args.concurrency} endpoint={join_api_url(api_base, endpoint_path)}"
        ),
        file=sys.stderr,
        flush=True,
    )

    progress = tqdm(
        total=args.target_total,
        initial=min(len(accepted_samples), args.target_total),
        desc="teacher_detail_supplement",
    )
    job_cursor = 0
    submitted = 0
    in_flight: dict[Future[tuple[dict[str, Any], list[dict[str, Any]]]], TeacherJob] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        while not target_satisfied():
            while (
                len(in_flight) < max(1, args.concurrency)
                and job_cursor < len(jobs)
                and not target_satisfied()
            ):
                job = jobs[job_cursor]
                job_cursor += 1
                job_signature = (job.category, job.task_type, job.case_slug, job.question_topic)
                if job_signature in completed_job_signatures:
                    continue
                if current_category_counts.get(job.category, 0) >= category_targets.get(job.category, 0):
                    continue
                future = executor.submit(
                    execute_job,
                    job=job,
                    api_base=api_base,
                    endpoint_path=endpoint_path,
                    api_key=str(api_key),
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    json_mode=json_mode,
                    extra_headers=extra_headers,
                )
                in_flight[future] = job
                submitted += 1
                print(
                    f"[submit] {job.request_id} category={job.category} task={job.task_type} case={job.case_slug}",
                    file=sys.stderr,
                    flush=True,
                )

            if not in_flight:
                break

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                job = in_flight.pop(future)
                request_record, records = future.result()
                request_records.append(request_record)
                save_jsonl(requests_log_path, request_records)
                save_json(requests_dir / f"{request_record['request_id']}.json", request_record)

                added = 0
                for record in records:
                    fp = make_sample_fingerprint(record)
                    category = str(record.get("meta", {}).get("supplement_category") or "unknown")
                    if fp in current_seen:
                        continue
                    if current_category_counts.get(category, 0) >= category_targets.get(category, 0):
                        continue
                    current_seen.add(fp)
                    record["fingerprint"] = fp
                    accepted_samples.append(record)
                    current_category_counts[category] += 1
                    added += 1
                    if len(accepted_samples) >= args.target_total:
                        break

                accepted_samples = dedupe_samples(accepted_samples)
                write_dataset(
                    output_dir=output_dir,
                    records=accepted_samples,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
                progress.n = min(len(accepted_samples), args.target_total)
                progress.set_postfix(
                    submitted=submitted,
                    accepted=len(accepted_samples),
                    added=added,
                    category=job.category,
                )
                progress.refresh()
                print(
                    (
                        f"[done] {job.request_id} accepted={request_record['accepted_samples']} "
                        f"added={added} parsed_ok={request_record['parsed_ok']} "
                        f"latency={request_record['latency_seconds']:.1f}s "
                        f"error={request_record['error'] or ''}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    progress.close()

    accepted_samples = dedupe_samples(accepted_samples[: args.target_total])
    splits = write_dataset(
        output_dir=output_dir,
        records=accepted_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    task_distribution = Counter(record.get("task_type") or "unknown" for record in accepted_samples)
    category_distribution = Counter(
        str(record.get("meta", {}).get("supplement_category") or "unknown")
        for record in accepted_samples
    )
    decision_distribution = Counter(
        str(record.get("meta", {}).get("decision_case") or "none")
        for record in accepted_samples
        if record.get("task_type") == CONCLUSION_TASK_TYPE
    )
    summary: dict[str, Any] = {
        "generator": "generate_tool_detail_reasoning_supplement_from_teacher",
        "created_at": int(time.time()),
        "api_base": api_base,
        "endpoint_path": endpoint_path,
        "model": model,
        "api_key_env": api_key_env,
        "documents": str(documents_path),
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "target_total": args.target_total,
        "samples": len(accepted_samples),
        "requests_submitted": submitted,
        "requests_total": len(request_records),
        "category_targets": category_targets,
        "task_type_distribution": dict(task_distribution),
        "supplement_category_distribution": dict(category_distribution),
        "decision_distribution": dict(decision_distribution),
        "splits": {name: len(records) for name, records in splits.items()},
    }

    if not args.no_merge:
        merge_manifest = merge_datasets_for_merged_flow(
            base_dir=base_dir,
            supplement_dir=output_dir,
            output_dir=merged_output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        summary["merge"] = {
            "output_dir": str(merged_output_dir),
            "merged_total_after_dedupe": merge_manifest["stats"]["merged_total_after_dedupe"],
            "split_sizes": merge_manifest["stats"]["split_sizes"],
        }

    if args.export_llama_factory:
        export_source = merged_output_dir if not args.no_merge else output_dir
        export_llama_factory(export_source, llama_factory_output_dir)
        summary["llama_factory"] = {
            "source_dir": str(export_source),
            "output_dir": str(llama_factory_output_dir),
        }

    save_json(output_dir / "manifest.json", summary)
    save_json(output_dir / "stats.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(accepted_samples) < args.target_total:
        print(
            f"WARNING: accepted {len(accepted_samples)} samples, below target {args.target_total}. "
            "Increase --max-requests or inspect requests/*.json errors.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
