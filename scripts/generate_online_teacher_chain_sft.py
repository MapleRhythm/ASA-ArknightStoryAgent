#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from threading import Lock
import time
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import EMBEDDING_MODEL_DIR, INDEX_ROOT, QueryConfig, RERANKER_MODEL_DIR  # noqa: E402
from goldenglow.data.sft_teacher import (  # noqa: E402
    TeacherApiConfig,
    call_teacher_api,
    parse_teacher_json,
)
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    ConclusionResult,
    HypothesisDocument,
    build_retrieval_query,
    classify_retrieval_query_mode,
    merge_hypotheses,
    merge_ranked_hits,
    normalize_conclusion_payload,
    normalize_hypothesis_payload,
    render_evidence_blocks,
    rerank_hits,
    select_prompt_evidence,
    summarize_evidence_for_trace,
    validate_conclusion_grounding,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.build_short_prompt_sft_dataset import compact_text  # noqa: E402


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_current_short_prompt_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_online_chain_short_prompt_v1"
DEFAULT_MINIRAG_INDEX = PROJECT_ROOT / "indexes/arknights_story_minirag_v3/graph.json"
JSONL_WRITE_LOCK = Lock()
LOG_LOCK = Lock()
GROUNDING_MODES = ("strict", "soft", "off")

STUDENT_SYSTEM = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
QUESTION_RE = re.compile(r"^(?:question|用户问题|用户原问题)\s*[:：]\s*(.+?)\s*$")
CONTEXT_RE = re.compile(r"^dialogue_context\s*[:：]\s*(.+?)\s*$")
GENERIC_VOICE_TAGS = (
    "信赖触摸",
    "信赖提升后交谈",
    "行动失败",
    "行动出发",
    "选中干员",
    "任命助理",
    "任命队长",
    "编入队伍",
    "闲置",
    "戳一下",
    "问候",
)
IDENTITY_OR_REASON_WORDS = (
    "身份",
    "关系",
    "为什么",
    "为何",
    "原因",
    "动机",
    "目的",
    "真相",
    "身世",
    "来历",
)
TOKEN_RE = re.compile(r"[\u4e00-\u9fff·]{2,16}|[A-Za-z][A-Za-z0-9_.-]{1,31}")
NOISY_ANCHORS = {
    "什么",
    "为什么",
    "怎么",
    "如何",
    "身份",
    "关系",
    "原因",
    "剧情",
    "故事",
    "角色",
    "用户问题",
    "明日方舟",
}
BAD_QUESTION_PATTERNS = (
    "片段",
    "上述",
    "下列",
    "以下",
    "证据",
    "检索",
    "chunk",
    "Chunk",
    "CHUNK",
    "行动失败",
    "行动开始",
    "编入队伍",
    "选中干员",
    "信赖触摸",
    "信赖提升后交谈",
    "任命助理",
    "任命队长",
    "闲置",
    "戳一下",
    "临床诊断分析",
    "综合体检测试",
    "高难行动",
    "行动结束",
    "结束行动",
    "战斗失败",
    "生日时",
    "生日语音",
    "标题是什么身份",
    "标题和明日方舟",
    "进驻设施是什么身份",
    "生日是什么身份",
)
ALLOWED_COMPLEX_QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
COMPLEX_QUERY_TYPE_ALIASES = {
    "timeline": "fact",
    "compare": "reasoning",
    "comparison": "reasoning",
    "plot_reasoning": "reasoning",
    "plot_fact": "fact",
}
QUESTION_MAX_CHARS = 64
GENERIC_KEYWORDS = {
    "原因",
    "目的",
    "动机",
    "关系",
    "信息",
    "问题",
    "影响",
    "情况",
    "经过",
    "发生",
    "剧情",
    "分析",
    "背景",
    "资料",
    "内容",
    "角色",
    "人物",
    "故事",
    "具体",
}
WEAK_FOLLOW_UP_NEW_TERMS = GENERIC_KEYWORDS | {
    "暗示",
    "冲突",
    "计划",
    "后果",
    "描述",
    "互动",
    "用途",
    "解释",
    "细节",
    "过程",
    "线索",
    "真相",
    "补充",
    "关键",
    "事件",
    "相关",
    "直接",
    "间接",
    "说明",
    "观点",
    "看法",
    "态度",
    "回应",
}
BAD_RETRIEVAL_TERM_MARKERS = (
    "为什么",
    "为何",
    "什么",
    "如何",
    "怎么",
    "是否",
    "有没有",
    "哪",
    "吗",
    "？",
    "?",
    "片段",
    "上述",
    "证据",
    "检索",
    "chunk",
    "用户问题",
    "这件事",
    "这种情况",
    "这个过程",
)
KALTSIT_FORBIDDEN_QUESTION_TERMS = (
    "凯尔希",
    "凯尔希医生",
    "凯尔希所长",
    "凯尔希·思衡托",
)
KALTSIT_INTERNAL_ALIAS = "凯尔希·思衡托"
KALTSIT_NATURAL_NAME = "凯尔希"
NULLISH_FIELD_STRINGS = {"", "none", "null", "nil", "n/a", "na", "无", "无。", "没有", "不需要"}
GENERIC_MISSING_SLOT_PATTERNS = (
    "更多信息",
    "更多证据",
    "完整剧情",
    "具体背景",
    "相关资料",
    "深层含义",
    "进一步信息",
    "上下文信息",
)
BANNED_MISSING_SLOT_VALUES = GENERIC_KEYWORDS | {
    "更多信息",
    "更多证据",
    "完整剧情",
    "具体背景",
    "相关资料",
    "深层含义",
    "进一步信息",
    "上下文信息",
}
SLOT_ALIGNMENT_KEYWORDS = (
    "原因",
    "动机",
    "目的",
    "身份",
    "关系",
    "真相",
    "对话",
    "建议",
    "除名",
    "前因",
    "后果",
    "立场",
    "经过",
    "时间",
    "地点",
    "组织",
    "桥接",
    "矛盾",
    "冲突",
    "行动",
    "证据",
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with JSONL_WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def log_progress(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_key(*parts: str) -> str:
    text = "\n".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def is_nullish_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULLISH_FIELD_STRINGS or value.strip() in NULLISH_FIELD_STRINGS
    return False


def normalize_optional_string(value: Any) -> str:
    if is_nullish_field(value):
        return ""
    return str(value).strip()


def sanitize_dialogue_context(value: Any) -> str:
    text = normalize_optional_string(value)
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if any(marker in compact for marker in ("片段", "上述", "下列", "证据", "检索", "chunk", "Chunk", "CHUNK")):
        return ""
    return text


def sanitize_teacher_conclusion_payload(payload: dict[str, Any], *, max_round_reached: bool) -> dict[str, Any]:
    cleaned = dict(payload)
    conclusion_fields = {
        "question",
        "next_action",
        "answer",
        "missing_slots",
        "clarification_question",
        "follow_up_hypothesis",
    }
    if "next_action" not in cleaned:
        return cleaned
    cleaned = {key: value for key, value in cleaned.items() if key in conclusion_fields}
    action = str(cleaned.get("next_action") or "").strip()
    cleaned["clarification_question"] = normalize_optional_string(cleaned.get("clarification_question", ""))
    if is_nullish_field(cleaned.get("missing_slots")):
        cleaned["missing_slots"] = []
    if is_nullish_field(cleaned.get("follow_up_hypothesis")):
        cleaned["follow_up_hypothesis"] = None
    if action in {"answer_directly", "abstain", "clarify_user"}:
        cleaned["follow_up_hypothesis"] = None
    if action != "clarify_user":
        cleaned["clarification_question"] = ""
    if action == "abstain" and not normalize_optional_string(cleaned.get("answer", "")):
        if max_round_reached:
            cleaned["answer"] = "现有检索证据不足以确认该问题，且已达到检索轮次上限。"
        else:
            cleaned["answer"] = "现有检索证据不足以确认该问题。"
    return cleaned


def as_follow_up_payload(hypothesis: HypothesisDocument) -> dict[str, Any]:
    payload = asdict(hypothesis)
    payload.pop("intent", None)
    return payload


def as_initial_payload(hypothesis: HypothesisDocument) -> dict[str, Any]:
    return asdict(hypothesis)


def as_conclusion_payload(conclusion: ConclusionResult, *, question: str) -> dict[str, Any]:
    return {
        "question": question,
        "next_action": conclusion.next_action,
        "answer": conclusion.answer,
        "missing_slots": conclusion.missing_slots,
        "clarification_question": conclusion.clarification_question,
        "follow_up_hypothesis": (
            as_follow_up_payload(conclusion.follow_up_hypothesis)
            if conclusion.follow_up_hypothesis is not None
            else None
        ),
    }


def extract_conversations(record: dict[str, Any]) -> list[dict[str, str]]:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        return conversations
    messages = record.get("messages")
    if isinstance(messages, list):
        converted = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                converted.append({"from": "human", "value": str(message.get("content") or "")})
            elif role == "assistant":
                converted.append({"from": "gpt", "value": str(message.get("content") or "")})
        return converted
    return []


def first_user_text(record: dict[str, Any]) -> str:
    for message in extract_conversations(record):
        if message.get("from") in {"human", "user"}:
            return str(message.get("value") or message.get("content") or "")
    return ""


def extract_question_and_context(prompt: str) -> tuple[str, str]:
    question = ""
    dialogue_context = ""
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        question_match = QUESTION_RE.match(line)
        if question_match and not question:
            question = question_match.group(1).strip()
            continue
        context_match = CONTEXT_RE.match(line)
        if context_match and not dialogue_context:
            value = context_match.group(1).strip()
            dialogue_context = "" if value == "无" else value
    if not question:
        for pattern in (r"用户问题[:：]\s*(.+)", r"用户原问题[:：]\s*(.+)"):
            match = re.search(pattern, prompt)
            if match:
                question = match.group(1).strip()
                break
    return question, dialogue_context


def collect_source_questions(
    source_dir: Path,
    *,
    splits: list[str],
    source_task_types: set[str],
    limit: int | None,
    seed: int,
    filter_bad_questions: bool,
) -> list[dict[str, str]]:
    if limit is not None and limit <= 0:
        return []
    candidates: dict[str, dict[str, str]] = {}
    for split in splits:
        path = source_dir / f"{split}.json"
        if not path.exists():
            continue
        records = load_json(path)
        if not isinstance(records, list):
            continue
        for record in records:
            task_type = str(record.get("task_type") or "")
            if source_task_types and task_type not in source_task_types:
                continue
            question, dialogue_context = extract_question_and_context(first_user_text(record))
            if len(question) < 4:
                continue
            if filter_bad_questions and is_bad_question(question):
                continue
            dialogue_context = sanitize_dialogue_context(dialogue_context)
            key = stable_key(question, dialogue_context)
            candidates.setdefault(
                key,
                {
                    "question_key": key,
                    "question": question,
                    "dialogue_context": dialogue_context,
                    "source_split": split,
                    "source_task_type": task_type,
                },
            )
    items = list(candidates.values())
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:limit] if limit is not None and limit > 0 else items


def is_bad_question(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return True
    if len(text) > QUESTION_MAX_CHARS:
        return True
    if any(pattern in text for pattern in BAD_QUESTION_PATTERNS):
        return True
    if any(marker in text for marker in ("chunk", "level_", "activities/", "obt/", "{", "}", "[", "]")):
        return True
    if text.count("？") + text.count("?") > 1:
        return True
    if any(marker in text for marker in ("这件事", "这种情况", "这个过程", "这些人", "他们", "她们", "它们")):
        return True
    return False


def is_kaltsit_related_question(question: str, entities: list[Any] | None = None) -> bool:
    text = str(question or "")
    if entities:
        text += " " + " ".join(str(item) for item in entities)
    return any(term in text for term in KALTSIT_FORBIDDEN_QUESTION_TERMS)


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def normalize_teacher_entity_or_keyword(value: str, *, field: str) -> str:
    item = str(value or "").strip()
    if not item:
        return ""
    if KALTSIT_INTERNAL_ALIAS in item:
        if field == "keywords":
            return KALTSIT_NATURAL_NAME
        if item == KALTSIT_INTERNAL_ALIAS:
            return KALTSIT_NATURAL_NAME
        item = item.replace(KALTSIT_INTERNAL_ALIAS, KALTSIT_NATURAL_NAME)
    return item


def normalize_teacher_field_items(value: str, *, field: str) -> list[str]:
    item = normalize_teacher_entity_or_keyword(value, field=field)
    if not item:
        return []
    if field == "keywords" and re.search(r"\s+", item):
        parts = [
            normalize_teacher_entity_or_keyword(part, field=field)
            for part in re.split(r"\s+", item)
            if part.strip()
        ]
        clean_parts = [
            part
            for part in parts
            if part and len(part) >= 2 and not is_bad_retrieval_term(part, field=field)
        ]
        # Teacher occasionally emits "A B C" as one keyword. Split those
        # because the runtime treats keywords as atomic retrieval terms.
        if len(clean_parts) >= 2:
            return clean_parts
        return []
    return [item]


def is_bad_retrieval_term(value: str, *, field: str) -> bool:
    item = str(value or "").strip()
    if not item:
        return True
    if field == "keywords" and re.search(r"\s+", item):
        return True
    compact = re.sub(r"\s+", "", item)
    if compact in GENERIC_KEYWORDS:
        return True
    if any(marker in compact for marker in BAD_RETRIEVAL_TERM_MARKERS):
        return True
    if field == "entities" and (
        len(compact) > 14
        or any(marker in compact for marker in ("原因", "目的", "动机", "关系", "影响", "态度", "看法"))
    ):
        return True
    if field == "keywords" and len(compact) > 24:
        return True
    return False


def sanitize_teacher_alias_fields(payload: Any) -> Any:
    if isinstance(payload, list):
        return [sanitize_teacher_alias_fields(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"entities", "keywords"} and isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, (str, int, float)):
                    normalized.extend(normalize_teacher_field_items(str(item), field=key))
            output[key] = [
                item
                for item in dedupe_keep_order(normalized)
                if item and not is_bad_retrieval_term(item, field=key)
            ]
            continue
        output[key] = sanitize_teacher_alias_fields(value)
    return output


def sanitize_hypothesis_document(hypothesis: HypothesisDocument) -> HypothesisDocument:
    entities = [
        item
        for item in dedupe_keep_order(
            [
                term
                for item in hypothesis.entities
                for term in normalize_teacher_field_items(item, field="entities")
            ]
        )
        if item and not is_bad_retrieval_term(item, field="entities")
    ][:8]
    keywords = [
        item
        for item in dedupe_keep_order(
            [
                term
                for item in hypothesis.keywords
                for term in normalize_teacher_field_items(item, field="keywords")
            ]
        )
        if item and not is_bad_retrieval_term(item, field="keywords")
    ][:12]
    if not entities:
        raise ValueError("hypothesis rejected: no valid entities after cleanup")
    if not keywords:
        raise ValueError("hypothesis rejected: no valid keywords after cleanup")
    return HypothesisDocument(
        question=hypothesis.question,
        intent=hypothesis.intent,
        query_type=hypothesis.query_type,
        entities=entities,
        keywords=keywords,
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=sanitize_dialogue_context(hypothesis.dialogue_context),
    )


def question_anchors(question: str, hypothesis: HypothesisDocument | None = None) -> list[str]:
    terms: list[str] = []
    if hypothesis is not None:
        terms.extend(hypothesis.entities)
        terms.extend(hypothesis.keywords[:6])
    terms.extend(TOKEN_RE.findall(question))
    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = str(term).strip()
        if len(term) < 2 or term in NOISY_ANCHORS or term in seen:
            continue
        seen.add(term)
        output.append(term)
        if len(output) >= 12:
            break
    return output


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence:
        doc = item.get("document") or {}
        parts.extend(
            [
                str(doc.get("id") or ""),
                str(doc.get("activity_name") or ""),
                str(doc.get("story_name") or ""),
                str(doc.get("stage_code") or ""),
                str(doc.get("avg_tag") or ""),
                str(item.get("evidence_chain_text") or ""),
                str(doc.get("clean_text") or ""),
            ]
        )
    return "\n".join(parts)


def evidence_mentions_anchor(question: str, hypothesis: HypothesisDocument, evidence: list[dict[str, Any]]) -> bool:
    text = evidence_text(evidence)
    return any(anchor in text for anchor in question_anchors(question, hypothesis)[:8])


def generic_voice_evidence_only(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return True
    matched = 0
    for item in evidence:
        doc = item.get("document") or {}
        text = " ".join(
            str(doc.get(key) or "")
            for key in ("activity_name", "story_name", "story_id", "avg_tag", "clean_text")
        )
        if any(tag in text for tag in GENERIC_VOICE_TAGS):
            matched += 1
    return matched == len(evidence)


def missing_slot_is_generic(slot: str) -> bool:
    normalized = re.sub(r"\s+", "", str(slot or ""))
    if len(normalized) < 4:
        return True
    if normalized in BANNED_MISSING_SLOT_VALUES:
        return True
    return any(pattern in normalized for pattern in GENERIC_MISSING_SLOT_PATTERNS)


def follow_up_has_new_terms(
    previous_hypothesis: HypothesisDocument,
    follow_up: HypothesisDocument,
    missing_slots: list[str],
) -> bool:
    previous_terms = set(previous_hypothesis.entities + previous_hypothesis.keywords)
    current_terms = set(follow_up.entities + follow_up.keywords)
    new_terms = [
        term
        for term in current_terms - previous_terms
        if is_strong_follow_up_new_term(term)
    ]
    if len(new_terms) >= 2:
        return True
    slot_text = " ".join(missing_slots)
    return bool(new_terms) and any(term in slot_text for term in new_terms)


def is_strong_follow_up_new_term(term: str) -> bool:
    normalized = re.sub(r"\s+", "", str(term or ""))
    if is_bad_retrieval_term(normalized, field="keywords"):
        return False
    if normalized in WEAK_FOLLOW_UP_NEW_TERMS:
        return False
    if len(normalized) < 2:
        return False
    if any(marker in normalized for marker in ("原因", "背景", "信息", "线索", "剧情", "情况", "过程")):
        return False
    return True


def follow_up_mentions_slot_terms(follow_up: HypothesisDocument, missing_slots: list[str]) -> bool:
    if not missing_slots:
        return False
    haystack = "\n".join(
        [
            " ".join(follow_up.entities),
            " ".join(follow_up.keywords),
            follow_up.expected_answer_type,
        ]
    )
    slot_terms: list[str] = []
    for slot in missing_slots:
        for term in TOKEN_RE.findall(slot):
            if term not in NOISY_ANCHORS and len(term) >= 2:
                slot_terms.append(term)
                for part in re.split(r"[的与和及、，,；;：:（）()]+", term):
                    if len(part) >= 2 and part not in NOISY_ANCHORS:
                        slot_terms.append(part)
        slot_terms.extend(keyword for keyword in SLOT_ALIGNMENT_KEYWORDS if keyword in slot)
    if not slot_terms:
        return False
    return any(term in haystack for term in slot_terms)


def render_evidence_brief(evidence: list[dict[str, Any]], *, max_items: int, max_chars: int) -> list[str]:
    brief: list[str] = []
    for item in evidence[:max_items]:
        doc = item.get("document") or {}
        label = doc.get("id") or doc.get("stage_code") or doc.get("story_name") or "evidence"
        text = str(item.get("evidence_chain_text") or doc.get("clean_text") or "")
        brief.append(f"{label}: {compact_text(text, max_chars)}")
    return brief


def build_student_hypothesis_prompt(question: str, dialogue_context: str) -> str:
    lines = [
        f"task: {INITIAL_HYPOTHESIS_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append("output_schema: hypothesis_v2")
    return "\n".join(lines)


def build_student_conclusion_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    round_index: int,
    max_rounds: int,
    evidence: list[dict[str, Any]],
    max_evidence_items: int,
    max_evidence_chars: int,
) -> str:
    lines = [
        f"task: {CONCLUSION_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append(f"hypothesis: {compact_json(as_initial_payload(hypothesis))}")
    lines.append(f"round: {round_index}/{max_rounds}")
    brief = render_evidence_brief(evidence, max_items=max_evidence_items, max_chars=max_evidence_chars)
    if brief:
        lines.append("evidence_brief:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(brief, start=1))
    lines.append("output_schema: conclusion_v2")
    return "\n".join(lines)


def build_student_follow_up_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    conclusion: ConclusionResult,
    round_index: int,
    max_rounds: int,
    evidence: list[dict[str, Any]],
    max_evidence_items: int,
    max_evidence_chars: int,
) -> str:
    lines = [
        f"task: {FOLLOW_UP_HYPOTHESIS_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append(f"hypothesis: {compact_json(as_initial_payload(hypothesis))}")
    lines.append(f"round: {round_index}/{max_rounds}")
    brief = render_evidence_brief(evidence, max_items=max_evidence_items, max_chars=max_evidence_chars)
    if brief:
        lines.append("evidence_brief:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(brief, start=1))
    if conclusion.missing_slots:
        lines.append("missing_slots: " + "; ".join(conclusion.missing_slots[:5]))
    lines.append("output_schema: follow_up_hypothesis_v2")
    return "\n".join(lines)


def build_teacher_system() -> str:
    return (
        "你是《明日方舟》剧情 RAG 系统的教师标注模型。"
        "只输出一个合法 JSON 对象，不输出 markdown、解释或思维过程。"
        "如果证据不足，必须选择继续检索或拒答，不要硬答。"
    )


def build_teacher_hypothesis_prompt(question: str, dialogue_context: str) -> str:
    return "\n".join(
        [
            "任务: 生成检索假设 JSON。",
            f"用户问题: {question}",
            f"多轮上下文: {dialogue_context or '无'}",
            "",
            "输出字段严格为:",
            "question, intent, query_type, entities, keywords, expected_answer_type, dialogue_context",
            "",
            "规则:",
            "1. 不回答问题，只生成服务召回的假设。",
            "2. entities 只放角色、组织、地点、事件、物件等明确实体。",
            "3. keywords 放短检索词，包含自然称谓、关键事件词、章节/地点/关系线索；禁止只写“原因/目的/动机/关系/信息/影响/情况/剧情/分析”等泛词。",
            "4. 指代必须结合上下文消解为完整实体名。",
            "5. query_type 只能取 fact/relation/causality/reasoning/reveal/mystery/answerability。",
            "6. intent 只能取 plot_fact/plot_reasoning/timeline/character_relation/event_summary/compare/persona_chat/out_of_scope。",
            "7. 如果涉及凯尔希，entities 可以写“凯尔希”；keywords 只写“凯尔希/凯尔希医生/凯尔希所长”等自然文本称谓，禁止输出“凯尔希·思衡托”。",
            "8. 不要输出内部别名、内部代号、长 alias 串或逗号分隔实体串；只输出自然语言中常见的实体名和检索词。",
            "9. entities/keywords 禁止包含问句碎片或描述短语，例如“为什么会”“这件事”“如何影响”“具体原因”。",
            "10. keywords 建议 4-8 个，entities 建议 1-5 个；宁可少而准，不要塞满。",
        ]
    )


def render_seed_docs(docs: list[dict[str, Any]], *, max_chars_per_doc: int = 900) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        text = str(doc.get("clean_text") or doc.get("search_text") or "")
        blocks.append(
            "\n".join(
                [
                    f"[片段 {index}]",
                    f"id: {doc.get('id') or ''}",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or doc.get('story_id') or ''}",
                    f"stage_code: {doc.get('stage_code') or doc.get('stage_id') or ''}",
                    "text:",
                    compact_text(text, max_chars_per_doc),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_teacher_complex_questions_prompt(
    docs: list[dict[str, Any]],
    *,
    count: int,
    query_mix: str,
) -> str:
    return "\n".join(
        [
            f"任务: 基于下面同一批剧情片段生成 {count} 个复杂但可检索的用户问题。",
            "",
            "输出 JSON 字段严格为 questions。",
            "questions 是数组，每个元素字段严格为 question, dialogue_context, query_type, entities, difficulty。",
            "",
            "问题要求:",
            "1. 问题必须像真实用户会问的剧情问题，绝对不要提到“片段/上述/证据/检索/chunk/id/文本中”。",
            "2. 优先生成需要多跳证据的问题：原因动机、人物立场、关系变化、真相揭示、时间线、对比。",
            "3. 不要生成“X是什么身份/是谁”这种简单模板，除非证据里确实有隐藏身份或身份揭示。",
            "4. 不要生成干员语音/档案字段问题，例如行动失败、选中干员、信赖触摸、临床诊断分析。",
            "5. 每个问题必须包含明确人物/组织/地点/事件锚点，不能只问“这件事/这种情况/他们为什么”。",
            "6. 每个问题最多一个主问题；允许一个必要子问，但禁止三个以上问点，整句不超过 64 个汉字。",
            "7. 不要生成需要证据外百科常识才能回答的问题；答案必须能由给定剧情片段或相邻召回证据支撑。",
            "8. dialogue_context 只在追问型问题需要上下文时填写，否则为空字符串。",
            "9. query_type 只能是 fact/relation/causality/reasoning/reveal/mystery/answerability；不要输出 timeline/compare。",
            "10. entities 只写明确实体，禁止写“这件事/这种矛盾/学生们/他们”等泛称。",
            "11. 不要生成凯尔希相关问题；如果片段主要围绕凯尔希、凯尔希医生、凯尔希所长，则跳过该片段，换其他实体出题。",
            f"12. 目标类型倾向: {query_mix}",
            "",
            "剧情片段:",
            render_seed_docs(docs),
        ]
    )


def build_teacher_conclusion_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    round_index: int,
    max_rounds: int,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    prompt_evidence_top_k: int,
) -> str:
    history = []
    for step in retrieval_trace[-2:]:
        line = f"round={step.get('round')} action={step.get('planner_action')}"
        missing = step.get("missing_slots") or []
        if missing:
            line += " missing=" + ",".join(str(item) for item in missing[:4])
        history.append(line)
    prompt_evidence = select_prompt_evidence(
        question,
        hypothesis,
        evidence,
        prompt_evidence_top_k=prompt_evidence_top_k,
    )
    return "\n".join(
        [
            "任务: 根据当前真实检索证据生成阶段结论 JSON。你正在控制线上 RAG，多轮未结束时可以选择 retrieve_more 触发下一轮真实召回，不要急着硬答。",
            f"用户问题: {question}",
            f"多轮上下文: {dialogue_context or '无'}",
            f"当前轮次: {round_index}/{max_rounds}",
            "当前假设:",
            compact_json(as_initial_payload(hypothesis)),
            "历史摘要:",
            "\n".join(history) if history else "无",
            "当前证据:",
            render_evidence_blocks(prompt_evidence, max_chars_per_doc=520, max_total_chars=5000),
            "",
            "输出字段严格为:",
            "question, next_action, answer, missing_slots, clarification_question, follow_up_hypothesis",
            "",
            "决策规则:",
            "1. next_action 只能是 answer_directly/retrieve_more/clarify_user/abstain。",
            "2. 只有证据直接包含问题主实体且支持答案的关键判断，才允许 answer_directly。",
            "3. 当前轮次小于最大轮次时，只要答案需要证据外事实、只有局部片段、缺关键因果/身份/关系桥接，就选择 retrieve_more。",
            "4. 最后一轮仍缺关键证据时选择 abstain，并在 answer 中说明当前证据缺了什么；不要继续 retrieve_more。",
            "5. 身份/关系/原因/真相类问题遇到通用语音、信赖触摸、行动失败/出发、选中干员等碎片，禁止 answer_directly。",
            "6. 只能依据当前证据和当前假设做决策；不要引入证据外的图谱关系或别名串。",
            "7. answer 只能写当前证据明示或相邻证据直接推出的信息；证据只支持一句台词时，只回答这句台词支持的内容，不要扩展人设/动机/结局。",
            "8. retrieve_more 时 answer 必须为空，missing_slots 必须是 2-5 个具体可检索缺口，每条 6-24 字，禁止“更多信息/完整剧情/具体背景/相关资料”等泛词。",
            "9. retrieve_more 时 follow_up_hypothesis 必须非空；它是下一轮召回假设文档，不是追问用户的问题。",
            "10. follow_up_hypothesis 只能包含 question, query_type, entities, keywords, expected_answer_type, dialogue_context。",
            "11. follow_up_hypothesis.question 必须原样保留用户问题；不要改写成新的 follow-up question。",
            "12. follow_up_hypothesis.entities 必须保留主实体；关系问题保留双方实体；新增桥接实体必须来自当前假设、证据或上下文。",
            "13. follow_up_hypothesis 必须面向下一轮召回，必须比当前假设新增至少 2 个有效桥接词；如果没有新方向就不要 retrieve_more，最后一轮选 abstain。",
            "14. follow_up_hypothesis.keywords 要包含 missing_slots 中的关键名词/关系词/因果词，用于下一轮召回；禁止只写“原因/目的/动机/关系/互动/联系/身份/角色/信息/影响”。",
            "15. follow_up_hypothesis.entities 只放明确实体，禁止问句碎片或描述短语，例如“为什么会”“如何影响”“这件事”“主意同意某人”。",
            "16. follow_up_hypothesis.keywords 只写 4-10 个自然检索词，禁止把当前证据里所有别名/图谱关系照抄进去。",
            "13.1 keywords 只写自然检索词，不要照抄 MiniRAG alias、内部代号或长别名；如果涉及凯尔希，只写“凯尔希/凯尔希医生/凯尔希所长”，禁止输出“凯尔希·思衡托”。",
            "17. answer_directly/abstain/clarify_user 时 follow_up_hypothesis 必须为 null。",
            "18. clarification_question 只在 clarify_user 时填写；其他动作必须为空字符串，不要写 null/None/无。",
            "19. answer_directly 的 answer 不要写“证据1/根据证据/证据显示”等标签，尽量复用证据原词，减少解释性改写。",
        ]
    )


def build_teacher_conclusion_retry_prompt(
    *,
    base_prompt: str,
    rejected_payload: dict[str, Any],
    validation_error: str,
    max_round_reached: bool,
) -> str:
    if max_round_reached:
        action_hint = "当前已经是最后一轮。若证据不能直接支撑答案，必须改为 abstain，并写清证据不足点。"
    else:
        action_hint = "当前还可以继续召回。若上次 answer_directly 因证据不足被拒绝，优先改为 retrieve_more，并给出具体 missing_slots 与新的召回假设文档；不要生成追问问题，follow_up_hypothesis.question 保留原用户问题。"
    return "\n".join(
        [
            base_prompt,
            "",
            "上一次输出未通过本地质量校验。",
            f"校验错误: {validation_error}",
            f"上一次输出: {compact_json(rejected_payload)}",
            "",
            "请重新输出同一 schema 的单个 JSON 对象。",
            action_hint,
            "不要重复无效答案；不要引入当前证据和假设之外的新事实。",
        ]
    )


def parse_teacher_payload(text: str) -> dict[str, Any]:
    try:
        payload = parse_teacher_json(text)
    except Exception:
        stripped = text.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("teacher payload is not an object")
    if isinstance(payload.get("samples"), list) and payload["samples"]:
        first = payload["samples"][0]
        if isinstance(first, dict):
            return first
    for key in ("hypothesis", "conclusion", "output", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def doc_is_story_seed(doc: dict[str, Any]) -> bool:
    source_path = str(doc.get("source_path") or "")
    text = str(doc.get("clean_text") or "")
    if len(text) < 120:
        return False
    if any(pattern in text for pattern in GENERIC_VOICE_TAGS):
        return False
    if "[uc]info" in source_path:
        return False
    if "charword_table" in source_path or "handbook_info_table" in source_path:
        return False
    return True


def build_seed_doc_groups(
    documents: list[dict[str, Any]],
    *,
    group_size: int,
    max_groups: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        if not doc_is_story_seed(doc):
            continue
        key = str(doc.get("source_path") or doc.get("story_id") or doc.get("activity_id") or "")
        if not key:
            continue
        grouped[key].append(doc)
    groups: list[list[dict[str, Any]]] = []
    for docs in grouped.values():
        ordered = sorted(docs, key=lambda item: str(item.get("id") or ""))
        if len(ordered) < 2:
            continue
        for start in range(0, len(ordered), group_size):
            chunk = ordered[start : start + group_size]
            if len(chunk) >= 2:
                groups.append(chunk)
    rng = random.Random(seed)
    rng.shuffle(groups)
    return groups[:max_groups]


def normalize_complex_question_item(raw: Any, *, source_key: str, index: int) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    question = str(raw.get("question") or "").strip()
    query_type = str(raw.get("query_type") or "").strip()
    query_type = COMPLEX_QUERY_TYPE_ALIASES.get(query_type, query_type)
    if query_type not in ALLOWED_COMPLEX_QUERY_TYPES:
        return None
    entities = raw.get("entities") if isinstance(raw.get("entities"), list) else []
    if len(question) < 6 or is_bad_question(question) or is_kaltsit_related_question(question, entities):
        return None
    normalized_entities = [
        normalize_teacher_entity_or_keyword(str(item), field="entities")
        for item in entities
        if isinstance(item, (str, int, float))
    ]
    normalized_entities = [
        item
        for item in dedupe_keep_order(normalized_entities)
        if item and not is_bad_retrieval_term(item, field="entities")
    ]
    if not normalized_entities:
        return None
    dialogue_context = sanitize_dialogue_context(raw.get("dialogue_context"))
    difficulty = str(raw.get("difficulty") or "hard").strip() or "hard"
    key = stable_key("teacher_complex", question, dialogue_context)
    return {
        "question_key": key,
        "question": question,
        "dialogue_context": dialogue_context,
        "source_split": "teacher_complex",
        "source_task_type": "teacher_complex_question_generation",
        "source_seed_key": source_key,
        "source_seed_index": str(index),
        "query_type": query_type,
        "entities": ",".join(normalized_entities)[:200],
        "difficulty": difficulty,
    }


def generate_teacher_complex_questions(
    *,
    api_config: TeacherApiConfig,
    documents_path: Path,
    output_dir: Path,
    target_questions: int,
    questions_per_request: int,
    seed_doc_count: int,
    seed: int,
    api_retries: int,
    retry_sleep: float,
    raw_output: Path,
    query_mix: str,
    resume: bool,
    parallel: int,
) -> list[dict[str, str]]:
    cache_path = output_dir / "teacher_complex_questions.jsonl"
    cached = load_jsonl(cache_path) if resume else []
    by_key: dict[str, dict[str, str]] = {
        str(item.get("question_key")): {key: str(value) for key, value in item.items()}
        for item in cached
        if item.get("question_key")
    }
    if len(by_key) >= target_questions:
        log_progress(f"[complex-questions] cache hit {len(by_key)}/{target_questions}")
        return list(by_key.values())[:target_questions]

    log_progress(f"[complex-questions] load documents {documents_path}")
    documents = load_documents(documents_path)
    log_progress(f"[complex-questions] documents={len(documents)}")
    groups = build_seed_doc_groups(
        documents,
        group_size=seed_doc_count,
        max_groups=max(target_questions * 2, 20),
        seed=seed,
    )
    log_progress(
        f"[complex-questions] groups={len(groups)} target={target_questions} cached={len(by_key)} parallel={parallel}"
    )
    system_prompt = build_teacher_system()

    def run_group(group_index: int, docs: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
        source_key = stable_key(*(str(doc.get("id") or "") for doc in docs))
        log_progress(f"[complex-start] group={group_index:05d} docs={len(docs)} source={source_key}")
        started = time.time()
        try:
            payload = call_teacher_json(
                api_config,
                system_prompt=system_prompt,
                user_prompt=build_teacher_complex_questions_prompt(
                    docs,
                    count=questions_per_request,
                    query_mix=query_mix,
                ),
                retries=api_retries,
                retry_sleep=retry_sleep,
                raw_output=raw_output,
                request_meta={
                    "question_key": source_key,
                    "task_type": "teacher_complex_question_generation",
                    "round": 0,
                },
            )
            questions = payload.get("questions")
            if not isinstance(questions, list):
                return [], None
            items = []
            for item_index, raw_item in enumerate(questions):
                item = normalize_complex_question_item(raw_item, source_key=source_key, index=item_index)
                if item is not None:
                    items.append(item)
            log_progress(
                f"[complex-done] group={group_index:05d} questions={len(items)} latency={time.time() - started:.1f}s"
            )
            return items, None
        except Exception as exc:  # noqa: BLE001
            log_progress(
                f"[complex-failed] group={group_index:05d} error={str(exc)[:180]} latency={time.time() - started:.1f}s"
            )
            return [], {
                "source_key": source_key,
                "group_index": group_index,
                "doc_ids": [doc.get("id") for doc in docs],
                "error": str(exc),
                "created_at": int(time.time()),
            }

    parallel = max(1, parallel)
    if parallel == 1:
        iterator = enumerate(tqdm(groups, desc="teacher complex questions", unit="batch"))
        for group_index, docs in iterator:
            if len(by_key) >= target_questions:
                break
            items, error_record = run_group(group_index, docs)
            if error_record is not None:
                append_jsonl(output_dir / "failed_complex_question_batches.jsonl", error_record)
                continue
            for item in items:
                if item["question_key"] in by_key:
                    continue
                by_key[item["question_key"]] = item
                append_jsonl(cache_path, item)
                if len(by_key) >= target_questions:
                    break
        return list(by_key.values())[:target_questions]

    executor = ThreadPoolExecutor(max_workers=parallel)
    try:
        future_to_group = {
            executor.submit(run_group, group_index, docs): group_index
            for group_index, docs in enumerate(groups)
        }
        progress = tqdm(as_completed(future_to_group), total=len(future_to_group), desc="teacher complex questions", unit="batch")
        for future in progress:
            if len(by_key) >= target_questions:
                break
            items, error_record = future.result()
            if error_record is not None:
                append_jsonl(output_dir / "failed_complex_question_batches.jsonl", error_record)
                continue
            for item in items:
                if item["question_key"] in by_key:
                    continue
                by_key[item["question_key"]] = item
                append_jsonl(cache_path, item)
                if len(by_key) >= target_questions:
                    break
            progress.set_postfix({"questions": len(by_key)})
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return list(by_key.values())[:target_questions]


def call_teacher_json(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    retries: int,
    retry_sleep: float,
    raw_output: Path,
    request_meta: dict[str, Any],
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        raw_text = None
        raw_response = None
        task_type = str(request_meta.get("task_type") or "api")
        question_key = str(request_meta.get("question_key") or "")[:16]
        round_id = request_meta.get("round", "")
        validation_attempt = request_meta.get("validation_attempt")
        validation_suffix = "" if validation_attempt is None else f" validation={validation_attempt}"
        log_progress(
            f"[api-start] task={task_type} key={question_key} round={round_id}{validation_suffix} "
            f"attempt={attempt}/{retries + 1} model={api_config.model}"
        )
        started = time.time()
        try:
            raw_text, raw_response = call_teacher_api(
                api_config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            payload = parse_teacher_payload(raw_text)
            log_progress(
                f"[api-done] task={task_type} key={question_key} round={round_id}{validation_suffix} "
                f"attempt={attempt} latency={time.time() - started:.1f}s"
            )
            append_jsonl(
                raw_output,
                {
                    **request_meta,
                    "attempt": attempt,
                    "ok": True,
                    "api_type": api_config.api_type,
                    "model": api_config.model,
                    "base_url": api_config.base_url,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "created_at": int(time.time()),
                },
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            log_progress(
                f"[api-error] task={task_type} key={question_key} round={round_id}{validation_suffix} "
                f"attempt={attempt} latency={time.time() - started:.1f}s error={last_error[:180]}"
            )
            append_jsonl(
                raw_output,
                {
                    **request_meta,
                    "attempt": attempt,
                    "ok": False,
                    "api_type": api_config.api_type,
                    "model": api_config.model,
                    "base_url": api_config.base_url,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "error": last_error,
                    "created_at": int(time.time()),
                },
            )
            if attempt <= retries:
                time.sleep(retry_sleep)
    raise RuntimeError(last_error or "teacher api failed")


def validate_initial_hypothesis(payload: dict[str, Any], *, question: str, dialogue_context: str) -> HypothesisDocument:
    payload = sanitize_teacher_alias_fields(payload)
    return sanitize_hypothesis_document(
        normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=None,
        )
    )


def validate_conclusion(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_round_reached: bool,
    grounding_mode: str = "soft",
    grounding_warnings: list[dict[str, Any]] | None = None,
) -> ConclusionResult:
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(f"invalid grounding_mode: {grounding_mode}")
    payload = sanitize_teacher_alias_fields(payload)
    payload = sanitize_teacher_conclusion_payload(payload, max_round_reached=max_round_reached)
    conclusion = normalize_conclusion_payload(
        payload,
        question=question,
        dialogue_context=dialogue_context,
        current_intent=hypothesis.intent,
        max_round_reached=max_round_reached,
    )
    if conclusion.next_action == "answer_directly":
        if not evidence_mentions_anchor(question, hypothesis, evidence):
            raise ValueError("answer_directly rejected: evidence does not mention question anchor")
        if generic_voice_evidence_only(evidence) and any(word in question for word in IDENTITY_OR_REASON_WORDS):
            raise ValueError("answer_directly rejected: generic voice evidence only")
        if grounding_mode != "off":
            grounded = validate_conclusion_grounding(
                question=question,
                hypothesis=hypothesis,
                evidence=evidence,
                conclusion=conclusion,
                max_round_reached=max_round_reached,
            )
            grounding_changed = (
                grounded.next_action != conclusion.next_action
                or grounded.answer != conclusion.answer
            )
            if grounding_changed:
                warning = {
                    "warning_type": "grounding_validator_downgraded_output",
                    "warning_tags": ["grounding_soft_downgrade"],
                    "grounding_mode": grounding_mode,
                    "original_action": conclusion.next_action,
                    "original_answer": conclusion.answer,
                    "downgraded_action": grounded.next_action,
                    "downgraded_answer": grounded.answer,
                    "max_round_reached": max_round_reached,
                    "evidence_doc_ids": [
                        (item.get("document") or {}).get("id")
                        for item in evidence[:12]
                    ],
                }
                if grounding_warnings is not None:
                    grounding_warnings.append(warning)
                if grounding_mode == "strict":
                    raise ValueError("answer_directly rejected: grounding validator downgraded output")
                # Soft mode keeps the sidecar warning, but the training record
                # must not teach "answer_directly" with a synthetic fallback.
                # For SFT, this is a safe terminal abstain target.
                conclusion = ConclusionResult(
                    next_action="abstain",
                    answer=grounded.answer,
                    missing_slots=grounded.missing_slots,
                    clarification_question="",
                    follow_up_hypothesis=None,
                )
            else:
                conclusion = grounded
    if conclusion.next_action == "retrieve_more":
        if max_round_reached:
            raise ValueError("retrieve_more rejected at max round")
        if conclusion.follow_up_hypothesis is None:
            raise ValueError("retrieve_more missing follow_up_hypothesis")
        conclusion.follow_up_hypothesis = sanitize_hypothesis_document(
            normalize_hypothesis_payload(
                sanitize_teacher_alias_fields(as_follow_up_payload(conclusion.follow_up_hypothesis)),
                question=question,
                dialogue_context=dialogue_context,
                current_intent=hypothesis.intent,
            )
        )
        if not any(not missing_slot_is_generic(slot) for slot in conclusion.missing_slots):
            raise ValueError("retrieve_more rejected: missing_slots are too generic")
        if not follow_up_mentions_slot_terms(conclusion.follow_up_hypothesis, conclusion.missing_slots):
            raise ValueError("retrieve_more rejected: follow_up_hypothesis does not align with missing_slots")
        if not follow_up_has_new_terms(hypothesis, conclusion.follow_up_hypothesis, conclusion.missing_slots):
            raise ValueError("retrieve_more rejected: follow_up_hypothesis adds no useful new retrieval terms")
    return conclusion


def call_and_validate_teacher_conclusion(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    base_user_prompt: str,
    retries: int,
    retry_sleep: float,
    raw_output: Path,
    warnings_output: Path,
    request_meta: dict[str, Any],
    record_id: str,
    validation_retries: int,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_round_reached: bool,
    grounding_mode: str,
) -> ConclusionResult:
    user_prompt = base_user_prompt
    last_error: Exception | None = None
    rejected_payload: dict[str, Any] = {}
    for validation_attempt in range(validation_retries + 1):
        payload = call_teacher_json(
            api_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retries=retries,
            retry_sleep=retry_sleep,
            raw_output=raw_output,
            request_meta={
                **request_meta,
                "validation_attempt": validation_attempt,
            },
        )
        rejected_payload = sanitize_teacher_conclusion_payload(
            payload,
            max_round_reached=max_round_reached,
        )
        try:
            grounding_warnings: list[dict[str, Any]] = []
            conclusion = validate_conclusion(
                rejected_payload,
                question=question,
                dialogue_context=dialogue_context,
                hypothesis=hypothesis,
                evidence=evidence,
                max_round_reached=max_round_reached,
                grounding_mode=grounding_mode,
                grounding_warnings=grounding_warnings,
            )
            for warning in grounding_warnings:
                append_jsonl(
                    warnings_output,
                    {
                        **request_meta,
                        "record_id": record_id,
                        "validation_attempt": validation_attempt,
                        "question": question,
                        **warning,
                        "created_at": int(time.time()),
                    },
                )
            return conclusion
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            append_jsonl(
                raw_output,
                {
                    **request_meta,
                    "validation_attempt": validation_attempt,
                    "ok": False,
                    "api_type": api_config.api_type,
                    "model": api_config.model,
                    "base_url": api_config.base_url,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_text": None,
                    "raw_response": None,
                    "error": f"validation rejected: {exc}",
                    "rejected_payload": rejected_payload,
                    "created_at": int(time.time()),
                },
            )
            if validation_attempt >= validation_retries:
                break
            user_prompt = build_teacher_conclusion_retry_prompt(
                base_prompt=base_user_prompt,
                rejected_payload=rejected_payload,
                validation_error=str(exc),
                max_round_reached=max_round_reached,
            )
    raise last_error or RuntimeError("teacher conclusion validation failed")


def make_sft_record(
    *,
    record_id: str,
    task_type: str,
    student_prompt: str,
    assistant_payload: dict[str, Any],
    question_item: dict[str, str],
    round_index: int | None,
    retrieval_query: str | None,
    evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "category": "tool",
        "task_family": task_type,
        "generation_mode": "online_teacher_chain_short_prompt_v1",
        "short_prompt_schema": "current_pipeline_v1",
        "source_question_key": question_item["question_key"],
        "source_question": question_item["question"],
        "source_dialogue_context": question_item.get("dialogue_context") or "",
        "source_split": question_item.get("source_split") or "",
        "source_task_type": question_item.get("source_task_type") or "",
    }
    if round_index is not None:
        meta["round"] = round_index
    if retrieval_query:
        meta["retrieval_query"] = retrieval_query
    if evidence is not None:
        meta["evidence_doc_ids"] = [
            (item.get("document") or {}).get("id")
            for item in evidence[:12]
        ]
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "tool",
        "system": STUDENT_SYSTEM,
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": student_prompt},
            {"from": "gpt", "value": compact_json(assistant_payload)},
        ],
        "meta": meta,
    }


def retrieve_round(
    retriever: ArknightsHybridRetriever,
    *,
    question: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
    query_config: QueryConfig,
) -> list[dict[str, Any]]:
    dense_ranked_lists: list[list[dict[str, Any]]] = []
    sparse_ranked_lists: list[list[dict[str, Any]]] = []
    minirag_ranked_lists: list[list[dict[str, Any]]] = []
    for query in queries:
        dense_ranked_lists.append(retriever.dense_search(query, top_k=query_config.dense_top_k))
        sparse_ranked_lists.append(retriever.sparse_search(query, top_k=query_config.sparse_top_k))
        minirag_hits = retriever.minirag_search(query, top_k=query_config.minirag_top_k)
        if minirag_hits:
            minirag_ranked_lists.append(minirag_hits)
    dense_hits = merge_ranked_hits(*dense_ranked_lists)
    sparse_hits = merge_ranked_hits(*sparse_ranked_lists)
    minirag_hits = merge_ranked_hits(*minirag_ranked_lists)
    minirag_weight = retriever.effective_minirag_weight(question, config=query_config)
    if query_config.minirag_fusion_mode == "append":
        primary_hits = retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=[],
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
            minirag_weight=0.0,
        )
        fused_hits = retriever.append_supplemental_hits(
            primary_hits,
            minirag_hits if minirag_weight > 0 else [],
            top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
            source_name="minirag",
        )
    else:
        fused_hits = retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=minirag_hits if minirag_weight > 0 else [],
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
            minirag_weight=minirag_weight,
        )
    if query_config.enable_neighbor_expansion:
        fused_hits = retriever.expand_hits_with_neighbors(
            fused_hits,
            max_seed_docs=query_config.neighbor_max_seed_docs,
            story_window=query_config.neighbor_story_window,
            activity_story_sort_window=query_config.neighbor_activity_story_sort_window,
            top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
        )
    return rerank_hits(
        retriever,
        question,
        fused_hits,
        top_k=query_config.rerank_top_k,
        batch_size=query_config.rerank_batch_size,
        query_mode=classify_retrieval_query_mode(hypothesis),
    )


def generate_question_chain(
    *,
    question_item: dict[str, str],
    api_config: TeacherApiConfig,
    retriever: ArknightsHybridRetriever,
    query_config: QueryConfig,
    max_rounds: int,
    prompt_evidence_top_k: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    api_retries: int,
    retry_sleep: float,
    raw_output: Path,
    warnings_output: Path,
    validation_retries: int,
    grounding_mode: str,
    retrieval_lock: Any | None = None,
) -> list[dict[str, Any]]:
    question = question_item["question"]
    dialogue_context = question_item.get("dialogue_context") or ""
    question_key = question_item["question_key"]
    records: list[dict[str, Any]] = []
    system_prompt = build_teacher_system()
    log_progress(f"[chain-start] key={question_key} question={compact_text(question, 80)}")

    initial_payload = call_teacher_json(
        api_config,
        system_prompt=system_prompt,
        user_prompt=build_teacher_hypothesis_prompt(question, dialogue_context),
        retries=api_retries,
        retry_sleep=retry_sleep,
        raw_output=raw_output,
        request_meta={
            "question_key": question_key,
            "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
            "round": 0,
        },
    )
    hypothesis = validate_initial_hypothesis(
        initial_payload,
        question=question,
        dialogue_context=dialogue_context,
    )
    records.append(
        make_sft_record(
            record_id=f"{question_key}-{INITIAL_HYPOTHESIS_TASK_TYPE}",
            task_type=INITIAL_HYPOTHESIS_TASK_TYPE,
            student_prompt=build_student_hypothesis_prompt(question, dialogue_context),
            assistant_payload=as_initial_payload(hypothesis),
            question_item=question_item,
            round_index=None,
            retrieval_query=None,
            evidence=None,
        )
    )
    log_progress(f"[hypothesis-done] key={question_key} entities={len(hypothesis.entities)} keywords={len(hypothesis.keywords)}")

    retrieval_trace: list[dict[str, Any]] = []
    pending_queries = [question, build_retrieval_query(hypothesis)]
    for round_index in range(1, max_rounds + 1):
        log_progress(
            f"[retrieval-start] key={question_key} round={round_index}/{max_rounds} queries={len(pending_queries)}"
        )
        retrieval_started = time.time()
        if retrieval_lock is None:
            evidence = retrieve_round(
                retriever,
                question=question,
                hypothesis=hypothesis,
                queries=pending_queries,
                query_config=query_config,
            )
        else:
            with retrieval_lock:
                evidence = retrieve_round(
                    retriever,
                    question=question,
                    hypothesis=hypothesis,
                    queries=pending_queries,
                    query_config=query_config,
                )
        log_progress(
            f"[retrieval-done] key={question_key} round={round_index}/{max_rounds} "
            f"evidence={len(evidence)} latency={time.time() - retrieval_started:.1f}s"
        )
        evidence_summary = summarize_evidence_for_trace(evidence)
        step_record: dict[str, Any] = {
            "round": round_index,
            "queries": list(pending_queries),
            "planner_action": "retrieval_completed",
            "hypothesis": as_initial_payload(hypothesis),
            "evidence_summary": evidence_summary,
        }
        retrieval_trace.append(step_record)
        conclusion_prompt = build_teacher_conclusion_prompt(
            question=question,
            dialogue_context=dialogue_context,
            hypothesis=hypothesis,
            round_index=round_index,
            max_rounds=max_rounds,
            evidence=evidence,
            retrieval_trace=retrieval_trace,
            prompt_evidence_top_k=prompt_evidence_top_k,
        )
        conclusion_record_id = f"{question_key}-{CONCLUSION_TASK_TYPE}-{round_index:02d}"
        conclusion = call_and_validate_teacher_conclusion(
            api_config,
            system_prompt=system_prompt,
            base_user_prompt=conclusion_prompt,
            retries=api_retries,
            retry_sleep=retry_sleep,
            raw_output=raw_output,
            warnings_output=warnings_output,
            request_meta={
                "question_key": question_key,
                "task_type": CONCLUSION_TASK_TYPE,
                "round": round_index,
            },
            record_id=conclusion_record_id,
            validation_retries=validation_retries,
            question=question,
            dialogue_context=dialogue_context,
            hypothesis=hypothesis,
            evidence=evidence,
            max_round_reached=round_index >= max_rounds,
            grounding_mode=grounding_mode,
        )
        log_progress(
            f"[conclusion-done] key={question_key} round={round_index}/{max_rounds} action={conclusion.next_action}"
        )
        step_record["planner_action"] = conclusion.next_action
        step_record["conclusion"] = as_conclusion_payload(conclusion, question=question)
        step_record["missing_slots"] = conclusion.missing_slots
        records.append(
            make_sft_record(
                record_id=conclusion_record_id,
                task_type=CONCLUSION_TASK_TYPE,
                student_prompt=build_student_conclusion_prompt(
                    question=question,
                    dialogue_context=dialogue_context,
                    hypothesis=hypothesis,
                    round_index=round_index,
                    max_rounds=max_rounds,
                    evidence=evidence,
                    max_evidence_items=max_evidence_items,
                    max_evidence_chars=max_evidence_chars,
                ),
                assistant_payload=as_conclusion_payload(conclusion, question=question),
                question_item=question_item,
                round_index=round_index,
                retrieval_query="\n\n".join(pending_queries),
                evidence=evidence,
            )
        )

        if conclusion.next_action != "retrieve_more":
            break
        if conclusion.follow_up_hypothesis is None:
            break
        records.append(
            make_sft_record(
                record_id=f"{question_key}-{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}-{round_index:02d}",
                task_type=FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                student_prompt=build_student_follow_up_prompt(
                    question=question,
                    dialogue_context=dialogue_context,
                    hypothesis=hypothesis,
                    conclusion=conclusion,
                    round_index=round_index,
                    max_rounds=max_rounds,
                    evidence=evidence,
                    max_evidence_items=max_evidence_items,
                    max_evidence_chars=max_evidence_chars,
                ),
                assistant_payload=as_follow_up_payload(conclusion.follow_up_hypothesis),
                question_item=question_item,
                round_index=round_index,
                retrieval_query="\n\n".join(pending_queries),
                evidence=evidence,
            )
        )
        hypothesis = merge_hypotheses(hypothesis, conclusion.follow_up_hypothesis)
        pending_queries = [build_retrieval_query(hypothesis)]
        step_record["follow_up_hypothesis"] = as_initial_payload(hypothesis)
        step_record["next_round_queries"] = pending_queries

    log_progress(f"[chain-done] key={question_key} records={len(records)}")
    return records


def export_splits(
    records_jsonl: Path,
    output_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    records = load_jsonl(records_jsonl)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str((record.get("meta") or {}).get("source_question_key") or record.get("id") or "")
        by_question[key].append(record)
    keys = list(by_question)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_cut = int(len(keys) * train_ratio)
    val_cut = train_cut + int(len(keys) * val_ratio)
    split_keys = {
        "train": set(keys[:train_cut]),
        "val": set(keys[train_cut:val_cut]),
        "test": set(keys[val_cut:]),
    }
    summary: dict[str, Any] = {"questions": len(keys), "records": len(records), "splits": {}}
    for split, selected_keys in split_keys.items():
        split_records = [record for key in keys if key in selected_keys for record in by_question[key]]
        write_json(output_dir / f"{split}.json", split_records)
        counter = Counter(str(record.get("task_type") or "") for record in split_records)
        summary["splits"][split] = {
            "questions": len(selected_keys),
            "records": len(split_records),
            "task_counts": dict(counter),
        }
    dataset_name = output_dir.name
    role_tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "observation_tag": "observation",
        "function_tag": "function_call",
    }
    def dataset_entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": role_tags,
        }

    write_json(
        output_dir / "dataset_info.json",
        {
            f"{dataset_name}_train": dataset_entry("train.json"),
            f"{dataset_name}_val": dataset_entry("val.json"),
            f"{dataset_name}_test": dataset_entry("test.json"),
        },
    )
    write_json(output_dir / "build_summary.json", summary)
    return summary


def parse_mode_weights(value: str) -> dict[str, float]:
    if not value.strip():
        return {}
    output: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw = item.split("=", 1)
        output[key.strip()] = float(raw)
    return output


def compute_teacher_complex_target(
    *,
    question_source: str,
    max_questions: int,
    source_questions: int,
    explicit_complex_questions: int,
) -> int:
    if question_source == "source":
        return 0
    if explicit_complex_questions > 0:
        return explicit_complex_questions
    if question_source == "teacher_complex":
        return max_questions
    return max(0, max_questions - source_questions)


def cli_option_provided(*names: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:] for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short-prompt online RAG chain SFT data with a teacher API."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-splits", default="train,val,test")
    parser.add_argument(
        "--source-task-types",
        default=f"{INITIAL_HYPOTHESIS_TASK_TYPE},{CONCLUSION_TASK_TYPE}",
        help="Comma-separated source task types used only for collecting original questions.",
    )
    parser.add_argument("--question-source", choices=("source", "teacher_complex", "mixed"), default="source")
    parser.add_argument("--source-question-ratio", type=float, default=0.6)
    parser.add_argument("--filter-bad-source-questions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--documents", type=Path, default=INDEX_ROOT / "documents.jsonl")
    parser.add_argument(
        "--teacher-complex-questions",
        type=int,
        default=0,
        help="Explicit number of API-generated complex questions. 0 means derive from --max-questions and --source-question-ratio.",
    )
    parser.add_argument("--teacher-complex-questions-per-request", type=int, default=4)
    parser.add_argument("--teacher-complex-seed-doc-count", type=int, default=5)
    parser.add_argument(
        "--teacher-complex-query-mix",
        default="causality/reasoning/reveal/relation/timeline/compare，减少简单身份题",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only collect and print source questions; do not call API or load retriever.")

    parser.add_argument("--teacher-config", type=Path, default=None, help="Optional JSON config containing a teacher_api section.")
    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "responses"), default=None)
    parser.add_argument("--api-base", required=False, default="")
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--auth-header", choices=("bearer", "x-api-key", "both"), default="bearer")
    parser.add_argument("--model", required=False, default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=2,
        help="Retry teacher conclusion calls with validation feedback before rejecting a question.",
    )
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--anthropic-disable-thinking", action="store_true")

    parser.add_argument("--parallel", type=int, default=4, help="Number of independent question chains to run concurrently.")
    parser.add_argument(
        "--parallel-retrieval",
        action="store_true",
        help="Allow concurrent GPU retrieval/reranker calls. Default keeps retrieval serialized to reduce VRAM risk.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=RERANKER_MODEL_DIR)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--minirag-index", type=Path, default=DEFAULT_MINIRAG_INDEX)
    parser.add_argument("--dense-top-k", type=int, default=80)
    parser.add_argument("--sparse-top-k", type=int, default=80)
    parser.add_argument("--minirag-top-k", type=int, default=120)
    parser.add_argument("--fusion-top-k", type=int, default=80)
    parser.add_argument("--rerank-top-k", type=int, default=12)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=120)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--minirag-weight", type=float, default=0.35)
    parser.add_argument("--minirag-mode-weights", type=parse_mode_weights, default={})
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default="score")
    parser.add_argument("--enable-neighbor-expansion", action="store_true")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=24)
    parser.add_argument("--neighbor-story-window", type=int, default=2)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=1)

    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--prompt-evidence-top-k", type=int, default=8)
    parser.add_argument("--max-evidence-items", type=int, default=6)
    parser.add_argument("--max-evidence-chars", type=int, default=220)
    parser.add_argument("--max-minirag-entities", type=int, default=8)
    parser.add_argument("--max-minirag-relations", type=int, default=6)
    parser.add_argument(
        "--grounding-mode",
        choices=GROUNDING_MODES,
        default="soft",
        help=(
            "strict: reject token-level grounding downgrades; "
            "soft: accept teacher answer and write sidecar warnings; "
            "off: skip token-level grounding validator."
        ),
    )
    parser.add_argument("--allow-ungrounded-answer", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = output_dir / "records.jsonl"
    raw_output = output_dir / "raw_teacher_requests.jsonl"
    failed_output = output_dir / "failed_questions.jsonl"
    warnings_output = output_dir / "record_warnings.jsonl"

    if args.export_only:
        summary = export_splits(
            records_jsonl,
            output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    source_dir = resolve_path(args.source_dir)
    source_splits = [item.strip() for item in args.source_splits.split(",") if item.strip()]
    source_task_types = {item.strip() for item in args.source_task_types.split(",") if item.strip()}
    source_limit = args.max_questions
    if args.question_source == "mixed":
        if args.teacher_complex_questions > 0:
            source_limit = max(0, args.max_questions - args.teacher_complex_questions)
        else:
            source_limit = max(0, int(round(args.max_questions * max(0.0, min(1.0, args.source_question_ratio)))))
    elif args.question_source == "teacher_complex":
        source_limit = 0
    source_questions = collect_source_questions(
        source_dir,
        splits=source_splits,
        source_task_types=source_task_types,
        limit=source_limit,
        seed=args.seed,
        filter_bad_questions=args.filter_bad_source_questions,
    )
    log_progress(
        f"[source-questions] source={source_dir} collected={len(source_questions)} "
        f"question_source={args.question_source}"
    )
    if args.dry_run:
        teacher_complex_target = compute_teacher_complex_target(
            question_source=args.question_source,
            max_questions=args.max_questions,
            source_questions=len(source_questions),
            explicit_complex_questions=args.teacher_complex_questions,
        )
        preview = {
            "source_dir": str(source_dir),
            "question_source": args.question_source,
            "source_questions": len(source_questions),
            "teacher_complex_target": teacher_complex_target,
            "preview": source_questions[:10],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    teacher_cfg: dict[str, Any] = {}
    if args.teacher_config:
        loaded_cfg = load_json(resolve_path(args.teacher_config))
        teacher_cfg = loaded_cfg.get("teacher_api", loaded_cfg) if isinstance(loaded_cfg, dict) else {}
    api_type = str(
        args.api_type
        if cli_option_provided("--api-type")
        else teacher_cfg.get("api_type") or "chat_completions"
    )
    api_base = str(args.api_base or teacher_cfg.get("base_url") or "")
    api_key_env = str(args.api_key_env if cli_option_provided("--api-key-env") else teacher_cfg.get("api_key_env") or args.api_key_env)
    auth_header = str(args.auth_header if cli_option_provided("--auth-header") else teacher_cfg.get("auth_header") or args.auth_header)
    model = str(args.model or teacher_cfg.get("model") or "")
    timeout = int(args.timeout if cli_option_provided("--timeout") else teacher_cfg.get("timeout_seconds") or args.timeout)
    temperature = float(
        args.temperature
        if cli_option_provided("--temperature")
        else teacher_cfg.get("temperature") if teacher_cfg.get("temperature") is not None else args.temperature
    )
    max_output_tokens = int(
        args.max_output_tokens
        if cli_option_provided("--max-output-tokens")
        else teacher_cfg.get("max_output_tokens") or args.max_output_tokens
    )
    json_mode = bool(
        (not args.no_json_mode)
        if cli_option_provided("--no-json-mode")
        else teacher_cfg.get("json_mode", not args.no_json_mode)
    ) and not args.no_json_mode
    extra_headers = teacher_cfg.get("extra_headers") if isinstance(teacher_cfg.get("extra_headers"), dict) else None
    anthropic_disable_thinking = bool(
        teacher_cfg.get("anthropic_disable_thinking", args.anthropic_disable_thinking)
    )
    grounding_mode = args.grounding_mode
    if args.allow_ungrounded_answer and not cli_option_provided("--grounding-mode"):
        grounding_mode = "off"

    if not api_base or not model:
        raise SystemExit("--api-base and --model are required unless --export-only or --dry-run is used.")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")

    query_config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        minirag_top_k=args.minirag_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=args.rerank_top_k,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights=args.minirag_mode_weights,
        minirag_fusion_mode=args.minirag_fusion_mode,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        enable_neighbor_expansion=args.enable_neighbor_expansion,
        neighbor_max_seed_docs=args.neighbor_max_seed_docs,
        neighbor_story_window=args.neighbor_story_window,
        neighbor_activity_story_sort_window=args.neighbor_activity_story_sort_window,
        rerank_batch_size=args.rerank_batch_size,
    )
    index_dir = resolve_path(args.index_dir)
    reranker_model_path = None if args.no_reranker else resolve_path(args.reranker_model)
    if reranker_model_path is not None and not reranker_model_path.exists():
        raise SystemExit(f"Reranker model not found: {reranker_model_path}. Pass --no-reranker to disable reranker mode.")
    log_progress(
        f"[retriever-load] start index={index_dir} device={args.device} "
        f"reranker={'off' if reranker_model_path is None else reranker_model_path}"
    )
    retriever_started = time.time()
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=resolve_path(args.embedding_model),
        reranker_model_path=reranker_model_path,
        documents_path=index_dir / "documents.jsonl",
        faiss_index_path=index_dir / "faiss.index",
        bm25_tokens_path=index_dir / "bm25_tokens.pkl",
        minirag_index_path=resolve_path(args.minirag_index),
        device=args.device,
    )
    log_progress(f"[retriever-load] done latency={time.time() - retriever_started:.1f}s")
    api_config = TeacherApiConfig(
        api_type=api_type,
        base_url=api_base,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        extra_headers=extra_headers,
        auth_header=auth_header,
        anthropic_disable_thinking=anthropic_disable_thinking,
    )

    teacher_complex_target = compute_teacher_complex_target(
        question_source=args.question_source,
        max_questions=args.max_questions,
        source_questions=len(source_questions),
        explicit_complex_questions=args.teacher_complex_questions,
    )
    complex_questions: list[dict[str, str]] = []
    if teacher_complex_target > 0:
        log_progress(f"[complex-questions] target={teacher_complex_target}")
        complex_questions = generate_teacher_complex_questions(
            api_config=api_config,
            documents_path=resolve_path(args.documents),
            output_dir=output_dir,
            target_questions=teacher_complex_target,
            questions_per_request=args.teacher_complex_questions_per_request,
            seed_doc_count=args.teacher_complex_seed_doc_count,
            seed=args.seed,
            api_retries=args.api_retries,
            retry_sleep=args.retry_sleep,
            raw_output=raw_output,
            query_mix=args.teacher_complex_query_mix,
            resume=args.resume,
            parallel=args.parallel,
        )
        log_progress(f"[complex-questions] collected={len(complex_questions)}")

    question_by_key: dict[str, dict[str, str]] = {}
    for item in source_questions + complex_questions:
        question_by_key.setdefault(item["question_key"], item)
    questions = list(question_by_key.values())[: args.max_questions]
    if not questions:
        raise SystemExit("No questions available after filtering/generation.")

    done_keys = {
        str((record.get("meta") or {}).get("source_question_key") or "")
        for record in load_jsonl(records_jsonl)
    } if args.resume else set()
    if done_keys:
        questions = [item for item in questions if item["question_key"] not in done_keys]
    log_progress(
        f"[run] questions={len(questions)} resume_done={len(done_keys)} parallel={args.parallel} "
        f"validation_retries={args.validation_retries} grounding_mode={grounding_mode}"
    )

    stats: Counter[str] = Counter()
    retrieval_lock = None if args.parallel_retrieval or args.parallel <= 1 else Lock()

    def run_question(question_item: dict[str, str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
        return question_item, generate_question_chain(
            question_item=question_item,
            api_config=api_config,
            retriever=retriever,
            query_config=query_config,
            max_rounds=args.max_rounds,
            prompt_evidence_top_k=args.prompt_evidence_top_k,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            api_retries=args.api_retries,
            retry_sleep=args.retry_sleep,
            raw_output=raw_output,
            warnings_output=warnings_output,
            validation_retries=args.validation_retries,
            grounding_mode=grounding_mode,
            retrieval_lock=retrieval_lock,
        )

    def record_success(records: list[dict[str, Any]]) -> None:
        for record in records:
            append_jsonl(records_jsonl, record)
            stats[f"task:{record['task_type']}"] += 1
        stats["completed_questions"] += 1
        stats["records"] += len(records)

    def record_failure(question_item: dict[str, str], exc: Exception) -> None:
        stats["failed_questions"] += 1
        append_jsonl(
            failed_output,
            {
                **question_item,
                "error": str(exc),
                "created_at": int(time.time()),
            },
        )

    parallel = max(1, args.parallel)
    if parallel == 1:
        progress = tqdm(questions, desc="online teacher chains", unit="q")
        for question_item in progress:
            try:
                _, records = run_question(question_item)
                record_success(records)
            except Exception as exc:  # noqa: BLE001
                record_failure(question_item, exc)
            progress.set_postfix(
                {
                    "ok": stats["completed_questions"],
                    "failed": stats["failed_questions"],
                    "records": stats["records"],
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_question = {
                executor.submit(run_question, question_item): question_item
                for question_item in questions
            }
            progress = tqdm(as_completed(future_to_question), total=len(future_to_question), desc="online teacher chains", unit="q")
            for future in progress:
                question_item = future_to_question[future]
                try:
                    _, records = future.result()
                    record_success(records)
                except Exception as exc:  # noqa: BLE001
                    record_failure(question_item, exc)
                progress.set_postfix(
                    {
                        "ok": stats["completed_questions"],
                        "failed": stats["failed_questions"],
                        "records": stats["records"],
                    }
                )

    summary = export_splits(
        records_jsonl,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    summary["run_stats"] = dict(stats)
    summary["source_dir"] = str(source_dir)
    summary["output_dir"] = str(output_dir)
    summary["question_source"] = args.question_source
    summary["source_question_count"] = len(source_questions)
    summary["teacher_complex_question_count"] = len(complex_questions)
    summary["parallel"] = args.parallel
    summary["parallel_retrieval"] = args.parallel_retrieval
    summary["validation_retries"] = args.validation_retries
    summary["grounding_mode"] = grounding_mode
    summary["grounding_soft_training_payload"] = (
        "downgraded_fallback" if grounding_mode == "soft" else grounding_mode
    )
    summary["record_warnings"] = str(warnings_output)
    summary["record_warnings_count"] = len(load_jsonl(warnings_output))
    summary["device"] = args.device
    summary["reranker_model"] = None if args.no_reranker else str(reranker_model_path)
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
