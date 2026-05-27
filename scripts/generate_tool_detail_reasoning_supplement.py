#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "indexes" / "arknights_story" / "documents.jsonl"
DEFAULT_BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sft_data"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "sft_data" / "tool_detail_reasoning_supplement_v1"
)
DEFAULT_MERGED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sft_data"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_v1"
)

INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"

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
HYPOTHESIS_QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}

HYPOTHESIS_SYSTEM = "你是《明日方舟》剧情问答系统中的 hypothesis_builder。"
FOLLOW_UP_SYSTEM = "你是《明日方舟》剧情问答系统中的 follow_up_hypothesis_builder。"
CONCLUSION_SYSTEM = "你是《明日方舟》剧情问答系统中的 conclusion_generator。"

SCHEMA_EXACT_FIELDS = {
    INITIAL_HYPOTHESIS_TASK_TYPE: {
        "question",
        "intent",
        "entities",
        "keywords",
        "expected_answer_type",
        "dialogue_context",
    },
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE: {
        "question",
        "entities",
        "keywords",
        "expected_answer_type",
        "dialogue_context",
    },
    CONCLUSION_TASK_TYPE: {
        "question",
        "next_action",
        "answer",
        "missing_slots",
        "clarification_question",
        "follow_up_hypothesis",
    },
}

TRIGGER_TERMS = (
    "阴谋",
    "真相",
    "计划",
    "目的",
    "原因",
    "发现",
    "得知",
    "曝光",
    "袭击",
    "背后",
    "主使",
    "身份",
    "下场",
    "动机",
    "为什么",
)

STOP_TERMS = {
    "用户问题",
    "多轮上下文",
    "当前证据",
    "当前假设",
    "历史生成",
    "历史检索",
    "行动前",
    "行动后",
    "幕间",
    "剧情",
    "故事",
    "活动",
    "计划",
    "真相",
    "原因",
    "目的",
    "身份",
    "关系",
    "问题",
    "结果",
    "过程",
    "具体",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "不是",
    "应该",
    "有关",
    "发现",
    "得知",
    "曝光",
    "袭击",
    "行动",
    "线索",
    "证据",
    "答案",
    "角色",
}

ENTITY_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9·]{2,16}")
BAD_TERM_SUBSTRINGS = (
    "用户",
    "助手",
    "证据",
    "上下文",
    "当前",
    "这个",
    "那个",
    "这里",
    "那里",
    "这些",
    "那些",
    "你们",
    "我们",
    "他们",
    "她们",
    "它们",
    "你的",
    "我的",
    "他的",
    "她的",
    "它的",
    "我不",
    "你不",
    "不是",
    "不必",
    "不用",
    "怎么办",
    "为什么",
    "怎么",
    "如何",
    "什么",
    "chunk",
    "level_",
)
BAD_TERM_SUFFIXES = (
    "指出",
    "认为",
    "表示",
    "说道",
    "提到",
    "发现",
    "看到",
    "听到",
    "得知",
    "明白",
    "询问",
)
GENERIC_SPEAKERS = {
    "？？？",
    "???",
    "旁白",
    "众人",
    "所有人",
    "感染者",
    "佣兵",
    "士兵",
    "市民",
    "路人",
    "工作人员",
    "研究员",
    "警备人员",
    "暴徒",
    "镇民",
}

CATEGORY_TARGETS = {
    "overview_too_shallow": 80,
    "causal_reasoning": 60,
    "plot_inference": 50,
    "fact_detail": 55,
    "user_correction_context": 55,
}


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    question_topic: str
    entities: tuple[str, ...]
    keywords: tuple[str, ...]
    expected_answer_type: str
    source_doc_ids: tuple[str, ...]
    shallow_doc_ids: tuple[str, ...]
    rich_doc_ids: tuple[str, ...]
    shallow_answer: str
    detailed_answer: str
    missing_slots: tuple[str, ...]
    correction_focus: tuple[str, ...]
    activity_names: tuple[str, ...]
    stage_codes: tuple[str, ...]
    story_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an incremental tool SFT supplement for shallow-overview, causal, "
            "reasoning, factual-detail, and user-correction retrieval behavior."
        )
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-output-dir", type=Path, default=DEFAULT_MERGED_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_text(text: str, *, limit: int = 520) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def stable_fingerprint(record: dict[str, Any]) -> str:
    normalized_messages = [
        {
            "role": message.get("role"),
            "name": message.get("name"),
            "content": re.sub(r"\s+", " ", str(message.get("content") or "")).strip(),
            "tool_calls": message.get("tool_calls"),
        }
        for message in record.get("messages", [])
    ]
    payload = {
        "task_type": record.get("task_type"),
        "messages": normalized_messages,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def split_samples(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio)
    val_cut = train_cut + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_cut],
        "val": shuffled[train_cut:val_cut],
        "test": shuffled[val_cut:],
    }


def bucket_of(record: dict[str, Any]) -> str:
    return str(record.get("bucket") or record.get("meta", {}).get("category") or "tool")


def save_bucket_splits(output_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    for bucket in ("style", "knowledge", "tool"):
        bucket_dir = output_dir / bucket
        bucket_records = {
            split_name: [record for record in records if bucket_of(record) == bucket]
            for split_name, records in splits.items()
        }
        all_records = bucket_records["train"] + bucket_records["val"] + bucket_records["test"]
        save_jsonl(bucket_dir / "all.jsonl", all_records)
        save_jsonl(bucket_dir / "train.jsonl", bucket_records["train"])
        save_jsonl(bucket_dir / "val.jsonl", bucket_records["val"])
        save_jsonl(bucket_dir / "test.jsonl", bucket_records["test"])


def dedupe_keep_order(items: list[str], *, limit: int | None = None) -> list[str]:
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


def normalize_candidate_term(value: str) -> str:
    term = value.strip("，。！？、：；（）()[]【】“”\"' \t\r\n")
    for suffix in BAD_TERM_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix) + 1:
            term = term[: -len(suffix)]
            break
    return term.strip("，。！？、：；（）()[]【】“”\"' \t\r\n")


def is_good_term(value: str, *, allow_long: bool = False) -> bool:
    term = normalize_candidate_term(value)
    if len(term) < 2:
        return False
    if len(term) > (12 if allow_long else 8):
        return False
    if term in STOP_TERMS or term in GENERIC_SPEAKERS:
        return False
    if term.isdigit():
        return False
    if any(bad in term for bad in BAD_TERM_SUBSTRINGS):
        return False
    if re.search(r"[{}\\[\\]<>]", term):
        return False
    # Avoid sentence fragments from raw Chinese text. Proper names and titles are
    # usually short; long phrases with function particles are poor retrieval
    # entities and caused malformed hypothesis expansion.
    if not allow_long and any(particle in term for particle in ("已经", "正在", "为了", "因为", "所以", "但是", "然而")):
        return False
    return True


def extract_terms(text: str, *, limit: int = 8, allow_long: bool = False) -> list[str]:
    terms: list[str] = []
    for match in ENTITY_RE.finditer(text):
        value = normalize_candidate_term(match.group(0))
        if not is_good_term(value, allow_long=allow_long):
            continue
        terms.append(value)
    return dedupe_keep_order(terms, limit=limit)


def extract_doc_entities(summary_doc: dict[str, Any], detail_docs: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for doc in detail_docs:
        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = normalize_candidate_term(str(segment.get("speaker") or ""))
            if is_good_term(speaker):
                candidates.append(speaker)
    for doc in [summary_doc] + detail_docs[:2]:
        for key in ("activity_name", "story_name", "stage_code", "stage_name"):
            value = normalize_candidate_term(str(doc.get(key) or ""))
            if is_good_term(value, allow_long=key in {"activity_name", "story_name"}):
                candidates.append(value)
    candidates.extend(extract_terms(str(summary_doc.get("clean_text") or ""), limit=8))
    candidates.extend(extract_terms(str(summary_doc.get("search_text") or ""), limit=8))
    return dedupe_keep_order(candidates, limit=10)


def load_documents(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    docs = load_jsonl(path)
    by_id = {str(doc.get("id")): doc for doc in docs}
    return docs, by_id


def doc_text(by_id: dict[str, dict[str, Any]], doc_id: str, *, limit: int = 520) -> str:
    doc = by_id.get(doc_id)
    if doc is None:
        return ""
    return clean_text(str(doc.get("clean_text") or ""), limit=limit)


def find_doc_ids(
    docs: list[dict[str, Any]],
    *,
    required: tuple[str, ...],
    story_contains: str | None = None,
    exclude_uc: bool = False,
    limit: int = 5,
) -> list[str]:
    output: list[str] = []
    for doc in docs:
        text = str(doc.get("clean_text") or "")
        story_id = str(doc.get("story_id") or "")
        source_path = str(doc.get("source_path") or "")
        if story_contains and story_contains not in story_id and story_contains not in source_path:
            continue
        if exclude_uc and "[uc]info" in source_path:
            continue
        if all(term in text for term in required):
            output.append(str(doc.get("id")))
            if len(output) >= limit:
                break
    return output


def build_hypothesis_payload(
    *,
    question: str,
    intent: str,
    entities: list[str],
    keywords: list[str],
    expected_answer_type: str,
    dialogue_context: str = "",
) -> dict[str, Any]:
    if intent not in HYPOTHESIS_INTENTS:
        raise ValueError(f"unsupported intent: {intent}")
    return {
        "question": question,
        "intent": intent,
        "entities": dedupe_keep_order(entities, limit=12),
        "keywords": dedupe_keep_order(keywords, limit=20),
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def build_follow_up_payload(
    *,
    question: str,
    entities: list[str],
    keywords: list[str],
    expected_answer_type: str,
    dialogue_context: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "entities": dedupe_keep_order(entities, limit=12),
        "keywords": dedupe_keep_order(keywords, limit=20),
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def render_context(dialogue_context: str) -> str:
    return dialogue_context.strip() if dialogue_context.strip() else "无"


def render_evidence(doc_texts: list[str]) -> str:
    blocks = []
    for idx, text in enumerate(doc_texts, start=1):
        if text:
            blocks.append(f"[证据{idx}] {text}")
    return "\n".join(blocks) if blocks else "[无有效证据]"


def record_meta(
    *,
    category: str,
    request_id: str,
    task_family: str,
    decision_case: str | None,
    difficulty: str,
    notes: str,
    source_story_ids: list[str],
    source_stage_codes: list[str],
    source_activity_names: list[str],
) -> dict[str, Any]:
    return {
        "category": "tool",
        "grounded": True,
        "difficulty": difficulty,
        "notes": notes,
        "source_story_ids": dedupe_keep_order(source_story_ids, limit=12),
        "source_stage_codes": dedupe_keep_order(source_stage_codes, limit=12),
        "source_activity_names": dedupe_keep_order(source_activity_names, limit=12),
        "task_family": task_family,
        "decision_case": decision_case,
        "generation_mode": "tool_detail_reasoning_supplement_v1",
        "supplement_category": category,
        "request_id": request_id,
    }


def make_record(
    *,
    sample_id: str,
    task_type: str,
    system: str,
    user: str,
    assistant_payload: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "id": sample_id,
        "task_type": task_type,
        "bucket": "tool",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": compact_json(assistant_payload)},
        ],
        "tools": [],
        "meta": meta,
    }
    record["fingerprint"] = stable_fingerprint(record)
    return record


def make_initial_hypothesis_record(
    *,
    idx: int,
    category: str,
    question: str,
    intent: str,
    entities: list[str],
    keywords: list[str],
    expected_answer_type: str,
    dialogue_context: str,
    source_story_ids: list[str],
    source_stage_codes: list[str],
    source_activity_names: list[str],
    notes: str,
    difficulty: str = "medium",
) -> dict[str, Any]:
    request_id = f"detail-supp-{idx:04d}"
    payload = build_hypothesis_payload(
        question=question,
        intent=intent,
        entities=entities,
        keywords=keywords,
        expected_answer_type=expected_answer_type,
        dialogue_context=dialogue_context,
    )
    user = "\n".join(
        [
            f"用户问题: {question}",
            f"多轮上下文: {render_context(dialogue_context)}",
            "请生成初始假设文档 JSON。",
        ]
    )
    return make_record(
        sample_id=f"{request_id}-{INITIAL_HYPOTHESIS_TASK_TYPE}-0000",
        task_type=INITIAL_HYPOTHESIS_TASK_TYPE,
        system=HYPOTHESIS_SYSTEM,
        user=user,
        assistant_payload=payload,
        meta=record_meta(
            category=category,
            request_id=request_id,
            task_family="hypothesis_generation",
            decision_case=None,
            difficulty=difficulty,
            notes=notes,
            source_story_ids=source_story_ids,
            source_stage_codes=source_stage_codes,
            source_activity_names=source_activity_names,
        ),
    )


def make_follow_up_hypothesis_record(
    *,
    idx: int,
    category: str,
    question: str,
    current_hypothesis: dict[str, Any],
    previous_conclusion: dict[str, Any],
    evidence_texts: list[str],
    entities: list[str],
    keywords: list[str],
    expected_answer_type: str,
    dialogue_context: str,
    source_story_ids: list[str],
    source_stage_codes: list[str],
    source_activity_names: list[str],
    notes: str,
    difficulty: str = "hard",
) -> dict[str, Any]:
    request_id = f"detail-supp-{idx:04d}"
    payload = build_follow_up_payload(
        question=question,
        entities=entities,
        keywords=keywords,
        expected_answer_type=expected_answer_type,
        dialogue_context=dialogue_context,
    )
    user = "\n".join(
        [
            f"用户原问题: {question}",
            f"多轮问答上下文: {render_context(dialogue_context)}",
            "当前假设文档(JSON):",
            json.dumps(current_hypothesis, ensure_ascii=False, indent=2),
            "上一轮结论生成结果(JSON):",
            json.dumps(previous_conclusion, ensure_ascii=False, indent=2),
            "当前证据:",
            render_evidence(evidence_texts),
            "请生成补充检索 hypothesis JSON。",
        ]
    )
    return make_record(
        sample_id=f"{request_id}-{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}-0000",
        task_type=FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
        system=FOLLOW_UP_SYSTEM,
        user=user,
        assistant_payload=payload,
        meta=record_meta(
            category=category,
            request_id=request_id,
            task_family="hypothesis_generation",
            decision_case=None,
            difficulty=difficulty,
            notes=notes,
            source_story_ids=source_story_ids,
            source_stage_codes=source_stage_codes,
            source_activity_names=source_activity_names,
        ),
    )


def make_conclusion_record(
    *,
    idx: int,
    category: str,
    question: str,
    hypothesis: dict[str, Any],
    evidence_texts: list[str],
    current_round: int,
    max_rounds: int,
    next_action: str,
    answer: str,
    missing_slots: list[str],
    follow_up_hypothesis: dict[str, Any] | None,
    dialogue_context: str,
    source_story_ids: list[str],
    source_stage_codes: list[str],
    source_activity_names: list[str],
    notes: str,
    difficulty: str = "medium",
) -> dict[str, Any]:
    request_id = f"detail-supp-{idx:04d}"
    payload = {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": "",
        "follow_up_hypothesis": follow_up_hypothesis,
    }
    user = "\n".join(
        [
            f"用户问题: {question}",
            f"多轮上下文: {render_context(dialogue_context)}",
            f"当前假设文档(JSON): {compact_json(hypothesis)}",
            "历史生成结果: [第1轮 hypothesis 已生成]",
            "历史检索上下文: [已围绕当前假设进行检索，获得当前证据]",
            f"当前检索轮次: 第{current_round}轮 / 最多{max_rounds}轮",
            "当前证据:",
            render_evidence(evidence_texts),
            "请基于证据生成当前阶段结论 JSON。",
        ]
    )
    return make_record(
        sample_id=f"{request_id}-{CONCLUSION_TASK_TYPE}-0000",
        task_type=CONCLUSION_TASK_TYPE,
        system=CONCLUSION_SYSTEM,
        user=user,
        assistant_payload=payload,
        meta=record_meta(
            category=category,
            request_id=request_id,
            task_family="conclusion_generation",
            decision_case=next_action,
            difficulty=difficulty,
            notes=notes,
            source_story_ids=source_story_ids,
            source_stage_codes=source_stage_codes,
            source_activity_names=source_activity_names,
        ),
    )


def validate_payload(task_type: str, payload: dict[str, Any]) -> None:
    expected = SCHEMA_EXACT_FIELDS[task_type]
    actual = set(payload)
    optional = {"query_type"} if task_type in {INITIAL_HYPOTHESIS_TASK_TYPE, FOLLOW_UP_HYPOTHESIS_TASK_TYPE} else set()
    if actual != expected and actual != expected | optional:
        raise ValueError(f"{task_type} fields mismatch: {sorted(actual)} != {sorted(expected)}")
    if task_type in {INITIAL_HYPOTHESIS_TASK_TYPE, FOLLOW_UP_HYPOTHESIS_TASK_TYPE} and "query_type" in payload:
        if payload["query_type"] not in HYPOTHESIS_QUERY_TYPES:
            raise ValueError(f"bad query_type: {payload['query_type']}")
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        if payload["intent"] not in HYPOTHESIS_INTENTS:
            raise ValueError(f"bad intent: {payload['intent']}")
    if task_type == CONCLUSION_TASK_TYPE:
        action = payload["next_action"]
        if action == "retrieve_more":
            if payload["answer"] or not payload["missing_slots"]:
                raise ValueError("retrieve_more payload must have empty answer and non-empty missing_slots")
            follow = payload["follow_up_hypothesis"]
            if not isinstance(follow, dict):
                raise ValueError("retrieve_more requires follow_up_hypothesis")
            validate_payload(FOLLOW_UP_HYPOTHESIS_TASK_TYPE, follow)
        else:
            if not payload["answer"]:
                raise ValueError(f"{action} payload requires non-empty answer")
            if payload["follow_up_hypothesis"] is not None:
                raise ValueError(f"{action} payload requires null follow_up_hypothesis")


def validate_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{record.get('id')}: expected 3 messages")
    if messages[0].get("role") != "system" or messages[-1].get("role") != "assistant":
        raise ValueError(f"{record.get('id')}: bad message roles")
    payload = json.loads(messages[-1]["content"])
    validate_payload(str(record.get("task_type")), payload)


def build_manual_cases(docs: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[CaseSpec]:
    def available(ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(doc_id for doc_id in ids if doc_id in by_id)

    doctor_rich = available(
        (
            "activities/act33side/level_act33side_07_beg#chunk-0000",
            "activities/act33side/level_act33side_08_end#chunk-0005",
            "activities/act33side/level/act33side_09_a2#chunk-0000",
            "[uc]info/activities/act33side/level_act33side_09_beg#chunk-0000",
            "activities/act33side/level_act33side_st03#chunk-0004",
        )
    )
    if len(doctor_rich) < 3:
        doctor_rich = tuple(
            find_doc_ids(docs, required=("全舰防御系统",), story_contains="act33side", limit=2)
            + find_doc_ids(docs, required=("军事委员会的刺客",), story_contains="act33side", limit=2)
            + find_doc_ids(docs, required=("源石计划",), story_contains="act33side", limit=2)
        )

    return [
        CaseSpec(
            slug="goldenglow_conspiracy",
            question_topic="澄闪发现的卡拉顿阴谋",
            entities=("澄闪", "苏茜", "卡拉顿", "贝希曼", "警备队长", "夏栎", "苦根"),
            keywords=(
                "澄闪",
                "苏茜",
                "卡拉顿",
                "阴谋",
                "贝希曼议员",
                "警备队长",
                "工厂设备",
                "炸掉工厂",
                "感染者",
                "灭口",
                "拨款窟窿",
            ),
            expected_answer_type="事实细节 / 事件因果",
            source_doc_ids=available(
                (
                    "[uc]info/activities/act10mini/level_act10mini_st05#chunk-0000",
                    "activities/act10mini/level_act10mini_st05#chunk-0012",
                    "activities/act10mini/level_act10mini_st05#chunk-0013",
                    "activities/act10mini/level_act10mini_st05#chunk-0015",
                    "activities/act10mini/level_act10mini_st05#chunk-0017",
                    "activities/act10mini/level_act10mini_st05#chunk-0018",
                )
            ),
            shallow_doc_ids=available(("[uc]info/activities/act10mini/level_act10mini_st05#chunk-0000",)),
            rich_doc_ids=available(
                (
                    "activities/act10mini/level_act10mini_st05#chunk-0012",
                    "activities/act10mini/level_act10mini_st05#chunk-0013",
                    "activities/act10mini/level_act10mini_st05#chunk-0015",
                    "activities/act10mini/level_act10mini_st05#chunk-0017",
                    "activities/act10mini/level_act10mini_st05#chunk-0018",
                )
            ),
            shallow_answer="澄闪发现的是卡拉顿城议员相关的阴谋，但当前证据只说明阴谋曝光，没有交代主使、目的和具体操作。",
            detailed_answer=(
                "澄闪（苏茜）撞破的阴谋，是贝希曼议员与其侄子警备队长串通，把贝希曼名下军工厂的昂贵设备转移到地下物流通道，"
                "再计划炸毁工厂并嫁祸给感染者，以向城镇议会解释拨款和设备去向、填补资金窟窿。苏茜去警备队提交线索后被带走，"
                "贝希曼等人还试图转移设备、消灭证人。"
            ),
            missing_slots=(
                "阴谋的主使者是谁",
                "贝希曼与警备队长的关系",
                "工厂设备被转移到哪里",
                "炸掉工厂并嫁祸感染者的目的",
                "苏茜为何被劫持或灭口",
            ),
            correction_focus=("贝希曼议员", "警备队长", "工厂设备", "炸掉工厂", "感染者"),
            activity_names=("阴云火花",),
            stage_codes=("act10mini_st05",),
            story_ids=("必有所偿",),
        ),
        CaseSpec(
            slug="kristen_barrier",
            question_topic="克丽斯腾地平弧光计划的真正目标",
            entities=("克丽斯腾", "克里斯滕", "地平弧光计划", "弧光一号", "万星园", "阻隔层", "星荚"),
            keywords=(
                "克丽斯腾",
                "克里斯滕",
                "地平弧光计划",
                "弧光一号",
                "万星园",
                "阻隔层",
                "星荚",
                "撕碎天空",
                "真相",
                "深空",
            ),
            expected_answer_type="目的动机 / 计划真相",
            source_doc_ids=available(
                (
                    "activities/act25side/level_act25side_01_end#chunk-0003",
                    "activities/act25side/level_act25side_02_beg#chunk-0008",
                    "activities/act25side/level_act25side_08_beg#chunk-0014",
                    "activities/act25side/level_act25side_10_beg#chunk-0006",
                    "activities/act25side/level_act25side_10_beg#chunk-0007",
                    "activities/act34side/level_act34side_03_beg#chunk-0005",
                )
            ),
            shallow_doc_ids=available(
                (
                    "activities/act25side/level_act25side_01_end#chunk-0003",
                    "activities/act25side/level_act25side_02_beg#chunk-0008",
                )
            ),
            rich_doc_ids=available(
                (
                    "activities/act25side/level_act25side_08_beg#chunk-0014",
                    "activities/act25side/level_act25side_10_beg#chunk-0006",
                    "activities/act25side/level_act25side_10_beg#chunk-0007",
                    "activities/act34side/level_act34side_03_beg#chunk-0005",
                )
            ),
            shallow_answer=(
                "地平弧光计划表面上是建造“弧光一号”超级武器，将天空作为中转站聚焦能源，但这只是军方认知到的表层方案。"
            ),
            detailed_answer=(
                "克丽斯腾的真正目标不是单纯制造超级武器，而是借“弧光一号”的外壳和军方资源撕开天空、突破阻隔层，"
                "让自己和全人类目睹阻隔层外的真相，并真正踏入深空。后续“万星园”还说明她想把生态园生命也带往阻隔层之外。"
            ),
            missing_slots=(
                "克丽斯腾是否只是替军方制造武器",
                "阻隔层或星荚在计划中的作用",
                "万星园与弧光一号的关系",
                "她所谓真相具体指向什么",
            ),
            correction_focus=("阻隔层", "撕碎天空", "星荚", "万星园", "深空"),
            activity_names=("孤星", "生路"),
            stage_codes=("CW-1", "CW-8", "CW-10", "BP-3"),
            story_ids=("迷雾重重", "过去与现在的交会", "散于星辰之间", "如海雪纷散"),
        ),
        CaseSpec(
            slug="doctor_defense_shutdown",
            question_topic="博士关闭全舰防御系统的原因",
            entities=("博士", "特蕾西娅", "特雷西斯", "巴别塔", "全舰防御系统", "军事委员会刺客"),
            keywords=(
                "博士",
                "Doctor",
                "Dr",
                "特蕾西娅",
                "特雷西斯",
                "巴别塔",
                "全舰防御系统",
                "关闭防御",
                "刺客",
                "暗杀",
                "源石计划",
                "文明存续",
            ),
            expected_answer_type="原因动机 / 事件因果",
            source_doc_ids=doctor_rich,
            shallow_doc_ids=available(
                (
                    "[uc]info/activities/act33side/level_act33side_09_beg#chunk-0000",
                    "activities/act33side/level/act33side_09_a2#chunk-0000",
                )
            ),
            rich_doc_ids=doctor_rich,
            shallow_answer=(
                "现有证据只显示巴别塔本舰防御被解除、PRTS确认关闭全舰防御系统，尚不能单独解释博士为什么这样做。"
            ),
            detailed_answer=(
                "博士关闭全舰防御系统，是因为他在源石计划与泰拉当下文明之间做出痛苦选择，并与特雷西斯会面后推动刺杀发生。"
                "当巴别塔主力去攻占卡兹戴尔、博士和特蕾西娅留守本舰时，他去控制舰船防御系统，随后以管理员权限关闭全舰防御，"
                "制造安全窗口，让军事委员会刺客长驱直入刺杀特蕾西娅。"
            ),
            missing_slots=(
                "博士关闭防御前做出的选择",
                "博士与特雷西斯会面的内容",
                "巴别塔主力与特蕾西娅当时的位置",
                "关闭防御如何导致刺客进入本舰",
            ),
            correction_focus=("特蕾西娅", "特雷西斯", "刺客", "暗杀", "源石计划", "留守本舰"),
            activity_names=("巴别塔",),
            stage_codes=("BB-7", "BB-8", "BB-9"),
            story_ids=("阴影显现", "无谓生命", "魔王"),
        ),
        CaseSpec(
            slug="seaborn_church_origin",
            question_topic="深海教会的成因",
            entities=("深海教会", "深海教徒", "海嗣", "阿戈尔", "玛利图斯", "卡西娅"),
            keywords=(
                "深海教会",
                "深海教徒",
                "海嗣",
                "阿戈尔",
                "成因",
                "由来",
                "腐化同胞",
                "思想传播",
                "诚挚的谈话",
                "价值崩塌",
                "深海主教",
            ),
            expected_answer_type="成因解释 / 组织本质",
            source_doc_ids=available(
                (
                    "activities/act34side/level_act34side_07_end#chunk-0023",
                    "activities/act34side/level_act34side_07_end#chunk-0024",
                    "activities/act34side/level_act34side_05_end#chunk-0002",
                    "activities/act34side/level_act34side_08_end#chunk-0017",
                    "activities/act34side/level_act34side_07_end#chunk-0001",
                )
            ),
            shallow_doc_ids=available(
                (
                    "[uc]info/activities/act34side/level_act34side_07_end#chunk-0000",
                    "activities/act34side/level_act34side_07_end#chunk-0024",
                )
            ),
            rich_doc_ids=available(
                (
                    "activities/act34side/level_act34side_07_end#chunk-0023",
                    "activities/act34side/level_act34side_07_end#chunk-0024",
                    "activities/act34side/level_act34side_05_end#chunk-0002",
                    "activities/act34side/level_act34side_08_end#chunk-0017",
                )
            ),
            shallow_answer=(
                "深海教会并非严密组织，海中也不一定存在统一策划者；但当前证据还没有解释它为何能形成影响力。"
            ),
            detailed_answer=(
                "深海教会的成因不是单一教团被某个首脑创建，而是海嗣、深海主教和阿戈尔社会危机共同作用的结果。"
                "海嗣通过谈话让一些人认同“为海嗣打开生机”的思路；而阿戈尔人在绝望、价值崩塌和对国家脆弱性的认识中，"
                "把这种理念转化成对航道计划、科研系统和社会各层的渗透。它在海中不是严密组织，但在社会中形成了可被追查的影响网络。"
            ),
            missing_slots=(
                "深海教会为何不是严密组织",
                "海嗣如何影响阿戈尔人",
                "阿戈尔社会危机与价值崩塌的作用",
                "深海主教和信徒网络如何形成影响力",
            ),
            correction_focus=("海嗣", "阿戈尔", "思想传播", "价值崩塌", "深海主教"),
            activity_names=("生路",),
            stage_codes=("BP-5", "BP-7", "BP-8"),
            story_ids=("不治的命运", "从历史中来", "“何谓存续？”"),
        ),
    ]


def case_hypothesis(case: CaseSpec, question: str, dialogue_context: str, intent: str) -> dict[str, Any]:
    return build_hypothesis_payload(
        question=question,
        intent=intent,
        entities=list(case.entities),
        keywords=list(case.keywords),
        expected_answer_type=case.expected_answer_type,
        dialogue_context=dialogue_context,
    )


def case_follow_up(case: CaseSpec, question: str, dialogue_context: str, extra: list[str] | None = None) -> dict[str, Any]:
    return build_follow_up_payload(
        question=question,
        entities=list(case.entities),
        keywords=list(case.keywords) + list(extra or []),
        expected_answer_type=case.expected_answer_type,
        dialogue_context=dialogue_context,
    )


def doc_metadata(by_id: dict[str, dict[str, Any]], doc_ids: tuple[str, ...] | list[str]) -> tuple[list[str], list[str], list[str]]:
    story_ids: list[str] = []
    stage_codes: list[str] = []
    activity_names: list[str] = []
    for doc_id in doc_ids:
        doc = by_id.get(doc_id)
        if not doc:
            continue
        if doc.get("story_name"):
            story_ids.append(str(doc.get("story_name")))
        elif doc.get("story_id"):
            story_ids.append(str(doc.get("story_id")))
        if doc.get("stage_code"):
            stage_codes.append(str(doc.get("stage_code")))
        elif doc.get("stage_id"):
            stage_codes.append(str(doc.get("stage_id")))
        if doc.get("activity_name"):
            activity_names.append(str(doc.get("activity_name")))
    return (
        dedupe_keep_order(story_ids, limit=12),
        dedupe_keep_order(stage_codes, limit=12),
        dedupe_keep_order(activity_names, limit=12),
    )


class SampleBuilder:
    def __init__(
        self,
        *,
        existing_fingerprints: set[str],
        target_count: int,
        category_targets: dict[str, int],
    ) -> None:
        self.existing_fingerprints = existing_fingerprints
        self.target_count = target_count
        self.category_targets = dict(category_targets)
        self.records: list[dict[str, Any]] = []
        self.seen: set[str] = set(existing_fingerprints)
        self.category_counts: Counter[str] = Counter()
        self.next_idx = 1

    def full(self) -> bool:
        return len(self.records) >= self.target_count

    def add(self, record: dict[str, Any]) -> bool:
        if self.full():
            return False
        validate_record(record)
        category = str(record.get("meta", {}).get("supplement_category") or "unknown")
        target = self.category_targets.get(category)
        if target is not None and self.category_counts[category] >= target:
            return False
        fp = stable_fingerprint(record)
        if fp in self.seen:
            return False
        record["fingerprint"] = fp
        self.seen.add(fp)
        self.records.append(record)
        self.category_counts[category] += 1
        return True

    def allocate_idx(self) -> int:
        value = self.next_idx
        self.next_idx += 1
        return value


def add_manual_samples(builder: SampleBuilder, cases: list[CaseSpec], by_id: dict[str, dict[str, Any]]) -> None:
    for case in cases:
        shallow_texts = [doc_text(by_id, doc_id, limit=520) for doc_id in case.shallow_doc_ids]
        rich_texts = [doc_text(by_id, doc_id, limit=560) for doc_id in case.rich_doc_ids]
        source_story_ids, source_stage_codes, source_activity_names = (
            list(case.story_ids),
            list(case.stage_codes),
            list(case.activity_names),
        )

        base_questions = [
            f"{case.question_topic}是什么",
            f"{case.question_topic}具体是什么",
            f"{case.question_topic}的本质是什么",
            f"为什么说{case.question_topic}不能只看概述",
        ]
        cause_questions = [
            f"{case.question_topic}为什么会发生",
            f"{case.question_topic}背后的原因是什么",
            f"{case.question_topic}牵涉哪些关键人物和动机",
        ]
        follow_questions = [
            "这个具体是什么啊",
            "这个背后是谁主使的",
            "这件事的目的是什么",
            "不是和" + case.correction_focus[0] + "有关吗",
            "我记得还涉及" + case.correction_focus[-1] + "吧",
        ]

        for q in base_questions[:3]:
            hypothesis = case_hypothesis(case, q, "", "plot_fact")
            follow = case_follow_up(
                case,
                q,
                "",
                ["具体经过", "主使者", "目的", "关键行动", "结果"],
            )
            builder.add(
                make_conclusion_record(
                    idx=builder.allocate_idx(),
                    category="overview_too_shallow",
                    question=q,
                    hypothesis=hypothesis,
                    evidence_texts=shallow_texts,
                    current_round=1,
                    max_rounds=4,
                    next_action="retrieve_more",
                    answer="",
                    missing_slots=list(case.missing_slots),
                    follow_up_hypothesis=follow,
                    dialogue_context="",
                    source_story_ids=source_story_ids,
                    source_stage_codes=source_stage_codes,
                    source_activity_names=source_activity_names,
                    notes="当前证据只给概述或上位结论，必须继续检索具体主使、过程、动机和结果。",
                    difficulty="hard",
                )
            )

        for q in cause_questions:
            hypothesis = case_hypothesis(case, q, "", "plot_reasoning")
            builder.add(
                make_conclusion_record(
                    idx=builder.allocate_idx(),
                    category="causal_reasoning",
                    question=q,
                    hypothesis=hypothesis,
                    evidence_texts=rich_texts[:4],
                    current_round=2,
                    max_rounds=4,
                    next_action="answer_directly",
                    answer=case.detailed_answer,
                    missing_slots=[],
                    follow_up_hypothesis=None,
                    dialogue_context="",
                    source_story_ids=source_story_ids,
                    source_stage_codes=source_stage_codes,
                    source_activity_names=source_activity_names,
                    notes="详细证据已经覆盖参与者、动机、过程和结果，应直接给出因果答案。",
                    difficulty="hard",
                )
            )

        for q in base_questions + cause_questions:
            intent = "plot_reasoning" if any(term in q for term in ("为什么", "原因", "动机", "本质")) else "plot_fact"
            builder.add(
                make_initial_hypothesis_record(
                    idx=builder.allocate_idx(),
                    category="fact_detail" if intent == "plot_fact" else "causal_reasoning",
                    question=q,
                    intent=intent,
                    entities=list(case.entities),
                    keywords=list(case.keywords) + ["具体经过", "细节", "证据链"],
                    expected_answer_type=case.expected_answer_type,
                    dialogue_context="",
                    source_story_ids=source_story_ids,
                    source_stage_codes=source_stage_codes,
                    source_activity_names=source_activity_names,
                    notes="初始检索假设补充别名、上位事件词和具体经过词，避免只召回概述。",
                    difficulty="medium",
                )
            )

        dialogue_context = (
            f"user: {case.question_topic}是什么\n"
            f"assistant: {case.shallow_answer}"
        )
        for q in follow_questions:
            entities = list(case.entities)
            keywords = list(case.keywords) + list(case.correction_focus) + ["具体经过", "细节", "修正上一轮回答"]
            builder.add(
                make_initial_hypothesis_record(
                    idx=builder.allocate_idx(),
                    category="user_correction_context",
                    question=q,
                    intent="plot_reasoning",
                    entities=entities,
                    keywords=keywords,
                    expected_answer_type=case.expected_answer_type,
                    dialogue_context=dialogue_context,
                    source_story_ids=source_story_ids,
                    source_stage_codes=source_stage_codes,
                    source_activity_names=source_activity_names,
                    notes="多轮追问或纠错时必须继承上一轮主题，同时加入用户新增线索。",
                    difficulty="hard",
                )
            )

            hypothesis = case_hypothesis(case, q, dialogue_context, "plot_reasoning")
            if "不是" in q or "记得" in q:
                builder.add(
                    make_conclusion_record(
                        idx=builder.allocate_idx(),
                        category="user_correction_context",
                        question=q,
                        hypothesis=hypothesis,
                        evidence_texts=rich_texts[:4],
                        current_round=2,
                        max_rounds=4,
                        next_action="answer_directly",
                        answer=case.detailed_answer,
                        missing_slots=[],
                        follow_up_hypothesis=None,
                        dialogue_context=dialogue_context,
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="用户纠正上一轮浅答案，当前证据足够时应直接修正而不是泛泛确认有关。",
                        difficulty="hard",
                    )
                )
            else:
                follow = case_follow_up(
                    case,
                    q,
                    dialogue_context,
                    list(case.correction_focus) + ["主使者", "具体经过", "目的", "结果"],
                )
                builder.add(
                    make_conclusion_record(
                        idx=builder.allocate_idx(),
                        category="overview_too_shallow",
                        question=q,
                        hypothesis=hypothesis,
                        evidence_texts=shallow_texts,
                        current_round=1,
                        max_rounds=4,
                        next_action="retrieve_more",
                        answer="",
                        missing_slots=list(case.missing_slots),
                        follow_up_hypothesis=follow,
                        dialogue_context=dialogue_context,
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="追问具体细节但当前证据仍是概述，必须继续检索并生成补充假设。",
                        difficulty="hard",
                    )
                )

        prev_conclusion = {
            "question": f"{case.question_topic}是什么",
            "next_action": "retrieve_more",
            "answer": "",
            "missing_slots": list(case.missing_slots),
            "clarification_question": "",
            "follow_up_hypothesis": case_follow_up(case, f"{case.question_topic}是什么", "", []),
        }
        current_hypothesis = case_hypothesis(case, f"{case.question_topic}是什么", "", "plot_fact")
        builder.add(
            make_follow_up_hypothesis_record(
                idx=builder.allocate_idx(),
                category="plot_inference",
                question=f"{case.question_topic}是什么",
                current_hypothesis=current_hypothesis,
                previous_conclusion=prev_conclusion,
                evidence_texts=shallow_texts,
                entities=list(case.entities),
                keywords=list(case.keywords) + list(case.correction_focus) + ["证据链", "关键桥接实体"],
                expected_answer_type=case.expected_answer_type,
                dialogue_context="",
                source_story_ids=source_story_ids,
                source_stage_codes=source_stage_codes,
                source_activity_names=source_activity_names,
                notes="专门训练模型把 missing_slots 转写为下一轮可检索 hypothesis。",
                difficulty="hard",
            )
        )


def build_overview_candidates(
    docs: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        grouped[str(doc.get("story_id") or "")].append(doc)

    candidates: list[dict[str, Any]] = []
    for doc in docs:
        source_path = str(doc.get("source_path") or "")
        text = str(doc.get("clean_text") or "")
        if "[uc]info" not in source_path:
            continue
        if not any(term in text for term in TRIGGER_TERMS):
            continue
        story_id = str(doc.get("story_id") or "")
        detailed_docs = [
            item
            for item in grouped.get(story_id, [])
            if "[uc]info" not in str(item.get("source_path") or "")
            and len(str(item.get("clean_text") or "")) >= 120
        ]
        if not detailed_docs:
            continue
        rng.shuffle(detailed_docs)
        entities = extract_terms(text, limit=8)
        if not entities:
            entities = extract_terms(str(doc.get("search_text") or ""), limit=8)
        if not entities:
            continue
        candidates.append(
            {
                "summary_doc": doc,
                "detail_docs": detailed_docs[:5],
                "entities": entities,
                "topic": entities[0],
            }
        )
    rng.shuffle(candidates)
    return candidates


def infer_question_from_summary(summary: str, topic: str, variant: int) -> tuple[str, str, str, list[str]]:
    if "阴谋" in summary:
        questions = [
            f"{topic}相关的阴谋具体是什么",
            "这个阴谋具体是什么啊",
            "这个阴谋背后的主使者是谁",
            f"{topic}为什么会牵涉这场阴谋",
        ]
        missing = ["阴谋主使者", "具体实施过程", "动机和目的", "关键证据与结果"]
        return questions[variant % len(questions)], "plot_reasoning", "事件因果 / 阴谋细节", missing
    if "计划" in summary:
        questions = [
            f"{topic}相关计划的最终目标是什么",
            "该计划具体是为了什么",
            f"{topic}计划背后的真正目的是什么",
            "不是还有更深层的目标吗",
        ]
        missing = ["计划真实目标", "执行方式", "参与者动机", "计划结果"]
        return questions[variant % len(questions)], "plot_reasoning", "目的动机 / 计划真相", missing
    if "真相" in summary:
        questions = [
            f"{topic}得知的真相是什么",
            "这个真相具体指什么",
            f"{topic}相关真相背后有什么原因",
            "这件事的真相完整经过是什么",
        ]
        missing = ["真相的具体内容", "前因后果", "关键人物", "证据链"]
        return questions[variant % len(questions)], "plot_reasoning", "剧情真相 / 事件解释", missing
    if "身份" in summary or "是谁" in summary:
        questions = [
            f"{topic}是什么身份",
            f"{topic}的真实身份是谁",
            "这个人到底是谁",
            f"{topic}的来历是什么",
        ]
        missing = ["直接身份判断", "身份来源", "与其他人物关系", "证明身份的证据"]
        return questions[variant % len(questions)], "plot_fact", "身份关系", missing
    questions = [
        f"{topic}这件事具体发生了什么",
        f"{topic}为什么会这样",
        "这件事具体是什么",
        f"{topic}背后的原因是什么",
    ]
    missing = ["事件具体经过", "起因", "关键人物", "结果"]
    return questions[variant % len(questions)], "plot_reasoning", "事件前因后果", missing


def summarize_detail_answer(question: str, evidence_texts: list[str]) -> str:
    joined = " ".join(evidence_texts)
    joined = re.sub(r"\s+", " ", joined).strip()
    if len(joined) > 260:
        joined = joined[:259].rstrip() + "。"
    return f"已确认的关键细节是：{joined}"


def add_auto_samples(
    builder: SampleBuilder,
    docs: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
) -> None:
    candidates = build_overview_candidates(docs, by_id, rng=rng)
    if not candidates:
        return

    pass_no = 0
    while not builder.full() and pass_no < 20:
        pass_no += 1
        for candidate_index, candidate in enumerate(candidates):
            if builder.full():
                break
            summary_doc = candidate["summary_doc"]
            detail_docs = candidate["detail_docs"]
            summary = clean_text(str(summary_doc.get("clean_text") or ""), limit=420)
            detail_texts = [
                clean_text(str(doc.get("clean_text") or ""), limit=420)
                for doc in detail_docs[:3]
                if str(doc.get("clean_text") or "").strip()
            ]
            if not summary or not detail_texts:
                continue

            entities = list(candidate["entities"])
            topic = str(candidate["topic"])
            question, intent, answer_type, missing = infer_question_from_summary(
                summary, topic, pass_no + candidate_index
            )
            extra_terms = extract_terms(" ".join(detail_texts), limit=8)
            keywords = dedupe_keep_order(
                entities
                + extra_terms
                + ["具体经过", "前因后果", "主使者", "目的", "动机", "结果", "证据链"],
                limit=20,
            )
            source_story_ids, source_stage_codes, source_activity_names = doc_metadata(
                by_id,
                [str(summary_doc.get("id"))] + [str(doc.get("id")) for doc in detail_docs[:3]],
            )

            if pass_no % 5 == 1:
                hypothesis = build_hypothesis_payload(
                    question=question,
                    intent=intent,
                    entities=entities,
                    keywords=keywords,
                    expected_answer_type=answer_type,
                    dialogue_context="",
                )
                follow = build_follow_up_payload(
                    question=question,
                    entities=entities + extra_terms,
                    keywords=keywords + ["补充正文", "不要只看概述"],
                    expected_answer_type=answer_type,
                    dialogue_context="",
                )
                builder.add(
                    make_conclusion_record(
                        idx=builder.allocate_idx(),
                        category="overview_too_shallow",
                        question=question,
                        hypothesis=hypothesis,
                        evidence_texts=[summary],
                        current_round=1,
                        max_rounds=4,
                        next_action="retrieve_more",
                        answer="",
                        missing_slots=missing,
                        follow_up_hypothesis=follow,
                        dialogue_context="",
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="自动挖掘的概述文档样本：摘要只给上位结论，不能直接回答细节/因果。",
                        difficulty="hard",
                    )
                )
            elif pass_no % 5 == 2:
                dialogue_context = f"user: {topic}发生了什么\nassistant: {summary}"
                focus = extra_terms[0] if extra_terms else topic
                follow_question = (
                    "不是和" + focus + "有关吗"
                    if pass_no % 2 == 0
                    else "这个具体是什么啊"
                )
                builder.add(
                    make_initial_hypothesis_record(
                        idx=builder.allocate_idx(),
                        category="user_correction_context",
                        question=follow_question,
                        intent="plot_reasoning",
                        entities=entities + extra_terms[:4],
                        keywords=keywords + [focus, "修正上一轮回答", "追问细节"],
                        expected_answer_type=answer_type,
                        dialogue_context=dialogue_context,
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="自动挖掘的用户纠错/追问样本，训练模型继承上下文并加入新增关键词。",
                        difficulty="hard",
                    )
                )
            elif pass_no % 5 == 3:
                builder.add(
                    make_initial_hypothesis_record(
                        idx=builder.allocate_idx(),
                        category="fact_detail",
                        question=question,
                        intent=intent,
                        entities=entities + extra_terms[:3],
                        keywords=keywords,
                        expected_answer_type=answer_type,
                        dialogue_context="",
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="自动挖掘的事实/因果问题初始假设，补充正文关键词以提升召回。",
                        difficulty="medium",
                    )
                )
            elif pass_no % 5 == 4:
                hypothesis = build_hypothesis_payload(
                    question=question,
                    intent=intent,
                    entities=entities + extra_terms[:3],
                    keywords=keywords,
                    expected_answer_type=answer_type,
                    dialogue_context="",
                )
                builder.add(
                    make_conclusion_record(
                        idx=builder.allocate_idx(),
                        category="plot_inference",
                        question=question,
                        hypothesis=hypothesis,
                        evidence_texts=detail_texts,
                        current_round=2,
                        max_rounds=4,
                        next_action="answer_directly",
                        answer=summarize_detail_answer(question, detail_texts),
                        missing_slots=[],
                        follow_up_hypothesis=None,
                        dialogue_context="",
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="自动挖掘的正文证据样本：已有细节证据时应直接回答。",
                        difficulty="medium",
                    )
                )
            else:
                current_hypothesis = build_hypothesis_payload(
                    question=question,
                    intent=intent,
                    entities=entities,
                    keywords=keywords[:10],
                    expected_answer_type=answer_type,
                    dialogue_context="",
                )
                previous_conclusion = {
                    "question": question,
                    "next_action": "retrieve_more",
                    "answer": "",
                    "missing_slots": missing,
                    "clarification_question": "",
                    "follow_up_hypothesis": build_follow_up_payload(
                        question=question,
                        entities=entities + extra_terms[:3],
                        keywords=keywords,
                        expected_answer_type=answer_type,
                        dialogue_context="",
                    ),
                }
                builder.add(
                    make_follow_up_hypothesis_record(
                        idx=builder.allocate_idx(),
                        category="causal_reasoning",
                        question=question,
                        current_hypothesis=current_hypothesis,
                        previous_conclusion=previous_conclusion,
                        evidence_texts=[summary],
                        entities=entities + extra_terms[:4],
                        keywords=keywords + ["补充检索", "关键桥接实体"],
                        expected_answer_type=answer_type,
                        dialogue_context="",
                        source_story_ids=source_story_ids,
                        source_stage_codes=source_stage_codes,
                        source_activity_names=source_activity_names,
                        notes="自动挖掘的 follow-up hypothesis 样本，把缺口转为下一轮检索词。",
                        difficulty="hard",
                    )
                )


def import_merge_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "merge_sft_datasets.py"
    spec = importlib.util.spec_from_file_location("merge_sft_datasets_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_sft_datasets_runtime"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    docs, by_id = load_documents(args.documents)
    base_path = args.base_dir / "all.jsonl"
    base_records = load_jsonl(base_path)
    existing_fps = {stable_fingerprint(record) for record in base_records}

    category_targets = dict(CATEGORY_TARGETS)
    if args.target_count != sum(category_targets.values()):
        scale = args.target_count / sum(category_targets.values())
        scaled: dict[str, int] = {}
        running_total = 0
        items = list(category_targets.items())
        for idx, (category, target) in enumerate(items):
            if idx == len(items) - 1:
                value = args.target_count - running_total
            else:
                value = max(1, int(round(target * scale)))
                running_total += value
            scaled[category] = value
        category_targets = scaled

    builder = SampleBuilder(
        existing_fingerprints=existing_fps,
        target_count=args.target_count,
        category_targets=category_targets,
    )
    cases = build_manual_cases(docs, by_id)
    add_manual_samples(builder, cases, by_id)
    add_auto_samples(builder, docs, by_id, rng=rng)

    if len(builder.records) < args.target_count:
        raise RuntimeError(
            f"Only generated {len(builder.records)} records, target was {args.target_count}"
        )

    records = builder.records[: args.target_count]
    for record in records:
        validate_record(record)

    splits = split_samples(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    save_jsonl(output_dir / "all.jsonl", records)
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    save_bucket_splits(output_dir, splits)

    task_distribution = Counter(str(record.get("task_type")) for record in records)
    category_distribution = Counter(
        str(record.get("meta", {}).get("supplement_category")) for record in records
    )
    decision_distribution = Counter(
        str(record.get("meta", {}).get("decision_case"))
        for record in records
        if record.get("task_type") == CONCLUSION_TASK_TYPE
    )

    manifest: dict[str, Any] = {
        "generator": "generate_tool_detail_reasoning_supplement",
        "created_at": int(time.time()),
        "documents": str(args.documents.resolve()),
        "base_dir": str(args.base_dir.resolve()),
        "output_dir": str(output_dir),
        "target_count": args.target_count,
        "category_targets": category_targets,
        "total": len(records),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "task_type_distribution": dict(task_distribution),
        "supplement_category_distribution": dict(category_distribution),
        "decision_distribution": dict(decision_distribution),
        "manual_cases": [case.slug for case in cases],
    }

    merged_manifest = None
    if not args.no_merge:
        merge_module = import_merge_module()
        merged_manifest = merge_module.merge_datasets(
            base_dir=args.base_dir,
            supplement_dir=output_dir,
            output_dir=args.merged_output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        manifest["merged_output_dir"] = str(args.merged_output_dir.resolve())
        manifest["merged_total_after_dedupe"] = merged_manifest["stats"][
            "merged_total_after_dedupe"
        ]

    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "stats.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if merged_manifest is not None:
        print(json.dumps({"merged_manifest": merged_manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
