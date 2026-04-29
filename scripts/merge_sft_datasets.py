#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import dedupe_samples, split_samples  # noqa: E402


DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "sft_data" / "teacher_v2"
DEFAULT_SUPPLEMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sft_data" / "prompt_supplement_v2"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "sft_data" / "teacher_v2_plus_prompt_supplement_v2"
)
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
CORE_JSON_TASK_TYPES = {
    INITIAL_HYPOTHESIS_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    CONCLUSION_TASK_TYPE,
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
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
EMPTY_HYPOTHESIS_PATTERNS = (
    "当前假设文档(JSON):{}",
    "当前假设文档:{}",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def with_dataset_source(records: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        meta = dict(payload.get("meta") or {})
        sources = meta.get("source_datasets")
        if isinstance(sources, list):
            merged_sources = [str(item) for item in sources if str(item).strip()]
        else:
            merged_sources = []
        if dataset_name not in merged_sources:
            merged_sources.append(dataset_name)
        meta["source_datasets"] = merged_sources
        payload["meta"] = meta
        output.append(payload)
    return output


def bucket_of(record: dict[str, Any]) -> str:
    return str(record.get("bucket") or record.get("meta", {}).get("category") or "tool")


def save_bucket_splits(output_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    for bucket in ("style", "knowledge", "tool"):
        bucket_dir = output_dir / bucket
        bucket_records = {
            split: [record for record in records if bucket_of(record) == bucket]
            for split, records in splits.items()
        }
        all_records = bucket_records["train"] + bucket_records["val"] + bucket_records["test"]
        save_jsonl(bucket_dir / "all.jsonl", all_records)
        save_jsonl(bucket_dir / "train.jsonl", bucket_records["train"])
        save_jsonl(bucket_dir / "val.jsonl", bucket_records["val"])
        save_jsonl(bucket_dir / "test.jsonl", bucket_records["test"])


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
    direct_value = str(payload.get("expected_answer_type") or "").strip()
    if direct_value:
        return direct_value
    constraints = payload.get("constraints")
    if isinstance(constraints, dict):
        return str(constraints.get("expected_answer_type") or "").strip()
    return ""


def _normalize_hypothesis_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if any(field not in payload for field in INITIAL_HYPOTHESIS_SCHEMA_FIELDS if field != "dialogue_context"):
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

    return {
        "question": question,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
    }


def _normalize_follow_up_hypothesis_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
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


def _normalize_conclusion_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    next_action = str(payload.get("next_action") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    missing_slots = _normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()

    if not question or next_action not in RETRIEVAL_ACTIONS:
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


def _parse_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _contains_empty_current_hypothesis(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = re.sub(r"\s+", "", str(message.get("content") or ""))
        if any(pattern in content for pattern in EMPTY_HYPOTHESIS_PATTERNS):
            return True
    return False


def _normalize_tool_call_arguments(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return None
        payload = json.loads(json.dumps(tool_call, ensure_ascii=False))
        function = payload.get("function")
        if not isinstance(function, dict):
            return None
        name = str(function.get("name") or "")
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                return None
            if name == "build_hypothesis":
                if "intent" in args:
                    args["intent"] = _normalize_intent(args.get("intent"))
                    if not args["intent"]:
                        return None
            if name == "detect_intent" and "intent" in args:
                args["intent"] = _normalize_intent(args.get("intent"))
            function["arguments"] = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        normalized_calls.append(payload)
    return normalized_calls


def _normalize_message_for_record(message: dict[str, Any]) -> dict[str, Any] | None:
    normalized = {
        "role": message.get("role"),
        "content": message.get("content", ""),
    }
    if "name" in message and message.get("name") is not None:
        normalized["name"] = message.get("name")
    if message.get("tool_calls"):
        tool_calls = _normalize_tool_call_arguments(message["tool_calls"])
        if tool_calls is None:
            return None
        normalized["tool_calls"] = tool_calls
    return normalized


def _system_positions(messages: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, message in enumerate(messages) if message.get("role") == "system"]


def _has_legacy_tool_trace(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _normalize_record(
    record: dict[str, Any],
    *,
    source_name: str,
    stats: Counter,
) -> dict[str, Any] | None:
    payload = json.loads(json.dumps(record, ensure_ascii=False))
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        stats["dropped_invalid_messages"] += 1
        return None

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            stats["dropped_invalid_messages"] += 1
            return None
        normalized_message = _normalize_message_for_record(message)
        if normalized_message is None:
            stats["dropped_invalid_tool_call"] += 1
            return None
        normalized_messages.append(normalized_message)

    payload["messages"] = normalized_messages
    task_type = str(payload.get("task_type") or "")

    system_positions = _system_positions(normalized_messages)
    if len(system_positions) > 1:
        stats["dropped_multi_system"] += 1
        return None
    if system_positions and system_positions[0] != 0:
        stats["dropped_system_not_first"] += 1
        return None
    if normalized_messages[-1].get("role") != "assistant":
        stats["dropped_last_not_assistant"] += 1
        return None

    if source_name.startswith("prompt_supplement"):
        if task_type in CORE_JSON_TASK_TYPES and _contains_empty_current_hypothesis(normalized_messages):
            stats["dropped_empty_current_hypothesis"] += 1
            return None

    if task_type in CORE_JSON_TASK_TYPES:
        if _has_legacy_tool_trace(normalized_messages):
            stats["dropped_legacy_tool_trace"] += 1
            return None
        assistant = normalized_messages[-1]
        assistant_payload = _parse_json_object(assistant.get("content"))
        if assistant_payload is None:
            stats["dropped_invalid_json"] += 1
            return None
        normalized_assistant = None
        if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
            normalized_assistant = _normalize_hypothesis_payload(assistant_payload)
        elif task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
            normalized_assistant = _normalize_follow_up_hypothesis_payload(assistant_payload)
        elif task_type == CONCLUSION_TASK_TYPE:
            normalized_assistant = _normalize_conclusion_payload(assistant_payload)
        if normalized_assistant is None:
            stats["dropped_invalid_core_payload"] += 1
            return None
        assistant["content"] = json.dumps(normalized_assistant, ensure_ascii=False, separators=(",", ":"))
        if task_type == CONCLUSION_TASK_TYPE:
            meta = dict(payload.get("meta") or {})
            meta["decision_case"] = normalized_assistant["next_action"]
            payload["meta"] = meta
        return payload

    if source_name.startswith("prompt_supplement"):
        return payload

    if source_name == "teacher_v2":
        for message in normalized_messages:
            name = str(message.get("name") or "")
            if message.get("role") == "tool" and name == "build_hypothesis":
                tool_payload = _parse_json_object(message.get("content"))
                if tool_payload is None:
                    stats["dropped_invalid_build_hypothesis_json"] += 1
                    return None
                normalized_payload = _normalize_hypothesis_payload(tool_payload)
                if normalized_payload is None:
                    stats["dropped_invalid_build_hypothesis_payload"] += 1
                    return None
                message["content"] = json.dumps(
                    normalized_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                stats["normalized_teacher_build_hypothesis"] += 1
            if message.get("role") == "tool" and name == "detect_intent":
                tool_payload = _parse_json_object(message.get("content"))
                if tool_payload is None:
                    continue
                if "intent" in tool_payload:
                    normalized_intent = _normalize_intent(tool_payload.get("intent"))
                    if not normalized_intent:
                        stats["dropped_invalid_detect_intent_payload"] += 1
                        return None
                    tool_payload["intent"] = normalized_intent
                    message["content"] = json.dumps(
                        tool_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    stats["normalized_teacher_detect_intent"] += 1
        return payload

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge teacher_v2 with prompt_supplement_v2 into a cleaned SFT dataset."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def merge_datasets(
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

    base_records = with_dataset_source(load_jsonl(base_dir / "all.jsonl"), base_dir.name)
    supplement_records = with_dataset_source(load_jsonl(supplement_dir / "all.jsonl"), supplement_dir.name)

    normalization_stats: Counter[str] = Counter()
    cleaned_base_records = [
        normalized
        for record in base_records
        if (normalized := _normalize_record(record, source_name=base_dir.name, stats=normalization_stats))
        is not None
    ]
    cleaned_supplement_records = [
        normalized
        for record in supplement_records
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
        "generator": "merge_sft_datasets",
        "base_dir": str(base_dir),
        "supplement_dir": str(supplement_dir),
        "output_dir": str(output_dir),
        "stats": {
            "base_total": len(base_records),
            "supplement_total": len(supplement_records),
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
    stats = manifest["stats"]
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "stats.json", stats)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = merge_datasets(
        base_dir=args.base_dir,
        supplement_dir=args.supplement_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
