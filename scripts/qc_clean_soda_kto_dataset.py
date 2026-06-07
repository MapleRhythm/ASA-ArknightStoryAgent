#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any


ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}

QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
HYPOTHESIS_RE = re.compile(r"(?s)^hypothesis:\s*(\{.*?\})\s*(?:^round:|\Z)", re.MULTILINE)
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
ALLOWED_INTENTS = {
    "character_relation",
    "compare",
    "event_summary",
    "out_of_scope",
    "persona_chat",
    "plot_fact",
    "plot_reasoning",
    "timeline",
}
ALLOWED_QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def prompt_key(record: dict[str, Any], fallback: int = 0) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    return str(meta.get("prompt_key") or record.get("id") or fallback)


def prompt_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[0].get("value") or "")
    return ""


def response_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[-1].get("value") or "").strip()
    return ""


def extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_hypothesis(prompt: str) -> dict[str, Any]:
    match = HYPOTHESIS_RE.search(prompt or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def tokenize(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text or ""):
        normalized = re.sub(r"\s+", "", token).strip("，。！？；：、（）()[]【】《》“”\"'")
        if normalized and normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)
    return tokens


def build_fallback_follow_up(question: str, prompt: str, missing_slots: list[str]) -> dict[str, Any]:
    hypothesis = extract_hypothesis(prompt)
    entities = [str(item).strip() for item in hypothesis.get("entities", []) if str(item).strip()]
    keywords = [str(item).strip() for item in hypothesis.get("keywords", []) if str(item).strip()]
    for slot in missing_slots:
        keywords.extend(tokenize(slot)[:6])
    if not entities:
        entities = tokenize(question)[:6]
    if not keywords:
        keywords = tokenize(question)[:12]
    return {
        "question": question,
        "query_type": str(hypothesis.get("query_type") or "reasoning"),
        "entities": entities[:12],
        "keywords": list(dict.fromkeys(keywords))[:24],
        "expected_answer_type": str(hypothesis.get("expected_answer_type") or "剧情问答"),
        "dialogue_context": str(hypothesis.get("dialogue_context") or ""),
    }


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_prompt[prompt_key(record, index)].append(record)
    keys = list(by_prompt)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = max(1, int(round(len(records) * val_ratio))) if len(records) > 10 and val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for key in keys:
        bucket = by_prompt[key]
        if val_count < target_val:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def exact_conflict_keys(records: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    groups: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    for record in records:
        key = (str(record.get("task_type") or ""), prompt_text(record), response_text(record))
        groups[key].add(bool(record.get("kto_tag")))
    return {key for key, tags in groups.items() if len(tags) > 1}


def rebuild_retrieve_more(record: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(record, ensure_ascii=False))
    meta = output.get("meta") if isinstance(output.get("meta"), dict) else {}
    verifier = meta.get("api_verifier") if isinstance(meta.get("api_verifier"), dict) else {}
    prompt = prompt_text(output)
    question = extract_question(prompt)
    missing_slots = verifier.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = []
    missing_slots = [str(item).strip() for item in missing_slots if str(item).strip()][:8]
    payload = {
        "question": question,
        "next_action": "retrieve_more",
        "answer": "",
        "missing_slots": missing_slots,
        "clarification_question": "",
        "follow_up_hypothesis": build_fallback_follow_up(question, prompt, missing_slots),
    }
    conversations = output.get("conversations")
    if isinstance(conversations, list) and conversations:
        conversations[-1]["value"] = compact_json(payload)
    meta["qc_clean_reason"] = "rebuild_positive_retrieve_more_conflicting_with_rejected"
    output["meta"] = meta
    return output


def normalize_enum_value(value: Any, *, allowed: set[str], fallback: str) -> tuple[str, bool]:
    raw = str(value or "").strip()
    if raw in allowed:
        return raw, False
    parts = [item.strip(" '\"[]") for item in re.split(r"[,，]", raw) if item.strip(" '\"[]")]
    for item in parts:
        if item in allowed:
            return item, True
    aliases = {
        "comparison": "reasoning",
        "attitude": "relation",
        "dialogue": "fact",
        "definition": "fact",
        "event_detail": "fact",
        "plot_reasoning": "reasoning",
        "reveal": "plot_fact",
        "mystery": "plot_reasoning",
        "causality": "plot_reasoning",
    }
    mapped = aliases.get(raw)
    if mapped in allowed:
        return mapped, True
    return fallback, True


def normalize_positive_hypothesis(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if record.get("task_type") != "user_question_hypothesis_generation" or record.get("kto_tag") is not True:
        return record, False
    output = json.loads(json.dumps(record, ensure_ascii=False))
    conversations = output.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return output, False
    try:
        payload = json.loads(str(conversations[-1].get("value") or ""))
    except json.JSONDecodeError:
        return output, False
    if not isinstance(payload, dict):
        return output, False
    intent, changed_intent = normalize_enum_value(
        payload.get("intent"),
        allowed=ALLOWED_INTENTS,
        fallback="plot_reasoning",
    )
    query_type, changed_query_type = normalize_enum_value(
        payload.get("query_type"),
        allowed=ALLOWED_QUERY_TYPES,
        fallback="reasoning",
    )
    changed = changed_intent or changed_query_type
    if not changed:
        return output, False
    payload["intent"] = intent
    payload["query_type"] = query_type
    conversations[-1]["value"] = compact_json(payload)
    meta = output.get("meta") if isinstance(output.get("meta"), dict) else {}
    meta["qc_clean_reason"] = (
        str(meta.get("qc_clean_reason") or "") + ";normalize_positive_hypothesis_enum"
    ).strip(";")
    output["meta"] = meta
    return output, True


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    conflict_keys = exact_conflict_keys(records)
    cleaned: list[dict[str, Any]] = []
    for record in records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        reason = str(meta.get("api_verifier_reason") or "")
        if record.get("task_type") == "conclusion_generation" and not reason:
            stats["drop:unverified_conclusion"] += 1
            continue
        key = (str(record.get("task_type") or ""), prompt_text(record), response_text(record))
        if (
            key in conflict_keys
            and record.get("kto_tag") is True
            and reason == "verifier_chosen_retrieve_more"
        ):
            record = rebuild_retrieve_more(record)
            stats["patch:rebuild_conflicting_retrieve_more_chosen"] += 1
        record, normalized = normalize_positive_hypothesis(record)
        if normalized:
            stats["patch:normalize_positive_hypothesis_enum"] += 1
        cleaned.append(record)
    return cleaned, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC-clean SODA KTO dataset after API verifier relabeling.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}. Pass --overwrite.")
    train_in = read_json(args.input_dir / "train.json", [])
    val_in = read_json(args.input_dir / "val.json", [])
    if not isinstance(train_in, list) or not isinstance(val_in, list):
        raise SystemExit(f"Invalid dataset: {args.input_dir}")

    records, stats = clean_records(train_in + val_in)
    train, val = split_records(records, seed=args.seed, val_ratio=max(0.0, min(0.5, args.val_ratio)))
    dataset_name = args.dataset_name or args.output_dir.name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "val.json", val)
    write_json(args.output_dir / "dataset_info.json", dataset_info(dataset_name))

    remaining_prompt_keys = {prompt_key(record) for record in records}
    verifier_records = [
        record
        for record in read_jsonl(args.input_dir / "api_verifier_records.jsonl")
        if str(record.get("prompt_key") or "") in remaining_prompt_keys and not record.get("error")
    ]
    write_jsonl(args.output_dir / "api_verifier_records.jsonl", verifier_records)
    teacher_full_chain = read_jsonl(args.input_dir / "teacher_full_chain.jsonl")
    if teacher_full_chain:
        write_jsonl(args.output_dir / "teacher_full_chain.jsonl", teacher_full_chain)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "records_total": len(records),
        "records_train": len(train),
        "records_val": len(val),
        "verifier_records": len(verifier_records),
        "teacher_full_chain_records": len(teacher_full_chain),
        "stats": dict(stats),
        "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
        "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
        "api_verifier_reasons": dict(
            Counter(str((record.get("meta") or {}).get("api_verifier_reason") or "") for record in records)
        ),
    }
    write_json(args.output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
