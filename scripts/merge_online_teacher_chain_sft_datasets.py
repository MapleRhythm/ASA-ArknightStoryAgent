#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KALTSIT_INTERNAL_ALIAS = "凯尔希·思衡托"
KALTSIT_NATURAL_NAME = "凯尔希"
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
    "MiniRAG",
    "minirag",
)
FALLBACK_ANSWER_MARKERS = (
    "已检索到的证据能确认：",
    "但原答案中的",
    "grounding 校验",
)
ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


def load_records(dataset_dir: Path) -> list[dict[str, Any]]:
    records_jsonl = dataset_dir / "records.jsonl"
    if records_jsonl.exists():
        return [
            json.loads(line)
            for line in records_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.json"
        if path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"No records found in {dataset_dir}")
    return records


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


def normalize_alias_value(value: str, *, field: str) -> str:
    item = str(value or "").strip()
    if not item:
        return ""
    if KALTSIT_INTERNAL_ALIAS in item:
        if field == "keywords":
            return KALTSIT_NATURAL_NAME
        if item == KALTSIT_INTERNAL_ALIAS:
            return KALTSIT_NATURAL_NAME
        return item.replace(KALTSIT_INTERNAL_ALIAS, KALTSIT_NATURAL_NAME)
    return item


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


def normalize_field_items(value: str, *, field: str) -> list[str]:
    item = normalize_alias_value(value, field=field)
    if not item:
        return []
    if field == "keywords" and re.search(r"\s+", item):
        parts = [
            normalize_alias_value(part, field=field)
            for part in re.split(r"\s+", item)
            if part.strip()
        ]
        return [
            part
            for part in parts
            if part and len(part) >= 2 and not is_bad_retrieval_term(part, field=field)
        ]
    return [item]


def is_grounding_fallback_answer(payload: dict[str, Any]) -> bool:
    if str(payload.get("next_action") or "") != "answer_directly":
        return False
    answer = str(payload.get("answer") or "")
    return any(marker in answer for marker in FALLBACK_ANSWER_MARKERS)


def sanitize_json_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [sanitize_json_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"entities", "keywords"} and isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, (str, int, float)):
                    normalized.extend(normalize_field_items(str(item), field=key))
            output[key] = [
                item
                for item in dedupe_keep_order(normalized)
                if item and not is_bad_retrieval_term(item, field=key)
            ]
            continue
        output[key] = sanitize_json_payload(value)
    if is_grounding_fallback_answer(output):
        output["next_action"] = "abstain"
        output["follow_up_hypothesis"] = None
        output["clarification_question"] = ""
    return output


def sanitize_message_value(value: str) -> str:
    # Keep the alias system in source/index files. The training copy uses natural
    # terms so the student does not learn to emit internal alias strings.
    lines = []
    for line in str(value or "").splitlines():
        if line.startswith("minirag_hints:"):
            continue
        lines.append(line)
    return "\n".join(lines).replace(KALTSIT_INTERNAL_ALIAS, KALTSIT_NATURAL_NAME)


def clean_record(record: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    payload = dict(record)
    conversations = []
    for message in payload.get("conversations") or []:
        item = dict(message)
        value = str(item.get("value") or "")
        if item.get("from") == "gpt":
            try:
                parsed = json.loads(value)
                value = json.dumps(sanitize_json_payload(parsed), ensure_ascii=False, separators=(",", ":"))
            except json.JSONDecodeError:
                value = sanitize_message_value(value)
        else:
            value = sanitize_message_value(value)
        item["value"] = value
        conversations.append(item)
    payload["conversations"] = conversations
    meta = dict(payload.get("meta") or {})
    sources = meta.get("merge_sources")
    if not isinstance(sources, list):
        sources = []
    if source_label not in sources:
        sources.append(source_label)
    meta["merge_sources"] = sources
    payload["meta"] = meta
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
        f"{dataset_name}_test": entry("test.json"),
    }


def question_key(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    return str(meta.get("source_question_key") or record.get("id") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and clean online teacher-chain SFT datasets.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [("base", args.base)] + [
        (f"supplement_{index}", path)
        for index, path in enumerate(args.supplement, start=1)
    ]
    merged: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    skipped_duplicates = 0
    for label, dataset_dir in sources:
        count = 0
        for record in load_records(dataset_dir.resolve()):
            record_id = str(record.get("id") or "")
            if not record_id:
                continue
            if record_id in merged:
                skipped_duplicates += 1
                continue
            merged[record_id] = clean_record(record, source_label=label)
            count += 1
        source_counts[label] = count

    records = list(merged.values())
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[question_key(record)].append(record)

    keys = list(by_question)
    random.Random(args.seed).shuffle(keys)
    train_cut = int(len(keys) * args.train_ratio)
    val_cut = train_cut + int(len(keys) * args.val_ratio)
    split_keys = {
        "train": set(keys[:train_cut]),
        "val": set(keys[train_cut:val_cut]),
        "test": set(keys[val_cut:]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    summary: dict[str, Any] = {
        "records": len(records),
        "questions": len(keys),
        "source_counts": source_counts,
        "skipped_duplicate_records": skipped_duplicates,
        "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
        "splits": {},
        "alias_cleaning": {
            "training_copy_replaced": {KALTSIT_INTERNAL_ALIAS: KALTSIT_NATURAL_NAME},
            "source_alias_files_modified": False,
        },
    }
    for split, selected in split_keys.items():
        split_records = [record for key in keys if key in selected for record in by_question[key]]
        write_json(args.output_dir / f"{split}.json", split_records)
        summary["splits"][split] = {
            "questions": len(selected),
            "records": len(split_records),
            "task_counts": dict(Counter(str(record.get("task_type") or "") for record in split_records)),
        }

    dataset_name = args.output_dir.name
    write_json(args.output_dir / "dataset_info.json", dataset_info(dataset_name))
    write_json(args.output_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
