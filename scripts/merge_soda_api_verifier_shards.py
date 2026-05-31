#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any


ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
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


def prompt_key(record: dict[str, Any], fallback: int) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    return str(meta.get("prompt_key") or record.get("id") or fallback)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge API-verifier relabeled SODA shard datasets.")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    input_dirs = [path.resolve() for path in args.input_dir]
    if output_dir.resolve() in input_dirs:
        raise SystemExit("Refusing to merge into an input directory. Use a separate --output-dir.")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Pass --overwrite.")

    records_by_id: dict[str, dict[str, Any]] = {}
    verifier_by_prompt: dict[str, dict[str, Any]] = {}
    teacher_full_chain_by_question: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    for input_dir in input_dirs:
        for split_name in ("train.json", "val.json"):
            for record in read_json(input_dir / split_name, []):
                record_id = str(record.get("id") or "")
                if record_id:
                    records_by_id.setdefault(record_id, record)
        for record in read_jsonl(input_dir / "api_verifier_records.jsonl"):
            key = str(record.get("prompt_key") or "")
            if key:
                verifier_by_prompt.setdefault(key, record)
        for record in read_jsonl(input_dir / "teacher_full_chain.jsonl"):
            question = str(record.get("question") or "")
            if question:
                teacher_full_chain_by_question.setdefault(question, record)
        summary = read_json(input_dir / "build_summary.json", None)
        if isinstance(summary, dict):
            summaries.append(summary)

    records = list(records_by_id.values())
    train, val = split_records(records, seed=args.seed, val_ratio=max(0.0, min(0.5, args.val_ratio)))
    dataset_name = args.dataset_name or output_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(dataset_name))
    write_jsonl(output_dir / "api_verifier_records.jsonl", list(verifier_by_prompt.values()))
    if teacher_full_chain_by_question:
        write_jsonl(output_dir / "teacher_full_chain.jsonl", list(teacher_full_chain_by_question.values()))

    summary = {
        "output_dir": str(output_dir),
        "input_dirs": [str(path) for path in input_dirs],
        "records_total": len(records),
        "records_train": len(train),
        "records_val": len(val),
        "verifier_records": len(verifier_by_prompt),
        "teacher_full_chain_records": len(teacher_full_chain_by_question),
        "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
        "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
        "api_verifier_reasons": dict(
            Counter(str((record.get("meta") or {}).get("api_verifier_reason") or "") for record in records)
        ),
        "input_summaries": summaries,
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
