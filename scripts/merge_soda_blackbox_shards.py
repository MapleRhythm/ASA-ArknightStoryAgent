#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        prompt_key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        by_prompt.setdefault(prompt_key, []).append(record)
    keys = list(by_prompt)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = max(1, int(round(len(records) * val_ratio))) if len(records) > 10 and val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for key in keys:
        target = val if val_count < target_val else train
        target.extend(by_prompt[key])
        if target is val:
            val_count += len(by_prompt[key])
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
    parser = argparse.ArgumentParser(description="Merge SODA black-box distillation shards into one LLaMA Factory KTO dataset.")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260529)
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
        raise SystemExit(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace generated files.")

    audit_by_id: dict[str, dict[str, Any]] = {}
    raw_by_prompt: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for input_dir in input_dirs:
        for record in read_jsonl(input_dir / "audit_records.jsonl"):
            record_id = str(record.get("id") or "")
            if record_id:
                audit_by_id.setdefault(record_id, record)
        for record in read_jsonl(input_dir / "raw_pairs.jsonl"):
            prompt_key = str(record.get("prompt_key") or "")
            if prompt_key:
                raw_by_prompt.setdefault(prompt_key, record)
        for record in read_jsonl(input_dir / "failed.jsonl"):
            failures.append(record)
        summary_path = input_dir / "build_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    records = list(audit_by_id.values())
    train, val = split_records(records, seed=args.seed, val_ratio=max(0.0, min(0.5, args.val_ratio)))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "audit_records.jsonl", records)
    write_jsonl(output_dir / "raw_pairs.jsonl", list(raw_by_prompt.values()))
    if failures:
        write_jsonl(output_dir / "failed.jsonl", failures)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(output_dir.name))
    write_json(
        output_dir / "build_summary.json",
        {
            "output_dir": str(output_dir),
            "input_dirs": [str(path) for path in input_dirs],
            "records_total": len(records),
            "records_train": len(train),
            "records_val": len(val),
            "raw_pairs": len(raw_by_prompt),
            "failures": len(failures),
            "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
            "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
            "input_summaries": summaries,
        },
    )
    print(f"[merged] records={len(records)} raw_pairs={len(raw_by_prompt)} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
