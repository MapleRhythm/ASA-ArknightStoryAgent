#!/usr/bin/env python3
"""Remove gold/legacy supervision from reranker teacher input JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answer_support_teacher_protocol import (
    clean_record,
    leaked_supervision_fields,
    validate_input_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--allow-non120", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    expected = None if args.allow_non120 else args.expected_candidates
    records = read_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    cleaned_records: list[dict[str, Any]] = []
    removed_fields = 0
    for line_number, record in enumerate(records, start=1):
        removed_fields += len(leaked_supervision_fields(record))
        cleaned = clean_record(record)
        validate_input_record(cleaned, expected)
        leaks = leaked_supervision_fields(cleaned)
        if leaks:
            raise ValueError(f"line {line_number} still leaks supervision: {leaks[:8]}")
        cleaned_records.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in cleaned_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(cleaned_records),
        "expected_candidates": expected,
        "removed_supervision_fields": removed_fields,
        "record_fields": sorted(cleaned_records[0]) if cleaned_records else [],
        "candidate_fields": (
            sorted(cleaned_records[0]["candidates"][0]) if cleaned_records else []
        ),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
