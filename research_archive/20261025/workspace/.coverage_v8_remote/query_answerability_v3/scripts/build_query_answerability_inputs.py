#!/usr/bin/env python3
"""Build and audit query-only teacher inputs from an existing candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from query_answerability_teacher_protocol import clean_record, validate_input_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--expected-records", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    keys: set[str] = set()
    digest = hashlib.sha256()
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            cleaned = clean_record(raw)
            validate_input_record(cleaned, args.expected_candidates)
            key = f"{cleaned['source_name']}\n{cleaned['query']}"
            if key in keys:
                raise ValueError(f"duplicate source/query key at line {line_number}")
            keys.add(key)
            encoded = (json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            sink.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1
    if args.expected_records is not None and count != args.expected_records:
        raise ValueError(f"record count is {count}, expected {args.expected_records}")
    summary = {
        "source": str(input_path),
        "output": str(output_path),
        "records": count,
        "candidates_per_record": args.expected_candidates,
        "sha256": digest.hexdigest(),
        "teacher_visible_fields": ["query", "query_type", "candidates.*.candidate_id/kind/text"],
        "contains_reference_answer": False,
        "contains_gold_or_retrieval_scores": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
