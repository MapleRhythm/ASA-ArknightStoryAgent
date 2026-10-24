#!/usr/bin/env python3
"""Audit gold-free ASA teacher JSONL outputs before materialization/training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_support_teacher_protocol import (
    PROTOCOL_VERSION,
    leaked_supervision_fields,
    normalize_text,
    record_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    leaks = leaked_supervision_fields(record)
    if leaks:
        errors.append(f"supervision_leak={leaks[:8]}")
    if str(record.get("teacher_protocol") or "") != PROTOCOL_VERSION:
        errors.append(f"protocol={record.get('teacher_protocol')}")
    candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in record.get("candidates") or []
    }
    claims = list((record.get("teacher") or {}).get("claims") or [])
    claim_ids = {str(claim.get("claim_id") or "") for claim in claims}
    ranking = list((record.get("teacher") or {}).get("ranking") or [])
    ranking_ids = [str(item.get("candidate_id") or "") for item in ranking]
    if len(ranking_ids) != len(candidates) or set(ranking_ids) != set(candidates):
        errors.append("ranking_candidate_set_mismatch")
    if len(set(ranking_ids)) != len(ranking_ids):
        errors.append("ranking_duplicate_ids")
    covered: set[str] = set()
    for item in ranking:
        candidate_id = str(item.get("candidate_id") or "")
        label = str(item.get("label") or "")
        supported = set(str(value) for value in item.get("supported_claim_ids") or [])
        quote = str(item.get("support_quote") or "")
        if not supported <= claim_ids:
            errors.append(f"unknown_claim_id={candidate_id}")
        if label in {"direct", "partial", "contradiction"}:
            text = str((candidates.get(candidate_id) or {}).get("text") or "")
            if len(normalize_text(quote)) < 4 or normalize_text(quote) not in normalize_text(text):
                errors.append(f"bad_quote={candidate_id}")
        if label in {"direct", "partial"}:
            covered.update(supported)
    uncovered = set(
        str(value) for value in (record.get("teacher") or {}).get("uncovered_claim_ids") or []
    )
    if uncovered != claim_ids - covered:
        errors.append("uncovered_claim_ids_inconsistent")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-total", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    duplicate_keys: list[str] = []
    seen: set[str] = set()
    for raw_path in args.input:
        for record in read_jsonl(resolve(raw_path)):
            key = record_key(record)
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            rows.append(record)

    invalid: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    claims = 0
    uncovered = 0
    for index, record in enumerate(rows, start=1):
        errors = audit_record(record)
        if errors:
            invalid.append(
                {
                    "line": index,
                    "query": str(record.get("query") or "")[:160],
                    "errors": errors,
                }
            )
        teacher = record.get("teacher") or {}
        claims += len(teacher.get("claims") or [])
        uncovered += len(teacher.get("uncovered_claim_ids") or [])
        labels.update(str(item.get("label") or "") for item in teacher.get("ranking") or [])
    summary = {
        "records": len(rows),
        "expected_total": args.expected_total,
        "complete": args.expected_total is None or len(seen) == args.expected_total,
        "unique_records": len(seen),
        "duplicate_records": len(duplicate_keys),
        "invalid_records": len(invalid),
        "claims": claims,
        "uncovered_claims": uncovered,
        "labels": dict(labels),
        "invalid_samples": invalid[:50],
    }
    if args.report:
        report_path = resolve(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(bool(invalid or duplicate_keys or (args.expected_total and len(seen) != args.expected_total)))


if __name__ == "__main__":
    raise SystemExit(main())
