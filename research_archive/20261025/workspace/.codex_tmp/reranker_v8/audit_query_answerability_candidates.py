#!/usr/bin/env python3
"""Audit frozen query-only candidate pools before external labelling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


RECORD_FIELDS = {"query", "query_type", "source_name", "story_family", "candidates"}
CANDIDATE_FIELDS = {"candidate_id", "text", "kind"}
FORBIDDEN_KEYS = {
    "answer",
    "reference_answer",
    "claims",
    "gold_evidence",
    "gold_evidence_ids",
    "coverable_evidence_ids",
    "required_evidence",
    "answer_evidence",
    "answer_focus",
    "retrieval_rank",
    "original_rank",
    "doc_id",
    "story_id",
    "stage_code",
    "rerank_score",
    "fusion_score",
    "score",
    "covered_evidence_ids",
    "relevance",
    "label",
}


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def fingerprint(candidates: list[dict[str, Any]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit(path: Path, *, expected_records: int, expected_candidates: int) -> dict[str, Any]:
    errors: list[str] = []
    keys: set[tuple[str, str]] = set()
    pool_hashes: set[str] = set()
    count = 0
    total_chars = 0
    for line_number, record in iter_jsonl(path):
        count += 1
        location = f"{path}:{line_number}"
        if set(record) != RECORD_FIELDS:
            errors.append(location + f" fields={sorted(record)}")
        leaked = sorted(set(record) & FORBIDDEN_KEYS)
        if leaked:
            errors.append(location + f" forbidden_record_keys={leaked}")
        key = (str(record.get("source_name") or ""), str(record.get("query") or ""))
        if not all(key) or key in keys:
            errors.append(location + " empty_or_duplicate_query_key")
        keys.add(key)
        if not str(record.get("story_family") or "").strip():
            errors.append(location + " empty_story_family")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_candidates:
            errors.append(location + f" candidate_count={len(candidates or [])}")
            continue
        candidate_ids: list[str] = []
        candidate_texts: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
                errors.append(location + f" candidate_{index}_fields")
                continue
            leaked = sorted(set(candidate) & FORBIDDEN_KEYS)
            if leaked:
                errors.append(location + f" candidate_{index}_forbidden_keys={leaked}")
            candidate_ids.append(str(candidate.get("candidate_id") or ""))
            text = str(candidate.get("text") or "")
            candidate_texts.append(text)
            total_chars += len(text)
            if str(candidate.get("kind") or "") != "document" or not text.strip():
                errors.append(location + f" candidate_{index}_invalid_content")
        if len(candidate_ids) != len(set(candidate_ids)) or not all(candidate_ids):
            errors.append(location + " duplicate_or_empty_candidate_ids")
        if len(candidate_texts) != len(set(candidate_texts)):
            errors.append(location + " duplicate_candidate_texts")
        pool_hashes.add(fingerprint(candidates))
    if count != expected_records:
        errors.append(f"record_count={count}, expected={expected_records}")
    return {
        "path": str(path),
        "records": count,
        "expected_records": expected_records,
        "candidates_per_record": expected_candidates,
        "unique_query_keys": len(keys),
        "unique_candidate_pools": len(pool_hashes),
        "mean_candidate_chars": round(
            total_chars / max(count * expected_candidates, 1), 3
        ),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(
        args.input,
        expected_records=args.expected_records,
        expected_candidates=args.expected_candidates,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["error_count"]:
        raise RuntimeError("candidate audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
