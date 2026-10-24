#!/usr/bin/env python3
"""Convert gold-free teacher judgements into compact reranker train groups.

Only verified direct/partial positives, selected hard-context/contradiction
negatives, and high-ranked retrieval negatives are retained.  Positive weights
are balanced by answer claim so repeated passages for an easy claim cannot
dominate a multi-claim query.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_support_teacher_protocol import PROTOCOL_VERSION, record_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIVE_BASE_WEIGHTS = {"direct": 1.0, "partial": 0.5}
NEGATIVE_LABELS = {"hard_context", "contradiction", "irrelevant"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def materialize_record(
    record: dict[str, Any],
    *,
    max_group_size: int,
    max_positives_per_claim: int,
    max_hard_context: int,
) -> dict[str, Any] | None:
    if str(record.get("teacher_protocol") or "") != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported teacher protocol for {record.get('query')}: "
            f"{record.get('teacher_protocol')}"
        )
    candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in record.get("candidates") or []
    }
    ranking = list((record.get("teacher") or {}).get("ranking") or [])
    claims = list((record.get("teacher") or {}).get("claims") or [])
    claim_ids = [str(claim.get("claim_id") or "") for claim in claims]

    positives_by_claim: dict[str, list[dict[str, Any]]] = {claim_id: [] for claim_id in claim_ids}
    for item in ranking:
        if item.get("label") not in POSITIVE_BASE_WEIGHTS:
            continue
        for claim_id in item.get("supported_claim_ids") or []:
            if claim_id in positives_by_claim:
                positives_by_claim[claim_id].append(item)

    selected_positive_ids: set[str] = set()
    for claim_id in claim_ids:
        ordered = sorted(
            positives_by_claim[claim_id],
            key=lambda item: (
                item.get("label") != "direct",
                int(item.get("rank") or 10**9),
            ),
        )
        selected_positive_ids.update(
            str(item.get("candidate_id") or "")
            for item in ordered[:max_positives_per_claim]
        )
    if not selected_positive_ids:
        return None

    selected_positive_items = [
        item for item in ranking if str(item.get("candidate_id") or "") in selected_positive_ids
    ]
    positive_budget = max(1, max_group_size - 1)  # reserve a negative
    if len(selected_positive_items) > positive_budget:
        # First retain one positive for as many distinct claims as possible,
        # then use remaining slots for direct/multi-claim examples.
        by_claim: dict[str, list[dict[str, Any]]] = {claim_id: [] for claim_id in claim_ids}
        for item in selected_positive_items:
            for claim_id in item.get("supported_claim_ids") or []:
                if claim_id in by_claim:
                    by_claim[claim_id].append(item)
        compact_positive: list[dict[str, Any]] = []
        compact_ids: set[str] = set()
        for claim_id in claim_ids:
            choices = sorted(
                by_claim[claim_id],
                key=lambda item: (
                    item.get("label") != "direct",
                    int(item.get("rank") or 10**9),
                ),
            )
            if choices and str(choices[0].get("candidate_id") or "") not in compact_ids:
                compact_positive.append(choices[0])
                compact_ids.add(str(choices[0].get("candidate_id") or ""))
                if len(compact_positive) >= positive_budget:
                    break
        if len(compact_positive) < positive_budget:
            for item in sorted(
                selected_positive_items,
                key=lambda value: (
                    value.get("label") != "direct",
                    -len(value.get("supported_claim_ids") or []),
                    int(value.get("rank") or 10**9),
                ),
            ):
                candidate_id = str(item.get("candidate_id") or "")
                if candidate_id in compact_ids:
                    continue
                compact_positive.append(item)
                compact_ids.add(candidate_id)
                if len(compact_positive) >= positive_budget:
                    break
        selected_positive_items = compact_positive
    frequency = Counter(
        claim_id
        for item in selected_positive_items
        for claim_id in item.get("supported_claim_ids") or []
        if claim_id in positives_by_claim
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for item in selected_positive_items:
        candidate_id = str(item.get("candidate_id") or "")
        candidate = candidates[candidate_id]
        base = POSITIVE_BASE_WEIGHTS[str(item.get("label"))]
        relevance = sum(
            base / max(frequency[claim_id], 1)
            for claim_id in item.get("supported_claim_ids") or []
            if claim_id in frequency
        )
        selected.append(
            {
                "candidate_id": candidate_id,
                "text": str(candidate.get("text") or ""),
                "relevance": round(relevance, 6),
                "teacher_label": item.get("label"),
                "supported_claim_ids": list(item.get("supported_claim_ids") or []),
                "support_quote": str(item.get("support_quote") or ""),
                "retrieval_rank": candidate.get("retrieval_rank"),
                "doc_id": candidate.get("doc_id"),
            }
        )
        selected_ids.add(candidate_id)

    hard_context_added = 0
    for item in ranking:
        candidate_id = str(item.get("candidate_id") or "")
        label = str(item.get("label") or "irrelevant")
        if candidate_id in selected_ids or label not in NEGATIVE_LABELS:
            continue
        if label == "hard_context":
            if hard_context_added >= max_hard_context:
                continue
            hard_context_added += 1
        candidate = candidates[candidate_id]
        selected.append(
            {
                "candidate_id": candidate_id,
                "text": str(candidate.get("text") or ""),
                "relevance": 0.0,
                "teacher_label": label,
                "supported_claim_ids": [],
                "support_quote": str(item.get("support_quote") or ""),
                "retrieval_rank": candidate.get("retrieval_rank"),
                "doc_id": candidate.get("doc_id"),
            }
        )
        selected_ids.add(candidate_id)
        if len(selected) >= max_group_size:
            break

    if not any(candidate["relevance"] == 0 for candidate in selected):
        return None
    return {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "source_name": str(record.get("source_name") or ""),
        "story_family": str(record.get("story_family") or ""),
        "answer": str(record.get("answer") or ""),
        "answer_claims": claims,
        "uncovered_claim_ids": list(
            (record.get("teacher") or {}).get("uncovered_claim_ids") or []
        ),
        "teacher_protocol": PROTOCOL_VERSION,
        "teacher_model": record.get("teacher_model"),
        "candidates": selected[:max_group_size],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-group-size", type=int, default=24)
    parser.add_argument("--max-positives-per-claim", type=int, default=2)
    parser.add_argument("--max-hard-context", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in args.input:
        for record in read_jsonl(resolve(raw_path)):
            key = record_key(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)

    output_records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for record in records:
        materialized = materialize_record(
            record,
            max_group_size=args.max_group_size,
            max_positives_per_claim=args.max_positives_per_claim,
            max_hard_context=args.max_hard_context,
        )
        if materialized is None:
            stats["skipped_non_trainable"] += 1
            continue
        output_records.append(materialized)
        stats["groups"] += 1
        stats["candidates"] += len(materialized["candidates"])
        stats["positives"] += sum(
            candidate["relevance"] > 0 for candidate in materialized["candidates"]
        )
        stats["negatives"] += sum(
            candidate["relevance"] == 0 for candidate in materialized["candidates"]
        )

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_records": len(records),
        "output_records": len(output_records),
        "stats": dict(stats),
        "max_group_size": args.max_group_size,
        "max_positives_per_claim": args.max_positives_per_claim,
        "max_hard_context": args.max_hard_context,
        "protocol": PROTOCOL_VERSION,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
