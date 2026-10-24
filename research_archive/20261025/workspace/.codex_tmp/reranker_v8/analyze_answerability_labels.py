#!/usr/bin/env python3
"""Audit answerability labels and quantify conservative training thresholds."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


POSITIVE = {"direct_answer", "partial_answer", "supporting_context"}
VALID = POSITIVE | {"hard_context", "irrelevant"}
CAPS = {
    "direct_answer": 12,
    "partial_answer": 20,
    "supporting_context": 16,
    "hard_context": 16,
}


def norm(value: Any) -> str:
    return "".join(str(value or "").split())


def fingerprint(candidates: list[dict[str, Any]]) -> str:
    stable = sorted(
        (
            str(x.get("candidate_id") or ""),
            str(x.get("kind") or ""),
            str(x.get("text") or ""),
        )
        for x in candidates
    )
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def audit(path: Path) -> dict[str, Any]:
    rows = 0
    keys: set[tuple[str, str]] = set()
    families: collections.Counter[str] = collections.Counter()
    labels: collections.Counter[str] = collections.Counter()
    decisions: collections.Counter[str] = collections.Counter()
    positive_hist: collections.Counter[int] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    quote_total = quote_verbatim = quote_bad = 0
    caps_exceeded: collections.Counter[str] = collections.Counter()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            continue
        key = (str(row.get("source_name") or ""), str(row.get("query") or ""))
        if key in keys:
            errors["duplicate_query_key"] += 1
        keys.add(key)
        families[str(row.get("story_family") or "")] += 1
        teacher = row.get("teacher") or {}
        ranking = teacher.get("ranking") or []
        candidates = {
            str(x.get("candidate_id") or ""): str(x.get("text") or "")
            for x in row.get("candidates") or []
            if isinstance(x, dict)
        }
        current = collections.Counter()
        seen: set[str] = set()
        tiers: list[int] = []
        for item in ranking:
            if not isinstance(item, dict):
                errors["ranking_non_object"] += 1
                continue
            cid = str(item.get("candidate_id") or "")
            label = str(item.get("label") or "")
            labels[label] += 1
            current[label] += 1
            if label not in VALID:
                errors["invalid_label"] += 1
            if cid in seen:
                errors["duplicate_ranked_id"] += 1
            seen.add(cid)
            tier = {
                "direct_answer": 0,
                "partial_answer": 1,
                "supporting_context": 2,
                "hard_context": 3,
                "irrelevant": 4,
            }.get(label)
            if tier is not None:
                tiers.append(tier)
            quote = str(item.get("support_quote") or "")
            if label in POSITIVE:
                quote_total += 1
                if 4 <= len(norm(quote)) <= 120 and norm(quote) in norm(
                    candidates.get(cid, "")
                ):
                    quote_verbatim += 1
                else:
                    quote_bad += 1
            elif quote.strip():
                errors["negative_has_quote"] += 1
        if tiers != sorted(tiers):
            errors["tier_order_violation"] += 1
        for label, cap in CAPS.items():
            if current[label] > cap:
                caps_exceeded[label] += 1
        positive_hist[sum(current[label] for label in POSITIVE)] += 1
        status = str((teacher.get("query_decision") or {}).get("status") or "answerable")
        decisions[status] += 1
        if status not in {"answerable", "ambiguous"}:
            errors["invalid_query_decision"] += 1
        if status == "ambiguous" and sum(current[label] for label in POSITIVE):
            errors["ambiguous_has_positive"] += 1
    threshold = {}
    for cap in (5, 6, 8, 10, 12, 16, 20):
        kept = sum(n for count, n in positive_hist.items() if 0 < count <= cap)
        threshold[str(cap)] = {
            "query_count": kept,
            "query_fraction": (kept / rows) if rows else 0.0,
        }
    return {
        "path": str(path),
        "rows": rows,
        "unique_query_keys": len(keys),
        "story_families": dict(sorted(families.items())),
        "decisions": dict(sorted(decisions.items())),
        "label_counts": dict(sorted(labels.items())),
        "positive_count_histogram": {
            str(k): v for k, v in sorted(positive_hist.items())
        },
        "threshold_sensitivity": threshold,
        "positive_quote_total": quote_total,
        "positive_quote_verbatim": quote_verbatim,
        "positive_quote_bad": quote_bad,
        "label_caps_exceeded": dict(sorted(caps_exceeded.items())),
        "errors": dict(sorted(errors.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"inputs": [audit(path) for path in args.input]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
