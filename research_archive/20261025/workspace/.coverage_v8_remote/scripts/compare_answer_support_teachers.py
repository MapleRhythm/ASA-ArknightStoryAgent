#!/usr/bin/env python3
"""Measure agreement between two answer_support_v2 teacher outputs.

Claim IDs are local to each teacher response.  Cross-teacher comparisons align
claims by normalized claim text rather than assuming that both teachers chose
the same Q-number decomposition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answer_support_teacher_protocol import normalize_text, record_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIVE = {"direct", "partial"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_map(path: Path) -> dict[str, dict[str, Any]]:
    with resolve(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {record_key(record): record for record in rows}


def claim_text_map(record: dict[str, Any]) -> dict[str, str]:
    return {
        str(claim.get("claim_id") or ""): normalize_text(claim.get("text"))
        for claim in (record.get("teacher") or {}).get("claims") or []
    }


def labels(record: dict[str, Any]) -> dict[str, tuple[str, frozenset[str]]]:
    claim_texts = claim_text_map(record)
    return {
        str(item.get("candidate_id") or ""): (
            str(item.get("label") or ""),
            frozenset(
                claim_texts.get(str(value), f"__unknown__:{value}")
                for value in item.get("supported_claim_ids") or []
            ),
        )
        for item in (record.get("teacher") or {}).get("ranking") or []
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = read_map(args.left)
    right = read_map(args.right)
    common = sorted(set(left) & set(right))
    candidate_total = 0
    exact = 0
    polarity = 0
    left_positive = 0
    right_positive = 0
    positive_intersection = 0
    top3_claim_overlap_sum = 0.0
    samples: list[dict[str, Any]] = []
    for key in common:
        left_labels = labels(left[key])
        right_labels = labels(right[key])
        for candidate_id in set(left_labels) & set(right_labels):
            candidate_total += 1
            left_value = left_labels[candidate_id]
            right_value = right_labels[candidate_id]
            exact += left_value == right_value
            left_is_positive = left_value[0] in POSITIVE
            right_is_positive = right_value[0] in POSITIVE
            polarity += left_is_positive == right_is_positive
            left_positive += left_is_positive
            right_positive += right_is_positive
            positive_intersection += left_is_positive and right_is_positive
        left_top = set().union(
            *(claims for _, claims in list(left_labels.values())[:3])
        )
        right_top = set().union(
            *(claims for _, claims in list(right_labels.values())[:3])
        )
        union = left_top | right_top
        top3_claim_overlap_sum += len(left_top & right_top) / max(len(union), 1)
        if len(samples) < 20 and left_labels != right_labels:
            samples.append({"query": left[key].get("query"), "key": key})
    precision = positive_intersection / max(left_positive, 1)
    recall = positive_intersection / max(right_positive, 1)
    summary = {
        "left_records": len(left),
        "right_records": len(right),
        "common_records": len(common),
        "candidate_judgements": candidate_total,
        "exact_label_claim_agreement": exact / max(candidate_total, 1),
        "positive_negative_agreement": polarity / max(candidate_total, 1),
        "left_positive_precision_against_right": precision,
        "left_positive_recall_against_right": recall,
        "positive_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_top3_claim_jaccard": top3_claim_overlap_sum / max(len(common), 1),
        "disagreement_samples": samples,
    }
    if args.report:
        report = resolve(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
