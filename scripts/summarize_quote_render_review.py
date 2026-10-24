#!/usr/bin/env python3
"""Summarize an explicit semantic review bound to the exact mechanical audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUSES = {"complete_supported", "partial_supported", "unsupported",
            "not_answered", "unparseable", "uncertain"}


def summarize(audit: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in review["rows"]}
    if (len(by_id) != len(review["rows"])
            or len({row["id"] for row in audit["rows"]}) != len(audit["rows"])
            or by_id.keys() != {row["id"] for row in audit["rows"]}):
        raise ValueError("require exact, unique review/audit ID sets")
    for key in ("matrix_sha256", "predictions_sha256"):
        if review[key] != audit[key]:
            raise ValueError(f"{key} mismatch")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit["rows"]:
        judgment = by_id[row["id"]]
        if any(judgment.get(field) not in STATUSES for field in ("raw_content", "validated_content")):
            raise ValueError("unknown review status")
        answered = row.get("validated", {}).get("next_action") == "answer_directly"
        if answered == (judgment["validated_content"] == "not_answered"):
            raise ValueError("review disagrees with validated action")
        if (judgment["validated_content"] == "complete_supported"
                and row["finish_reason"] != "stop"):
            raise ValueError("truncated output cannot count as primary success")
        groups[f'{row["template"]}/{row["renderer"]}'].append(judgment)
    return {
        name: {
            "questions": len(rows),
            "raw_content": dict(Counter(row["raw_content"] for row in rows)),
            "validated_content": dict(Counter(row["validated_content"] for row in rows)),
            "strict_complete_supported": sum(row["validated_content"] == "complete_supported" for row in rows),
            "upper_bound_if_uncertain_accepted": sum(
                row["validated_content"] in {"complete_supported", "uncertain"} for row in rows
            ),
        } for name, rows in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite summary")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.audit.read_bytes()).hexdigest()
    if digest != review["mechanical_audit_sha256"]:
        parser.error("mechanical audit hash mismatch")
    result = {
        "version": "asa_quote_render_review_summary_v1",
        "evaluation_only": True, "not_production_accuracy": True,
        "reviewer": review["reviewer"],
        "independent_human_review": review["independent_human_review"],
        "mechanical_audit_sha256": digest,
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "summary": summarize(audit, review),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
