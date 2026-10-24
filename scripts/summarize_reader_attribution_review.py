#!/usr/bin/env python3
"""Join an explicit source review to hash-pinned reader ablation predictions.

This does not judge semantics automatically. A model-authored review remains a
model review, not human gold. Protocol failures are counted separately from
semantic outcomes, and every matrix row remains in the denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_exx_outputs as exx_eval


OUTCOMES = {"complete", "partial", "incorrect", "unanswered", "uncertain"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    matrix: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    review: dict[str, Any],
) -> dict[str, Any]:
    gold = {row["id"]: row for row in matrix}
    if not gold or len(gold) != len(matrix):
        raise ValueError("empty or duplicate matrix IDs")
    conditions = review["condition_order"]
    if len(set(conditions)) != len(conditions):
        raise ValueError("duplicate conditions")
    labels = {}
    for row in review["reviews"]:
        if len(row["outcomes"]) != len(conditions) or len(row["reasons"]) != len(conditions):
            raise ValueError("review length does not match conditions")
        for condition, outcome, reason in zip(conditions, row["outcomes"], row["reasons"]):
            key = (row["reader"], row["source_id"], condition)
            if key in labels or outcome not in OUTCOMES or not str(reason).strip():
                raise ValueError("duplicate or invalid review label")
            labels[key] = (outcome, reason)
    seen_labels = set()
    details = []
    groups = defaultdict(Counter)
    for reader, rows in predictions.items():
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(gold):
            raise ValueError("prediction IDs must cover matrix exactly once")
        for row_id, expected in gold.items():
            row = by_id[row_id]
            if (
                row.get("system") != expected.get("system")
                or row["conversations"][0] != expected["conversations"][0]
                or row.get("diagnostic") != expected.get("diagnostic")
            ):
                raise ValueError("prediction prompt or diagnostic differs from matrix")
            diagnostic = expected["diagnostic"]
            key = (reader, diagnostic["source_id"], diagnostic["condition"])
            if key not in labels or key in seen_labels:
                raise ValueError("missing or multiply consumed review")
            seen_labels.add(key)
            outcome, reason = labels[key]
            payload, strict = exx_eval.payload_from_row(row)
            problems = exx_eval.validate_payload(
                payload, set(exx_eval.EVIDENCE_RE.findall(exx_eval.prompt_value(row)))
            )
            action = str((payload or {}).get("next_action") or "invalid")
            if (outcome == "unanswered") != (action in {"retrieve_more", "abstain"}):
                raise ValueError("review outcome conflicts with generated action")
            truncated = exx_eval.generation_was_truncated(row)
            success = outcome == "complete" and not problems and not truncated
            group = groups[(reader, diagnostic["condition"])]
            group.update({
                "rows": 1,
                outcome: 1,
                "primary_success": int(success),
                "schema_valid": int(not problems),
                "strict_json": int(strict),
                "generation_truncated": int(truncated),
                "over_retrieve": int(action == "retrieve_more"),
                "over_abstain": int(action == "abstain"),
                "prompt_tokens": int(row.get("prompt_token_count") or 0),
                "generation_tokens": int(row.get("generated_token_count") or 0),
            })
            details.append({
                "reader": reader, "id": row_id, "source_id": diagnostic["source_id"],
                "condition": diagnostic["condition"], "outcome": outcome,
                "reason": reason, "primary_success": success,
                "schema_problems": problems,
                "raw_output_sha256": hashlib.sha256(
                    str(row["raw_output"]).encode("utf-8")
                ).hexdigest(),
            })
    if seen_labels != set(labels):
        raise ValueError("unused review labels")
    results = {}
    for (reader, condition), counts in sorted(groups.items()):
        results.setdefault(reader, {})[condition] = {
            **dict(counts),
            "primary_success_rate": counts["primary_success"] / counts["rows"],
            "mean_prompt_tokens": counts["prompt_tokens"] / counts["rows"],
            "mean_generation_tokens": counts["generation_tokens"] / counts["rows"],
        }
    return {
        "protocol": "asa_reader_attribution_review_summary_v1",
        "reviewer": review["reviewer"],
        "independent_human_review": review["independent_human_review"],
        "evaluation_only": True,
        "criteria": review["criteria"],
        "limitations": review["limitations"],
        "reviewed_predictions": len(details),
        "results": results,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="READER=/path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite summary")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if sha256(args.matrix) != review["matrix_sha256"]:
        parser.error("matrix hash differs from reviewed artifact")
    predictions, paths = {}, {}
    for spec in args.prediction:
        name, path_str = spec.split("=", 1)
        path = Path(path_str)
        if name in predictions or sha256(path) != review["prediction_sha256"].get(name):
            parser.error("duplicate reader or prediction hash mismatch")
        predictions[name] = json.loads(path.read_text(encoding="utf-8"))
        paths[name] = {"path": str(path), "sha256": sha256(path)}
    if set(predictions) != set(review["prediction_sha256"]):
        parser.error("must supply exactly the reviewed readers")
    report = summarize(
        json.loads(args.matrix.read_text(encoding="utf-8")), predictions, review
    )
    report["inputs"] = {
        "matrix": {"path": str(args.matrix), "sha256": sha256(args.matrix)},
        "review": {"path": str(args.review), "sha256": sha256(args.review)},
        "predictions": paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["results"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
