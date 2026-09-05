#!/usr/bin/env python3
"""Evaluate Exx predictions against recalibrated evidence bindings.

The legacy evaluator compares predicted evidence IDs with an immutable teacher
answer.  This evaluator applies the GLM strict recalibration changes first,
optionally applies the ambiguity adjudication sidecar, and reports binding
metrics separately from action metrics.  It never edits the source gold or
prediction files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def payload_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    direct = row.get("payload")
    if isinstance(direct, dict):
        return direct
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        raw = conversations[-1].get("value")
    else:
        raw = row.get("raw_output") or row.get("output")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def fact_ids(payload: dict[str, Any] | None) -> set[str]:
    if not payload or payload.get("next_action") != "answer_directly":
        return set()
    return {
        str(item)
        for fact in payload.get("supported_facts") or []
        if isinstance(fact, dict)
        for item in fact.get("evidence_ids") or []
    }


def fact_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload or payload.get("next_action") != "answer_directly":
        return []
    result = []
    for fact in payload.get("supported_facts") or []:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("fact") or "").strip()
        if text:
            result.append(
                {
                    "fact": text,
                    "evidence_ids": {str(item) for item in fact.get("evidence_ids") or []},
                }
            )
    return result


def row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("task_id") or index)


def normalized_text(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value.lower())


def text_similarity(left: str, right: str) -> float:
    left_chars = set(normalized_text(left))
    right_chars = set(normalized_text(right))
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)


def apply_calibration(
    original: dict[str, Any] | None,
    changes: list[dict[str, Any]],
    ambiguity: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, set[int], list[str]]:
    """Return calibrated payload, ignored fact indexes, and applied labels."""
    if original is None:
        return None, set(), []
    payload = json.loads(json.dumps(original))
    facts = list(payload.get("supported_facts") or [])
    by_index = {
        int(row["fact_index"]): row
        for row in changes
        if row.get("fact_index") is not None
    }
    ambiguity_by_index = {
        int(row["fact_index"]): row
        for row in ambiguity
        if row.get("fact_index") is not None
    }
    ignored: set[int] = set()
    labels: list[str] = []
    retained: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        change = by_index.get(index)
        adjudication = ambiguity_by_index.get(index)
        label = str((adjudication or {}).get("label") or (change or {}).get("label") or "")
        if label:
            labels.append(label)
        keep_ids = None
        if adjudication and adjudication.get("keep_eids") is not None:
            keep_ids = [str(item) for item in adjudication.get("keep_eids") or []]
        elif change and change.get("kept_evidence_ids") is not None:
            keep_ids = [str(item) for item in change.get("kept_evidence_ids") or []]
        if label in {"unsupported"} or change and change.get("change_type") == "drop_fact":
            continue
        if label == "ambiguous":
            ignored.add(index)
        if label == "supported_by_some" and keep_ids == []:
            ignored.add(index)
        updated = dict(fact) if isinstance(fact, dict) else {}
        if keep_ids is not None:
            updated["evidence_ids"] = keep_ids
        if not updated.get("evidence_ids"):
            continue
        retained.append(updated)
    payload["supported_facts"] = retained
    for change in changes:
        if change.get("change_type") == "action_flip":
            payload["next_action"] = str(change.get("new_action") or payload.get("next_action"))
            if payload["next_action"] != "answer_directly":
                payload.pop("supported_facts", None)
    return payload, ignored, labels


def pair_fact_scores(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]], ignored: set[int]
) -> tuple[float, float]:
    usable_expected = [row for index, row in enumerate(expected) if index not in ignored]
    if not predicted or not usable_expected:
        return 0.0, 0.0
    pairs: list[tuple[float, float]] = []
    for item in predicted:
        best = max(
            (
                (
                    text_similarity(item["fact"], target["fact"]),
                    len(item["evidence_ids"] & target["evidence_ids"])
                    / len(item["evidence_ids"] | target["evidence_ids"])
                    if item["evidence_ids"] | target["evidence_ids"]
                    else 0.0,
                )
                for target in usable_expected
            ),
            default=(0.0, 0.0),
        )
        pairs.append(best)
    return (
        sum(score for score, _ in pairs) / len(pairs),
        sum(score for _, score in pairs) / len(pairs),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--original-gold", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--ambiguity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_json(args.predictions)
    original_gold = read_json(args.original_gold)
    changes = read_jsonl(args.changes)
    ambiguity = read_jsonl(args.ambiguity)
    gold_by_id = {row_key(row, index): row for index, row in enumerate(original_gold)}
    change_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in changes:
        change_by_id.setdefault(str(row.get("row_id") or ""), []).append(row)
    ambiguity_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in ambiguity:
        ambiguity_by_id.setdefault(str(row.get("row_id") or ""), []).append(row)

    counts: Counter[str] = Counter()
    evidence_jaccards: list[float] = []
    fact_similarities: list[float] = []
    local_binding_scores: list[float] = []
    records: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        key = row_key(prediction, index)
        predicted = payload_from_row(prediction)
        original = gold_by_id.get(key)
        calibrated, ignored, labels = apply_calibration(
            payload_from_row(original),
            change_by_id.get(key, []),
            ambiguity_by_id.get(key, []),
        )
        if calibrated is None:
            counts["missing_gold"] += 1
            continue
        predicted_action = str((predicted or {}).get("next_action") or "invalid")
        expected_action = str(calibrated.get("next_action") or "")
        counts["rows"] += 1
        counts["action_labelled"] += 1
        counts["action_correct"] += int(predicted_action == expected_action)
        counts[f"action:{predicted_action}"] += 1
        counts[f"gold_action:{expected_action}"] += 1
        predicted_ids = fact_ids(predicted)
        expected_ids = fact_ids(calibrated)
        union = predicted_ids | expected_ids
        jaccard = len(predicted_ids & expected_ids) / len(union) if union else 0.0
        record_fact_similarity = None
        record_local_binding_score = None
        if expected_action == "answer_directly":
            counts["gold_answer_rows"] += 1
            counts["exact_evidence_set"] += int(predicted_ids == expected_ids)
            evidence_jaccards.append(jaccard)
            predicted_facts = fact_rows(predicted)
            expected_facts = fact_rows(calibrated)
            fact_score, local_score = pair_fact_scores(predicted_facts, expected_facts, ignored)
            fact_similarities.append(fact_score)
            local_binding_scores.append(local_score)
            record_fact_similarity = fact_score
            record_local_binding_score = local_score
        records.append(
            {
                "id": key,
                "predicted_action": predicted_action,
                "calibrated_action": expected_action,
                "predicted_evidence_ids": sorted(predicted_ids),
                "calibrated_evidence_ids": sorted(expected_ids),
                "evidence_jaccard": jaccard,
                "fact_similarity": record_fact_similarity,
                "local_binding_score": record_local_binding_score,
                "ignored_fact_indexes": sorted(ignored),
                "calibration_labels": labels,
            }
        )

    total = counts["rows"] or 1
    labelled_answers = counts["gold_answer_rows"] or 1
    summary = {
        "predictions": str(args.predictions),
        "original_gold": str(args.original_gold),
        "changes": str(args.changes),
        "ambiguity": str(args.ambiguity) if args.ambiguity else None,
        "counts": dict(sorted(counts.items())),
        "rates": {
            "action_accuracy": counts["action_correct"] / total,
            "exact_evidence_set_rate": counts["exact_evidence_set"] / labelled_answers,
            "mean_evidence_jaccard": sum(evidence_jaccards) / labelled_answers,
            "mean_fact_similarity": sum(fact_similarities) / labelled_answers,
            "mean_local_binding_score": sum(local_binding_scores) / labelled_answers,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
