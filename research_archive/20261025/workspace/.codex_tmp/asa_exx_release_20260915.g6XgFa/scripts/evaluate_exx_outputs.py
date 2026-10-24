#!/usr/bin/env python3
"""Evaluate raw or normalized grounded_action_exx_v1 model outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
ACTIONS = ("answer_directly", "retrieve_more", "abstain")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def assistant_value(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[-1].get("value") or "")
    return str(row.get("raw_output") or row.get("output") or "")


def prompt_value(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[0].get("value") or "")
    return str(row.get("prompt") or "")


def payload_from_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    direct = row.get("payload")
    if isinstance(direct, dict):
        return direct, True
    text = assistant_value(row).strip()
    try:
        value = json.loads(text)
        return (value if isinstance(value, dict) else None), isinstance(value, dict)
    except json.JSONDecodeError:
        return None, False


def validate_payload(payload: dict[str, Any] | None, visible: set[str]) -> list[str]:
    if payload is None:
        return ["invalid_json"]
    problems: list[str] = []
    forbidden = FORBIDDEN.intersection(payload)
    if forbidden:
        problems.append("legacy_fields:" + ",".join(sorted(forbidden)))
    action = str(payload.get("next_action") or "")
    if action not in ACTIONS:
        problems.append("invalid_action")
        return problems
    allowed_top = {"next_action", "supported_facts"} if action == "answer_directly" else (
        {"next_action", "follow_up_hypothesis"} if action == "retrieve_more" else {"next_action", "reason"}
    )
    extra = set(payload) - allowed_top
    if extra:
        problems.append("unexpected_fields:" + ",".join(sorted(extra)))
    if action == "answer_directly":
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            problems.append("invalid_fact_count")
            return problems
        for index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                problems.append(f"fact_{index}_schema")
                continue
            ids = fact.get("evidence_ids")
            if not str(fact.get("fact") or "").strip():
                problems.append(f"fact_{index}_empty")
            if not isinstance(ids, list) or not 1 <= len(ids) <= 2:
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible for item in ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more" and not isinstance(payload.get("follow_up_hypothesis"), dict):
        problems.append("missing_follow_up")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_rows(args.predictions)
    gold_rows = read_rows(args.gold) if args.gold else []
    gold_by_id = {str(row.get("id") or row.get("task_id") or i): row for i, row in enumerate(gold_rows)}
    counts = Counter()
    action_matrix = Counter()
    records = []
    for index, row in enumerate(predictions):
        key = str(row.get("id") or row.get("task_id") or index)
        prompt = prompt_value(row)
        visible = set(EVIDENCE_RE.findall(prompt))
        payload, strict = payload_from_row(row)
        problems = validate_payload(payload, visible)
        counts["total"] += 1
        counts["strict_json"] += int(strict)
        counts["schema_valid"] += int(not problems)
        action = str((payload or {}).get("next_action") or "invalid")
        counts[f"action:{action}"] += 1
        counts["legacy_field_rows"] += int(any(item.startswith("legacy_fields:") for item in problems))
        counts["invalid_e_id_rows"] += int(any("unknown_id" in item for item in problems))
        gold = gold_by_id.get(key)
        gold_payload, _ = payload_from_row(gold) if gold else (None, False)
        gold_action = str((gold_payload or {}).get("next_action") or "")
        if gold_action:
            action_matrix[f"{gold_action}->{action}"] += 1
            counts["action_labelled"] += 1
            counts["action_correct"] += int(gold_action == action)
            counts["over_abstain"] += int(gold_action == "answer_directly" and action == "abstain")
            counts["over_retrieve"] += int(gold_action == "answer_directly" and action == "retrieve_more")
            counts["premature_answer"] += int(gold_action != "answer_directly" and action == "answer_directly")
        records.append({"id": key, "action": action, "gold_action": gold_action, "problems": problems})

    total = counts["total"] or 1
    labelled = counts["action_labelled"] or 1
    summary = {
        "counts": dict(sorted(counts.items())),
        "rates": {
            "strict_json_rate": round(counts["strict_json"] / total, 6),
            "schema_valid_rate": round(counts["schema_valid"] / total, 6),
            "legacy_field_rate": round(counts["legacy_field_rows"] / total, 6),
            "invalid_e_id_rate": round(counts["invalid_e_id_rows"] / total, 6),
            "action_accuracy": round(counts["action_correct"] / labelled, 6),
            "over_abstain_rate": round(counts["over_abstain"] / labelled, 6),
            "over_retrieve_rate": round(counts["over_retrieve"] / labelled, 6),
            "premature_answer_rate": round(counts["premature_answer"] / labelled, 6),
        },
        "action_matrix": dict(sorted(action_matrix.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
