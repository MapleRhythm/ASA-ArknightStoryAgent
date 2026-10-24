#!/usr/bin/env python3
"""Analyze protocol behavior and optional semantic review for the pilot."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
VALID_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
FOLLOW_UP_REQUIRED = {
    "question",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
}
FOLLOW_UP_OPTIONAL = {"dialogue_context"}
ANSWER_CONDITIONS = {
    "proof_a",
    "proof_b",
    "dual_proof",
    "renumbered_b_with_distractor",
    "restored_round2",
}
BROKEN_CONDITIONS = {"broken_round1", "broken_round2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def assistant_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("raw_output"), str):
        return row["raw_output"].strip()
    conversations = row.get("conversations") or []
    return str(conversations[-1].get("value") or "").strip() if conversations else ""


def prompt_text(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    return str(conversations[0].get("value") or "") if conversations else ""


def parse_payload(row: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    try:
        value = json.loads(assistant_text(row))
    except json.JSONDecodeError:
        return None, False
    return (value if isinstance(value, dict) else None), isinstance(value, dict)


def protocol_problems(payload: dict[str, Any] | None, visible: set[str]) -> list[str]:
    if payload is None:
        return ["invalid_json"]
    action = str(payload.get("next_action") or "")
    if action not in VALID_ACTIONS:
        return ["invalid_action"]
    problems = []
    forbidden = FORBIDDEN & set(payload)
    if forbidden:
        problems.append("forbidden:" + ",".join(sorted(forbidden)))
    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            problems.append("top_schema")
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return [*problems, "invalid_fact_count"]
        seen_facts = set()
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                problems.append(f"fact_{index}_schema")
                continue
            ids = fact.get("evidence_ids")
            if not isinstance(fact.get("fact"), str) or not fact["fact"].strip():
                problems.append(f"fact_{index}_empty")
            normalized = re.sub(
                r"[^\w\u3400-\u9fff]+", "", str(fact.get("fact") or "").lower()
            )
            if normalized in seen_facts:
                problems.append(f"fact_{index}_duplicate")
            elif normalized:
                seen_facts.add(normalized)
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or len(set(map(str, ids))) != len(ids)
            ):
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible for item in ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            problems.append("top_schema")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict):
            problems.append("missing_follow_up")
        else:
            keys = set(follow_up)
            if not FOLLOW_UP_REQUIRED <= keys or keys - (
                FOLLOW_UP_REQUIRED | FOLLOW_UP_OPTIONAL
            ):
                problems.append("follow_up_schema")
            for name in ("question", "query_type", "expected_answer_type"):
                if not isinstance(follow_up.get(name), str) or not follow_up[name].strip():
                    problems.append(f"follow_up_{name}")
            for name in ("entities", "keywords"):
                values = follow_up.get(name)
                if (
                    not isinstance(values, list)
                    or len(values) > 8
                    or len({str(item) for item in values}) != len(values)
                    or any(not isinstance(item, str) or not item.strip() for item in values)
                ):
                    problems.append(f"follow_up_{name}")
            if (
                "dialogue_context" in follow_up
                and not isinstance(follow_up["dialogue_context"], str)
            ):
                problems.append("follow_up_dialogue_context")
    else:
        if set(payload) != {"next_action", "reason"}:
            problems.append("top_schema")
        if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
            problems.append("missing_reason")
    return problems


def load_review(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows = read_array(path)
    result = {}
    for row in rows:
        key = (str(row.get("reader") or ""), str(row.get("id") or ""))
        if not all(key) or key in result:
            raise ValueError("semantic review has duplicate or missing reader/id")
        if row.get("semantic_label") not in {
            "complete",
            "partial",
            "incorrect",
            "appropriate_no_answer",
            "uncertain",
        }:
            raise ValueError(f"invalid semantic label for {key}")
        result[key] = row
    return result


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_reader(
    name: str,
    rows: list[dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    review: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    condition_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    family_rows: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id not in expected:
            raise ValueError(f"{name}: prediction ID is absent from matrix: {row_id}")
        source = expected[row_id]
        diagnostic = source["diagnostic"]
        condition = diagnostic["condition"]
        payload, strict_json = parse_payload(row)
        action = str((payload or {}).get("next_action") or "invalid")
        visible = set(EVIDENCE_RE.findall(prompt_text(row)))
        problems = protocol_problems(payload, visible)
        expected_action = diagnostic["expected_action"]
        semantic = review.get((name, row_id))
        semantic_label = semantic.get("semantic_label") if semantic else "not_reviewed"
        expected_complete = condition in ANSWER_CONDITIONS
        semantic_content_success = (
            semantic_label == "complete"
            and semantic.get("citation_support") is True
            if expected_complete
            else semantic_label == "appropriate_no_answer"
        )
        semantic_success = (
            semantic_content_success
            and not problems
            and action == expected_action
        )
        detail = {
            "reader": name,
            "id": row_id,
            "source_id": diagnostic["source_id"],
            "short_name": diagnostic["short_name"],
            "condition": condition,
            "proof_topology": diagnostic["proof_topology"],
            "expected_action": expected_action,
            "action": action,
            "strict_json": strict_json,
            "protocol_valid": not problems,
            "protocol_problems": problems,
            "action_correct": action == expected_action,
            "generation_truncated": bool(row.get("hit_max_new_tokens"))
            or str(row.get("finish_reason") or "") == "length",
            "semantic_label": semantic_label,
            "semantic_success": semantic_success if semantic else None,
            "citation_support": semantic.get("citation_support") if semantic else None,
            "review_notes": semantic.get("notes") if semantic else None,
        }
        details.append(detail)
        counter = condition_counts[condition]
        counter["rows"] += 1
        counter["strict_json"] += int(strict_json)
        counter["protocol_valid"] += int(not problems)
        counter["action_correct"] += int(action == expected_action)
        counter["truncated"] += int(detail["generation_truncated"])
        counter[f"action:{action}"] += 1
        if semantic:
            counter["reviewed"] += 1
            counter[f"semantic:{semantic_label}"] += 1
            counter["semantic_success"] += int(semantic_success)
            counter["citation_supported"] += int(
                semantic.get("citation_support") is True
            )
        family_rows[diagnostic["source_id"]][condition] = detail

    if len(details) != len(expected) or len({item["id"] for item in details}) != len(details):
        raise ValueError(f"{name}: predictions must cover every matrix row once")

    conditions = {}
    for condition, counter in sorted(condition_counts.items()):
        row_count = counter["rows"]
        reviewed = counter["reviewed"]
        conditions[condition] = {
            "rows": row_count,
            "strict_json": counter["strict_json"],
            "protocol_valid": counter["protocol_valid"],
            "action_correct": counter["action_correct"],
            "generation_truncated": counter["truncated"],
            "semantic_reviewed": reviewed,
            "semantic_success": counter["semantic_success"],
            "semantic_success_rate": safe_rate(counter["semantic_success"], reviewed),
            "complete": counter["semantic:complete"],
            "partial": counter["semantic:partial"],
            "incorrect": counter["semantic:incorrect"],
            "appropriate_no_answer": counter["semantic:appropriate_no_answer"],
            "uncertain": counter["semantic:uncertain"],
            "actions": {
                action: counter[f"action:{action}"] for action in sorted(VALID_ACTIONS)
            }
        }

    cycles = []
    for source_id, by_condition in sorted(family_rows.items()):
        proof_states = [
            by_condition[name]
            for name in (
                "proof_a",
                "proof_b",
                "dual_proof",
                "renumbered_b_with_distractor",
            )
        ]
        broken1 = by_condition["broken_round1"]
        broken2 = by_condition["broken_round2"]
        restored = by_condition["restored_round2"]
        cycles.append(
            {
                "source_id": source_id,
                "short_name": proof_states[0]["short_name"],
                "proof_topology": proof_states[0]["proof_topology"],
                "action_preserved_across_four_sufficient_states": all(
                    item["action"] == "answer_directly" for item in proof_states
                ),
                "semantic_preserved_across_four_sufficient_states": (
                    all(item["semantic_success"] is True for item in proof_states)
                    if all(item["semantic_label"] != "not_reviewed" for item in proof_states)
                    else None
                ),
                "broken_round1_identified": broken1["action"] == "retrieve_more",
                "broken_round2_identified": broken2["action"] == "abstain",
                "restored_action_recovered": restored["action"] == "answer_directly",
                "restored_semantic_recovered": (
                    restored["semantic_success"] is True
                    if restored["semantic_label"] != "not_reviewed"
                    else None
                ),
                "full_cycle_action_success": (
                    all(item["action"] == "answer_directly" for item in proof_states)
                    and broken1["action"] == "retrieve_more"
                    and broken2["action"] == "abstain"
                    and restored["action"] == "answer_directly"
                ),
                "full_cycle_semantic_success": (
                    all(item["semantic_success"] is True for item in proof_states)
                    and broken1["semantic_success"] is True
                    and broken2["semantic_success"] is True
                    and restored["semantic_success"] is True
                    if all(
                        item["semantic_label"] != "not_reviewed"
                        for item in [*proof_states, broken1, broken2, restored]
                    )
                    else None
                ),
            }
        )

    reviewed_all = all(item["semantic_label"] != "not_reviewed" for item in details)
    aggregate = {
        "rows": len(details),
        "strict_json": sum(item["strict_json"] for item in details),
        "protocol_valid": sum(item["protocol_valid"] for item in details),
        "action_correct": sum(item["action_correct"] for item in details),
        "generation_truncated": sum(item["generation_truncated"] for item in details),
        "semantic_review_complete": reviewed_all,
        "proof_retention_action": sum(
            item["action"] == "answer_directly"
            for item in details
            if item["condition"] in ANSWER_CONDITIONS
        ),
        "proof_retention_denominator": sum(
            item["condition"] in ANSWER_CONDITIONS for item in details
        ),
        "broken_state_action": sum(
            item["action"] == item["expected_action"]
            for item in details
            if item["condition"] in BROKEN_CONDITIONS
        ),
        "broken_state_denominator": sum(
            item["condition"] in BROKEN_CONDITIONS for item in details
        ),
        "full_cycle_action_success": sum(
            item["full_cycle_action_success"] for item in cycles
        ),
        "full_cycle_semantic_success": (
            sum(bool(item["full_cycle_semantic_success"]) for item in cycles)
            if reviewed_all
            else None
        ),
        "families": len(cycles),
    }
    if reviewed_all:
        aggregate["semantic_success"] = sum(
            bool(item["semantic_success"]) for item in details
        )
        aggregate["semantic_rows"] = len(details)
        aggregate["semantic_proof_retention"] = sum(
            item["semantic_success"] is True
            for item in details
            if item["condition"] in ANSWER_CONDITIONS
        )
        aggregate["semantic_broken_state_success"] = sum(
            item["semantic_success"] is True
            for item in details
            if item["condition"] in BROKEN_CONDITIONS
        )

    return {
        "aggregate": aggregate,
        "conditions": conditions,
        "cycles": cycles,
    }, details


def parse_named_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must have NAME=/path form")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("prediction must have NAME=/path form")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--prediction", type=parse_named_prediction, action="append", required=True
    )
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_rows = read_array(args.matrix)
    expected = {str(row["id"]): row for row in matrix_rows}
    if len(expected) != len(matrix_rows):
        raise ValueError("matrix contains duplicate IDs")
    review = load_review(args.semantic_review)
    readers = {}
    details = []
    prediction_files = {}
    for name, path in args.prediction:
        if name in readers:
            raise ValueError(f"duplicate reader name: {name}")
        summary, reader_details = summarize_reader(
            name, read_array(path), expected, review
        )
        readers[name] = summary
        details.extend(reader_details)
        prediction_files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    output = {
        "schema_version": "asa_alternative_proof_pilot_summary_v1",
        "matrix": {
            "path": str(args.matrix),
            "sha256": sha256_file(args.matrix),
            "rows": len(matrix_rows),
        },
        "prediction_files": prediction_files,
        "semantic_review": (
            {
                "path": str(args.semantic_review),
                "sha256": sha256_file(args.semantic_review),
                "rows": len(review),
            }
            if args.semantic_review
            else None
        ),
        "readers": readers,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "readers": readers}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
