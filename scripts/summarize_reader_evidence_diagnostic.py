#!/usr/bin/env python3
"""Summarize reader-by-evidence predictions and optional semantic audits."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import evaluate_exx_outputs as exx_eval  # noqa: E402


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DiagnosticSummaryError(ValueError):
    """Raised when predictions cannot be paired with the frozen matrix."""


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must have NAME=/path form")
    name, raw_path = value.split("=", 1)
    if not NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(f"invalid run name: {name}")
    return name, Path(raw_path)


def read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DiagnosticSummaryError(f"expected JSON object array: {path}")
    return [row for row in value if isinstance(row, dict)]


def read_audit_results(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise DiagnosticSummaryError(f"invalid grounding audit: {path}")
    results: dict[str, dict[str, Any]] = {}
    for row in value["results"]:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in results:
            raise DiagnosticSummaryError(f"duplicate/missing audit ID: {path}")
        results[row_id] = row
    return results


def diagnostic_value(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostic")
    if not isinstance(value, dict):
        raise DiagnosticSummaryError(f"row has no diagnostic metadata: {row.get('id')}")
    return value


def empty_group() -> dict[str, Any]:
    return {
        "rows": 0,
        "strict_json": 0,
        "schema_valid": 0,
        "generation_truncated": 0,
        "invalid_e_id": 0,
        "duplicate_fact": 0,
        "action_correct": 0,
        "over_retrieve": 0,
        "over_abstain": 0,
        "premature_answer": 0,
        "actions": collections.Counter(),
        "expected_actions": collections.Counter(),
        "semantic_status": collections.Counter(),
        "semantic_valid": 0,
        "set_support": collections.Counter(),
        "context_sufficiency": collections.Counter(),
        "action_appropriateness": collections.Counter(),
        "answer_rows_semantic": 0,
        "answer_with_nonentailed_fact": 0,
        "answer_with_unsupported_or_contradicted": 0,
        "answer_with_irrelevant_fact": 0,
        "answer_with_critical_unsupported": 0,
        "reference_sufficiency": collections.Counter(),
        "reference_sufficient_rows": 0,
        "reference_sufficient_complete": 0,
        "reference_sufficient_success": 0,
        "reference_insufficient_rows": 0,
        "reference_insufficient_appropriate_action": 0,
        "reference_insufficient_success": 0,
        "context_reference_agreement": 0,
        "reference_condition_success": 0,
    }


def add_protocol_result(
    group: dict[str, Any],
    *,
    prediction: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    prompt = exx_eval.prompt_value(prediction)
    visible = set(exx_eval.EVIDENCE_RE.findall(prompt))
    payload, strict = exx_eval.payload_from_row(prediction)
    problems = exx_eval.validate_payload(payload, visible)
    expected_payload, _ = exx_eval.payload_from_row(expected)
    action = str((payload or {}).get("next_action") or "invalid")
    expected_action = str((expected_payload or {}).get("next_action") or "invalid")
    group["rows"] += 1
    group["strict_json"] += int(strict)
    group["schema_valid"] += int(not problems)
    group["generation_truncated"] += int(
        exx_eval.generation_was_truncated(prediction)
    )
    group["invalid_e_id"] += int(any("unknown_id" in item for item in problems))
    group["duplicate_fact"] += int(any("duplicate" in item for item in problems))
    group["action_correct"] += int(action == expected_action)
    group["over_retrieve"] += int(
        expected_action == "answer_directly" and action == "retrieve_more"
    )
    group["over_abstain"] += int(
        expected_action == "answer_directly" and action == "abstain"
    )
    group["premature_answer"] += int(
        expected_action != "answer_directly" and action == "answer_directly"
    )
    group["actions"][action] += 1
    group["expected_actions"][expected_action] += 1
    return {
        "action": action,
        "expected_action": expected_action,
        "strict_json": strict,
        "schema_valid": not problems,
        "problems": problems,
    }


def add_semantic_result(
    group: dict[str, Any],
    audit: dict[str, Any] | None,
    *,
    protocol_action: str,
    reference_sufficiency: str,
) -> dict[str, Any]:
    reference_sufficiency = (
        reference_sufficiency
        if reference_sufficiency in {"sufficient", "insufficient"}
        else "unknown"
    )
    group["reference_sufficiency"][reference_sufficiency] += 1
    if audit is None:
        group["semantic_status"]["not_run"] += 1
        return {
            "status": "not_run",
            "reference_sufficiency": reference_sufficiency,
        }
    status = str(audit.get("status") or "unknown")
    group["semantic_status"][status] += 1
    if status != "ok" or not isinstance(audit.get("judgement"), dict):
        return {
            "status": status,
            "reference_sufficiency": reference_sufficiency,
        }
    audit_action = str(audit.get("action") or "unknown")
    if audit_action != protocol_action:
        group["semantic_status"]["action_mismatch"] += 1
        return {
            "status": "action_mismatch",
            "reference_sufficiency": reference_sufficiency,
            "audit_action": audit_action,
            "protocol_action": protocol_action,
        }
    judgement = audit["judgement"]
    group["semantic_valid"] += 1
    set_support = str(judgement.get("set_support") or "unknown")
    group["set_support"][set_support] += 1
    group["context_sufficiency"][
        str(judgement.get("context_sufficiency") or "unknown")
    ] += 1
    group["action_appropriateness"][
        str(judgement.get("action_appropriateness") or "unknown")
    ] += 1
    context_sufficiency = str(
        judgement.get("context_sufficiency") or "unknown"
    )
    action_appropriateness = str(
        judgement.get("action_appropriateness") or "unknown"
    )
    context_matches_reference = int(
        context_sufficiency == reference_sufficiency
    )
    group["context_reference_agreement"] += context_matches_reference
    reference_success = False
    if reference_sufficiency == "sufficient":
        group["reference_sufficient_rows"] += 1
        complete = set_support == "complete"
        group["reference_sufficient_complete"] += int(complete)
        reference_success = (
            complete
            and context_sufficiency == "sufficient"
            and action_appropriateness == "appropriate"
        )
        group["reference_sufficient_success"] += int(reference_success)
    elif reference_sufficiency == "insufficient":
        group["reference_insufficient_rows"] += 1
        appropriate_action = action_appropriateness == "appropriate"
        group["reference_insufficient_appropriate_action"] += int(
            appropriate_action
        )
        reference_success = (
            context_sufficiency == "insufficient"
            and appropriate_action
        )
        group["reference_insufficient_success"] += int(reference_success)
    group["reference_condition_success"] += int(reference_success)
    action = audit_action
    facts = [
        fact
        for fact in judgement.get("facts") or []
        if isinstance(fact, dict)
    ]
    if action == "answer_directly":
        group["answer_rows_semantic"] += 1
        group["answer_with_nonentailed_fact"] += int(
            any(fact.get("support") != "entailed" for fact in facts)
        )
        group["answer_with_unsupported_or_contradicted"] += int(
            any(
                fact.get("support") in {"unsupported", "contradicted"}
                for fact in facts
            )
        )
        group["answer_with_irrelevant_fact"] += int(
            any(fact.get("question_relevance") == "irrelevant" for fact in facts)
        )
        group["answer_with_critical_unsupported"] += int(
            int(judgement.get("critical_unsupported_claims") or 0) > 0
        )
    return {
        "status": status,
        "set_support": set_support,
        "context_sufficiency": context_sufficiency,
        "action_appropriateness": action_appropriateness,
        "reference_sufficiency": reference_sufficiency,
        "context_matches_reference": bool(context_matches_reference),
        "reference_success": reference_success,
        "audit_action": audit_action,
        "protocol_action": protocol_action,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    rows = int(group["rows"])
    semantic_valid = int(group["semantic_valid"])
    answer_rows = int(group["answer_rows_semantic"])
    count_fields = (
        "strict_json",
        "schema_valid",
        "generation_truncated",
        "invalid_e_id",
        "duplicate_fact",
        "action_correct",
        "over_retrieve",
        "over_abstain",
        "premature_answer",
    )
    protocol_counts = {field: int(group[field]) for field in count_fields}
    return {
        "rows": rows,
        "protocol": {
            "counts": protocol_counts,
            "rates": {
                field: safe_rate(value, rows)
                for field, value in protocol_counts.items()
            },
            "actions": dict(sorted(group["actions"].items())),
            "expected_actions": dict(sorted(group["expected_actions"].items())),
        },
        "semantic": {
            "status": dict(sorted(group["semantic_status"].items())),
            "valid": semantic_valid,
            "audit_coverage": safe_rate(semantic_valid, rows),
            "reference_sufficiency": dict(
                sorted(group["reference_sufficiency"].items())
            ),
            "set_support": dict(sorted(group["set_support"].items())),
            "context_sufficiency": dict(
                sorted(group["context_sufficiency"].items())
            ),
            "action_appropriateness": dict(
                sorted(group["action_appropriateness"].items())
            ),
            "complete_per_all_rows": safe_rate(
                group["set_support"]["complete"], rows
            ),
            "complete_per_valid_audit": safe_rate(
                group["set_support"]["complete"], semantic_valid
            ),
            "complete_per_reference_sufficient": safe_rate(
                group["reference_sufficient_complete"],
                group["reference_sufficient_rows"],
            ),
            "reference_condition_success_per_all_rows": safe_rate(
                group["reference_condition_success"], rows
            ),
            "reference_condition_success_per_valid_audit": safe_rate(
                group["reference_condition_success"], semantic_valid
            ),
            "appropriate_action_per_reference_insufficient": safe_rate(
                group["reference_insufficient_appropriate_action"],
                group["reference_insufficient_rows"],
            ),
            "reference_context_agreement_per_valid_audit": safe_rate(
                group["context_reference_agreement"], semantic_valid
            ),
            "answer_rows": answer_rows,
            "answer_level_rates": {
                "any_nonentailed_fact": safe_rate(
                    group["answer_with_nonentailed_fact"], answer_rows
                ),
                "any_unsupported_or_contradicted": safe_rate(
                    group["answer_with_unsupported_or_contradicted"],
                    answer_rows,
                ),
                "any_irrelevant_fact": safe_rate(
                    group["answer_with_irrelevant_fact"], answer_rows
                ),
                "any_critical_unsupported": safe_rate(
                    group["answer_with_critical_unsupported"], answer_rows
                ),
            },
        },
    }


def build_report(
    *,
    gold_rows: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    audits: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in gold_rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in gold_by_id:
            raise DiagnosticSummaryError("gold matrix IDs must be non-empty and unique")
        diagnostic_value(row)
        gold_by_id[row_id] = row
    unknown_audit_names = sorted(set(audits) - set(predictions))
    if unknown_audit_names:
        raise DiagnosticSummaryError(
            "audits have no matching prediction run: " + ",".join(unknown_audit_names)
        )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for reader, rows in predictions.items():
        prediction_ids: set[str] = set()
        reader_groups: dict[str, dict[str, Any]] = {}
        audit_rows = audits.get(reader, {})
        for prediction in rows:
            row_id = str(prediction.get("id") or "").strip()
            if not row_id or row_id in prediction_ids:
                raise DiagnosticSummaryError(
                    f"{reader}: prediction IDs must be non-empty and unique"
                )
            prediction_ids.add(row_id)
            expected = gold_by_id.get(row_id)
            if expected is None:
                raise DiagnosticSummaryError(
                    f"{reader}: prediction ID absent from gold matrix: {row_id}"
                )
            diagnostic = diagnostic_value(expected)
            condition = str(diagnostic.get("condition") or "").strip()
            if not condition:
                raise DiagnosticSummaryError(f"{row_id}: missing condition")
            group = reader_groups.setdefault(condition, empty_group())
            protocol = add_protocol_result(
                group,
                prediction=prediction,
                expected=expected,
            )
            semantic = add_semantic_result(
                group,
                audit_rows.get(row_id),
                protocol_action=protocol["action"],
                reference_sufficiency=str(
                    diagnostic.get("reference_sufficiency") or "unknown"
                ),
            )
            records.append(
                {
                    "reader": reader,
                    "id": row_id,
                    "source_id": diagnostic.get("source_id"),
                    "question_family": diagnostic.get("question_family"),
                    "condition": condition,
                    "reference_sufficiency": diagnostic.get(
                        "reference_sufficiency"
                    ),
                    **protocol,
                    "semantic": semantic,
                }
            )
        missing = sorted(set(gold_by_id) - prediction_ids)
        if missing:
            raise DiagnosticSummaryError(
                f"{reader}: predictions missing {len(missing)} gold rows"
            )
        extra_audits = sorted(set(audit_rows) - prediction_ids)
        if extra_audits:
            raise DiagnosticSummaryError(
                f"{reader}: audit IDs absent from predictions: "
                + ",".join(extra_audits)
            )
        grouped[reader] = {
            condition: finalize_group(group)
            for condition, group in sorted(reader_groups.items())
        }
    return {
        "protocol": "asa_reader_evidence_diagnostic_summary_v1",
        "gold_rows": len(gold_rows),
        "readers": grouped,
        "records": records,
        "interpretation": {
            "reference_condition_success_per_all_rows_is_primary": True,
            "complete_per_all_rows_is_a_lower_bound": True,
            "complete_is_only_answer_success_for_reference_sufficient_rows": True,
            "insufficient_rows_use_appropriate_action_as_success": True,
            "semantic_not_run_is_not_counted_as_verified_success": True,
            "schema_or_action_accuracy_is_not_answer_correctness": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=named_path,
        required=True,
        help="NAME=/path/to/predictions.json",
    )
    parser.add_argument(
        "--audit",
        action="append",
        type=named_path,
        default=[],
        help="NAME=/path/to/set-audit.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    prediction_paths = dict(args.prediction)
    audit_paths = dict(args.audit)
    if len(prediction_paths) != len(args.prediction):
        parser.error("prediction run names must be unique")
    if len(audit_paths) != len(args.audit):
        parser.error("audit run names must be unique")
    report = build_report(
        gold_rows=read_array(args.gold),
        predictions={
            name: read_array(path) for name, path in prediction_paths.items()
        },
        audits={
            name: read_audit_results(path) for name, path in audit_paths.items()
        },
    )
    report["inputs"] = {
        "gold": str(args.gold.resolve()),
        "predictions": {
            name: str(path.resolve()) for name, path in prediction_paths.items()
        },
        "audits": {
            name: str(path.resolve()) for name, path in audit_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
