import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_reader_evidence_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_reader_evidence_diagnostic", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(
    row_id: str,
    condition: str,
    payload: dict,
    *,
    reference_sufficiency: str = "sufficient",
) -> dict:
    prompt = "\n".join(
        [
            "task: grounded_action_generation",
            "question: 甲为什么行动？",
            "hypothesis: {}",
            "round: 1/2",
            "evidence:",
            "[E1]",
            "甲为了保护乙而行动。",
            "output_schema: grounded_action_exx_v1",
            "rules: test",
        ]
    )
    return {
        "id": row_id,
        "task_type": "grounded_action_generation",
        "system": "只输出 JSON。",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(payload, ensure_ascii=False)},
        ],
        "diagnostic": {
            "source_id": "q1",
            "question_family": "family-1",
            "condition": condition,
            "reference_sufficiency": reference_sufficiency,
        },
    }


def _answer_payload() -> dict:
    return {
        "next_action": "answer_directly",
        "supported_facts": [
            {
                "fact": "甲为了保护乙而行动。",
                "evidence_ids": ["E1"],
            }
        ],
    }


def _audit(row_id: str) -> dict:
    return {
        "id": row_id,
        "status": "ok",
        "action": "answer_directly",
        "judgement": {
            "facts": [
                {
                    "fact_index": 0,
                    "support": "entailed",
                    "question_relevance": "direct",
                    "checked_evidence_ids": ["E1"],
                    "citation_complete": True,
                }
            ],
            "relations": [],
            "set_support": "complete",
            "missing_requirements": [],
            "critical_unsupported_claims": 0,
            "irrelevant_claims": 0,
            "context_sufficiency": "sufficient",
            "action_appropriateness": "appropriate",
        },
    }


def _retrieve_audit(row_id: str) -> dict:
    result = _audit(row_id)
    result["action"] = "retrieve_more"
    result["judgement"]["facts"] = []
    result["judgement"]["set_support"] = "none"
    result["judgement"]["context_sufficiency"] = "insufficient"
    result["judgement"]["action_appropriateness"] = "appropriate"
    return result


def test_build_report_keeps_protocol_and_semantic_metrics_separate() -> None:
    gold = _row("q1::minimal_sufficient", "minimal_sufficient", _answer_payload())
    prediction = _row(
        "q1::minimal_sufficient",
        "minimal_sufficient",
        _answer_payload(),
    )

    report = MODULE.build_report(
        gold_rows=[gold],
        predictions={"base4b": [prediction]},
        audits={"base4b": {prediction["id"]: _audit(prediction["id"])}},
    )

    group = report["readers"]["base4b"]["minimal_sufficient"]
    assert group["protocol"]["rates"]["action_correct"] == 1.0
    assert group["protocol"]["rates"]["schema_valid"] == 1.0
    assert group["semantic"]["complete_per_all_rows"] == 1.0
    assert group["semantic"]["complete_per_reference_sufficient"] == 1.0
    assert group["semantic"]["reference_condition_success_per_all_rows"] == 1.0
    assert group["semantic"]["answer_level_rates"][
        "any_unsupported_or_contradicted"
    ] == 0.0


def test_build_report_rejects_partial_prediction_matrix() -> None:
    gold_rows = [
        _row("q1::minimal_sufficient", "minimal_sufficient", _answer_payload()),
        _row(
            "q1::sufficient_with_distractors",
            "sufficient_with_distractors",
            _answer_payload(),
        ),
    ]
    with pytest.raises(MODULE.DiagnosticSummaryError, match="missing 1 gold rows"):
        MODULE.build_report(
            gold_rows=gold_rows,
            predictions={"base4b": [gold_rows[0]]},
            audits={},
        )


def test_insufficient_reference_uses_appropriate_retrieve_action() -> None:
    gold = _row(
        "q1::actual_prompt",
        "actual_prompt",
        {"next_action": "retrieve_more", "follow_up_hypothesis": {}},
        reference_sufficiency="insufficient",
    )
    prediction = _row(
        "q1::actual_prompt",
        "actual_prompt",
        {"next_action": "retrieve_more", "follow_up_hypothesis": {}},
        reference_sufficiency="insufficient",
    )
    report = MODULE.build_report(
        gold_rows=[gold],
        predictions={"base4b": [prediction]},
        audits={"base4b": {prediction["id"]: _retrieve_audit(prediction["id"])}},
    )
    semantic = report["readers"]["base4b"]["actual_prompt"]["semantic"]
    assert semantic["complete_per_all_rows"] == 0.0
    assert semantic["appropriate_action_per_reference_insufficient"] == 1.0
    assert semantic["reference_condition_success_per_all_rows"] == 1.0


def test_audit_action_mismatch_is_not_counted_as_valid_semantics() -> None:
    gold = _row(
        "q1::minimal_sufficient",
        "minimal_sufficient",
        _answer_payload(),
    )
    prediction = _row(
        "q1::minimal_sufficient",
        "minimal_sufficient",
        _answer_payload(),
    )
    audit = _audit(prediction["id"])
    audit["action"] = "retrieve_more"
    report = MODULE.build_report(
        gold_rows=[gold],
        predictions={"base4b": [prediction]},
        audits={"base4b": {prediction["id"]: audit}},
    )
    semantic = report["readers"]["base4b"]["minimal_sufficient"]["semantic"]
    assert semantic["valid"] == 0
    assert semantic["status"]["action_mismatch"] == 1
