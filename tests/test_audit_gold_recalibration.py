from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_gold_recalibration.py"
SPEC = importlib.util.spec_from_file_location("audit_gold_recalibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fact(fact_id: str, text: str, *, row_id: str = "row-1") -> dict:
    return {
        "fact_id": fact_id,
        "split": "train",
        "row_id": row_id,
        "question_key": "question-1",
        "fact_index": 0,
        "query_type": "fact",
        "claim": "甲做了某事",
        "cited_eids": ["E1"],
        "cited_texts": {"E1": text},
    }


def judgement(
    fact_id: str,
    *,
    merged: str,
    individual: str,
    chunk_chars: int,
    row_id: str = "row-1",
) -> dict:
    return {
        "fact_id": fact_id,
        "split": "train",
        "row_id": row_id,
        "fact_index": 0,
        "claim": "甲做了某事",
        "cited_eids": ["E1"],
        "merged": {"status": "ok", "verdict": merged},
        "individual": {"E1": {"status": "ok", "verdict": individual}},
        "judge_model": "glm-5.3",
        "reasoning_effort": "medium",
        "chunk_chars": chunk_chars,
    }


def test_audit_quarantines_truncation_and_conflict() -> None:
    facts = [
        fact("safe", "直接支持"),
        fact("truncated", "前文" * 400, row_id="row-2"),
        fact("conflict", "直接支持", row_id="row-3"),
    ]
    judgements = [
        judgement("safe", merged="supported", individual="supported", chunk_chars=600),
        judgement(
            "truncated",
            merged="unsupported",
            individual="unsupported",
            chunk_chars=600,
            row_id="row-2",
        ),
        judgement(
            "conflict",
            merged="unsupported",
            individual="supported",
            chunk_chars=600,
            row_id="row-3",
        ),
    ]
    report, quarantine = MODULE.build_report(facts, judgements)

    assert report["validation_errors"] == []
    assert report["strict_labels"] == {
        "clean_support": 1,
        "partial": 0,
        "unsupported": 2,
    }
    assert report["quarantined_facts"] == 2
    assert report["conservative_usable_by_label"] == {
        "clean_support": 1,
        "partial": 0,
        "unsupported": 0,
    }
    reasons = {row["fact_id"]: row["reasons"] for row in quarantine}
    assert reasons["truncated"] == ["judge_input_truncated"]
    assert reasons["conflict"] == ["merged_unsupported_ind_supported"]


def test_audit_marks_action_flip_unsafe_when_any_row_fact_is_quarantined() -> None:
    facts = [fact("truncated", "证据" * 400)]
    judgements = [
        judgement(
            "truncated",
            merged="unsupported",
            individual="unsupported",
            chunk_chars=600,
        )
    ]
    changes = [
        {
            "entity_type": "fact",
            "change_type": "drop_fact",
            "fact_id": "truncated",
        },
        {
            "entity_type": "row",
            "change_type": "action_flip",
            "split": "train",
            "row_id": "row-1",
        },
    ]
    report, _ = MODULE.build_report(facts, judgements, changes)

    assert report["changes"]["action_flip_safety"] == {
        "total": 1,
        "depends_on_quarantined_fact": 1,
    }


def test_audit_quarantines_non_ok_judge_status() -> None:
    facts = [fact("api-error", "直接支持")]
    row = judgement(
        "api-error",
        merged="supported",
        individual="supported",
        chunk_chars=600,
    )
    row["merged"]["status"] = "error"
    report, quarantine = MODULE.build_report(facts, [row])

    assert report["conservative_usable_facts"] == 0
    assert quarantine[0]["reasons"] == ["merged_status_not_ok"]
