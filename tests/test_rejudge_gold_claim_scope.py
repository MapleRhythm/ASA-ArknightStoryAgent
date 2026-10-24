import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rejudge_gold_claim_scope.py"
SPEC = importlib.util.spec_from_file_location("claim_scope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fact():
    return {
        "fact_id": "f", "row_id": "r", "fact_index": 0, "claim": "甲完成任务，并成为领袖。",
        "cited_eids": ["E1", "E2"], "cited_texts": {"E1": "甲完成任务。", "E2": "甲成为领袖。"},
        "hidden_reference": "PRIVATE_GOLD",
    }


def verdict(scope="fully_supported", eids=None, parts=None):
    return {
        "claim_support": scope, "supporting_eids": ["E1"] if eids is None else eids,
        "unsupported_parts": [] if parts is None else parts, "reason": "理由",
    }


def test_complete_claim_scope_is_independent_of_evidence_count():
    assert MODULE.validate(verdict(), fact())["claim_support"] == "fully_supported"
    assert MODULE.validate(verdict(eids=["E1", "E2"]), fact())["claim_support"] == "fully_supported"
    assert MODULE.LABEL_MAP["fully_supported"] == "supported_by_union"


def test_partial_requires_literal_unsupported_part():
    with pytest.raises(ValueError, match="parts_conflict"):
        MODULE.validate(verdict("partially_supported"), fact())
    assert MODULE.validate(verdict("partially_supported", parts=["成为领袖"]), fact())
    with pytest.raises(ValueError, match="literal"):
        MODULE.validate(verdict("partially_supported", parts=["原主张没有这句话"]), fact())
    with pytest.raises(ValueError, match="parts_conflict"):
        MODULE.validate(verdict(parts=["成为领袖"]), fact())


def test_invalid_or_duplicate_ids_are_not_silently_filtered():
    for eids in (["E9"], ["E1", "E1"], [], "E1"):
        with pytest.raises(ValueError):
            MODULE.validate(verdict(eids=eids), fact())
    assert MODULE.validate(verdict("unsupported", eids=[]), fact())


def test_prior_not_in_prompt_and_candidate_fingerprint_gate():
    source = fact()
    source["prior_label"] = "SECRET_OLD"
    prior = {
        "fact_id": "f", "row_id": "r", "fact_index": 0, "status": "ok",
        "full_evidence": True, "evidence_sha256": MODULE.evidence_fingerprint(source),
        "label": "supported_by_some", "reason": "PRIVATE_REASON", "keep_eids": ["E1"],
    }
    selected = MODULE.candidates([source], [prior])
    sent = json.dumps(MODULE.messages(selected[0]), ensure_ascii=False)
    assert "PRIVATE" not in sent and "SECRET" not in sent
    assert "只需一段就证明全部时也必须用此标签" in sent
    source["cited_texts"]["E1"] += "stale"
    with pytest.raises(ValueError, match="stale"):
        MODULE.candidates([source], [prior])


def test_duplicate_fact_ids_and_changed_prompt_keys():
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.candidates([fact(), fact()], [])
    source = fact()
    initial = MODULE.input_key(source, 4096)
    assert initial != MODULE.input_key(source, 8192)
    source["claim"] += "新限定"
    assert initial != MODULE.input_key(source, 4096)


def test_resume_can_reuse_completed_old_budget_but_not_stale_or_partial_output():
    source = fact()
    row = {
        **verdict(), "fact_id": "f", "row_id": "r", "fact_index": 0,
        "protocol": MODULE.PROTOCOL, "endpoint": MODULE.ENDPOINT, "judge_model": MODULE.MODEL,
        "max_tokens": 4096, "input_key": MODULE.input_key(source, 4096),
        "status": "ok", "finish_reason": "stop", "full_evidence": True,
        "evidence_sha256": MODULE.evidence_fingerprint(source),
        "label": "supported_by_union", "keep_eids": ["E1"],
    }
    assert MODULE.reusable(source, row)
    assert not MODULE.reusable(source, row | {"finish_reason": "length"})
    assert not MODULE.reusable(source, row | {"label": "supported_by_some"})
    assert not MODULE.reusable(source, row | {"max_tokens": 8192})
