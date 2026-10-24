import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge_gold_claim_scope.py"
SPEC = importlib.util.spec_from_file_location("merge_scope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    from rejudge_gold_claim_scope import (
        ENDPOINT, MODEL, PROTOCOL, evidence_fingerprint, input_key,
    )
    fact = {"fact_id": "f", "row_id": "r", "fact_index": 0, "claim": "甲胜利。",
            "cited_eids": ["E1"], "cited_texts": {"E1": "甲胜利。"}}
    prior = {
        "fact_id": "f", "row_id": "r", "fact_index": 0, "status": "ok",
        "full_evidence": True, "evidence_sha256": evidence_fingerprint(fact),
        "label": "supported_by_some", "keep_eids": ["E1"],
    }
    scope = {
        **prior, "protocol": PROTOCOL, "endpoint": ENDPOINT, "judge_model": MODEL,
        "max_tokens": 8192, "finish_reason": "stop", "input_key": input_key(fact, 8192),
        "claim_support": "fully_supported", "supporting_eids": ["E1"],
        "unsupported_parts": [], "reason": "整条成立", "label": "supported_by_union",
    }
    return [fact], [prior], [scope]


def test_exact_verified_view_preserves_original_and_restores_single_document_full():
    args = inputs()
    frozen = copy.deepcopy(args)
    rows, report = MODULE.merge(*args)
    assert rows[0]["label"] == "supported_by_union"
    assert report["scope_replacements"] == 1
    assert report["training_ready"] is False
    assert args == frozen


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "stale", "budget", "projection"])
def test_incomplete_or_unbound_scope_cannot_create_replacement_view(change):
    facts, prior, scope = inputs()
    if change == "missing":
        scope = []
    elif change == "duplicate":
        scope.append(copy.deepcopy(scope[0]))
    elif change == "extra":
        scope.append(scope[0] | {"fact_id": "unknown"})
    elif change == "stale":
        facts[0]["claim"] = "乙胜利。"
    elif change == "budget":
        scope[0]["max_tokens"] = 4096
    else:
        scope[0]["claim_support"] = "partially_supported"
        scope[0]["unsupported_parts"] = ["胜利"]
    with pytest.raises(ValueError):
        MODULE.merge(facts, prior, scope)


def test_secondary_review_quarantines_without_overwriting_teacher_label():
    facts, prior, scope = inputs()
    veto = {"protocol": "asa_claim_scope_review_quarantine_v1", "reviewer": "Codex",
            "results": [{k: scope[0][k] for k in ("fact_id", "input_key", "evidence_sha256")} | {"reason": "因果有疑问"}]}
    rows, report = MODULE.merge(facts, prior, scope, veto)
    assert rows[0]["label"] == "supported_by_union"
    assert rows[0]["training_eligible"] is False
    assert report["scope_review_quarantined"] == 1
    veto["results"][0]["input_key"] = "stale"
    with pytest.raises(ValueError, match="quarantine_content"):
        MODULE.merge(facts, prior, scope, veto)
