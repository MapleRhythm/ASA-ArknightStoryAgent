import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stage_full_evidence_sft_candidates.py"
SPEC = importlib.util.spec_from_file_location("stage_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    source = {
        "id": "r", "task_type": "grounded_action_generation", "system": "system",
        "conversations": [
            {"from": "human", "value": "question: 甲做了什么？\nround: 1/2\nevidence:\n[E1]\n甲完成任务。\noutput_schema: grounded_action_exx_v1\n"},
            {"from": "gpt", "value": '{"next_action":"abstain","reason":"旧判定"}'},
        ],
    }
    fact = {"fact_id": "f", "row_id": "r", "fact_index": 0, "query": "甲做了什么？",
            "claim": "甲完成任务。", "cited_eids": ["E1"], "cited_texts": {"E1": "甲完成任务。"}}
    judgement = {**fact, "chunk_chars": 100, "merged": {"status": "ok", "verdict": "supported"},
                 "individual": {"E1": {"status": "ok", "verdict": "supported"}}}
    lineage = {"protocol": "asa_training_lineage_audit_v1",
               "records": [{"id": "r", "split": "train", "partition_candidate": True,
                           "row_sha256": MODULE.content_sha256(source), "component_id": "family"}]}
    return [source], [fact], [judgement], [], [], lineage


def test_restored_claim_is_staged_not_auto_training_or_action_gold():
    args = inputs()
    original = copy.deepcopy(args)
    rows, report = MODULE.stage(*args)
    assert len(rows) == 1 and report["requires_fresh_set_audit"]
    assert not report["training_ready"]
    assert rows[0]["candidate_provenance"]["set_audit_pending"] is True
    assert json.loads(rows[0]["raw_output"])["supported_facts"][0]["fact"] == "甲完成任务。"
    assert args == original


def test_unknown_partition_or_changed_row_rejected():
    args = list(inputs())
    args[-1]["records"][0]["partition_candidate"] = False
    assert MODULE.stage(*args)[0] == []
    args[-1]["records"][0]["partition_candidate"] = True
    args[0][0]["system"] += "changed"
    with pytest.raises(ValueError, match="lineage_source_content"):
        MODULE.stage(*args)


def test_partial_fact_not_relabelled_to_full_or_automatic_retrieve():
    from build_binding_verifier_pairs import evidence_fingerprint

    args = list(inputs())
    args[3] = [{
        "fact_id": "f", "row_id": "r", "fact_index": 0, "status": "ok",
        "full_evidence": True, "label": "supported_by_some", "keep_eids": ["E1"],
        "evidence_sha256": evidence_fingerprint(args[1][0]),
    }]
    rows, report = MODULE.stage(*args)
    assert rows == []
    assert report["no_nonanswer_action_inferred_from_fact_deletion"]


def test_visible_evidence_must_match_full_fact_manifest():
    args = list(inputs())
    args[1][0]["cited_texts"]["E1"] += "不可见后缀"
    with pytest.raises(ValueError, match="visible_evidence_mismatch"):
        MODULE.stage(*args)


def test_new_scope_projection_cannot_turn_partial_into_union():
    from build_binding_verifier_pairs import evidence_fingerprint

    args = list(inputs())
    args[4] = [{
        "fact_id": "f", "row_id": "r", "fact_index": 0, "status": "ok",
        "full_evidence": True, "label": "supported_by_union", "keep_eids": ["E1"],
        "evidence_sha256": evidence_fingerprint(args[1][0]), "protocol": MODULE.SCOPE_PROTOCOL,
        "finish_reason": "stop", "claim_support": "partially_supported",
        "supporting_eids": ["E1"], "unsupported_parts": ["完成"], "reason": "未支持",
    }]
    with pytest.raises(ValueError, match="invalid_scope_sidecar"):
        MODULE.stage(*args)


def test_scope_requires_actual_judge_contract_not_just_evidence_sha():
    from rejudge_gold_claim_scope import ENDPOINT, MODEL, input_key, evidence_fingerprint

    args = list(inputs())
    fact = args[1][0]
    scope = {
        "fact_id": "f", "row_id": "r", "fact_index": 0, "status": "ok",
        "protocol": MODULE.SCOPE_PROTOCOL, "endpoint": ENDPOINT, "judge_model": MODEL,
        "finish_reason": "stop", "max_tokens": 8192, "input_key": input_key(fact, 8192),
        "full_evidence": True, "evidence_sha256": evidence_fingerprint(fact),
        "claim_support": "fully_supported", "supporting_eids": ["E1"],
        "unsupported_parts": [], "reason": "整条有支持",
        "label": "supported_by_union", "keep_eids": ["E1"],
    }
    args[4] = [scope]
    assert len(MODULE.stage(*args)[0]) == 1
    for field, value in (("input_key", "stale"), ("max_tokens", 4096), ("judge_model", "other")):
        args[4] = [scope | {field: value}]
        with pytest.raises(ValueError, match="invalid_scope_sidecar"):
            MODULE.stage(*args)


def test_unknown_lineage_protocol_and_conflicting_quarantine_fail_closed():
    args = list(inputs())
    args[-1]["protocol"] = "unknown"
    with pytest.raises(ValueError, match="lineage_protocol"):
        MODULE.stage(*args)
    args[-1]["protocol"] = "asa_training_lineage_audit_v1"
    args[-1]["records"][0]["quarantine_reasons"] = ["heldout_source"]
    assert MODULE.stage(*args)[0] == []


def test_review_quarantine_cannot_fall_back_to_old_positive():
    args = list(inputs())
    args[3] = [{"fact_id": "f", "training_eligible": False}]
    rows, report = MODULE.stage(*args)
    assert rows == []
    assert report["counts"]["fact_status:review_quarantine"] == 1
