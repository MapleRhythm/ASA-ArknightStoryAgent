import importlib.util
import json
import sys
import copy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_relevance_clean_exx_data.py"
SPEC = importlib.util.spec_from_file_location("build_relevance_clean_exx_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cleaned_payload_drops_supported_irrelevant_facts() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "甲完成任务。", "evidence_ids": ["E1"]},
            {"fact": "乙在白天离开。", "evidence_ids": ["E2"]},
        ],
    }
    judgement = {
        "facts": [
            {"support": "entailed", "question_relevance": "direct", "citation_complete": True},
            {"support": "entailed", "question_relevance": "irrelevant", "citation_complete": True},
        ],
        "set_support": "complete",
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    # The full audit validator also requires the fields omitted from this
    # focused unit fixture; the subset operation itself is tested separately.
    assert MODULE.propose_repair(payload, judgement) == {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }


def test_cleaned_payload_excludes_unsupported_only_answer() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }
    assert MODULE.cleaned_payload(
        payload,
        {
            "facts": [{"support": "contradicted", "question_relevance": "direct"}],
            "set_support": "contradicted",
            "context_sufficiency": "insufficient",
            "action_appropriateness": "inappropriate",
            "relations": [],
            "missing_requirements": [],
            "critical_unsupported_claims": 1,
            "irrelevant_claims": 0,
            "protocol": "asa_exx_set_grounding_audit_v3",
        },
    ) is None


def test_heldout_overlap_checks_source_id_and_normalized_question() -> None:
    heldout = [
        {
            "id": "heldout-1",
            "meta": json.dumps({"source_id": "source-1"}),
            "conversations": [{"value": "问题：谁完成了任务？"}],
        }
    ]
    same_source = {
        "id": "other-id",
        "meta": json.dumps({"source_id": "source-1"}),
        "conversations": [{"value": "问题：完全不同的问法"}],
    }
    same_question = {
        "id": "other-source",
        "meta": "{}",
        "conversations": [{"value": "问题：谁完成了任务？"}],
    }
    assert MODULE.heldout_overlap(same_source, heldout)
    assert MODULE.heldout_overlap(same_question, heldout)


def bound_audit():
    source = {
        "id": "source", "system": "system",
        "conversations": [{"value": "question: 谁完成任务？\nround: 1/2\nevidence:\n[E1]\n甲完成任务。\noutput_schema: grounded_action_exx_v1\n"}],
        "raw_output": '{"next_action":"answer_directly","supported_facts":[{"fact":"甲完成任务。","evidence_ids":["E1"]}]}',
    }
    summary = {"protocol": MODULE.PROTOCOL, "endpoint": "https://judge.invalid",
               "judge_model": "test-model", "max_tokens": 4096}
    result = {
        "id": source["id"], "status": "ok", "finish_reason": "stop", "full_evidence": True,
        "audited_input_sha256": MODULE.audited_input_sha256(source),
        "input_key": MODULE.input_key(source, summary["endpoint"], summary["judge_model"], summary["max_tokens"]),
    }
    return source, result, summary


def test_changed_claim_or_evidence_cannot_reuse_same_id_audit():
    source, result, summary = bound_audit()
    assert MODULE.audit_binding_issue(source, result, summary) is None
    for field in ("raw_output", "system", "evidence"):
        changed = copy.deepcopy(source)
        if field == "evidence":
            changed["conversations"][0]["value"] = changed["conversations"][0]["value"].replace("甲完成任务。", "甲没有完成任务。")
        else:
            changed[field] += "changed"
        assert MODULE.audit_binding_issue(changed, result, summary) == "audited_content_mismatch"


@pytest.mark.parametrize("change", ["legacy", "contract", "key", "finish", "truncated"])
def test_missing_provenance_and_truncation_fail_closed(change):
    source, result, summary = bound_audit()
    if change == "legacy":
        result.pop("audited_input_sha256")
    elif change == "contract":
        summary.pop("max_tokens")
    elif change == "key":
        result["input_key"] = "stale"
    elif change == "finish":
        result["finish_reason"] = "length"
    else:
        source["hit_max_new_tokens"] = True
    assert MODULE.audit_binding_issue(source, result, summary) is not None


@pytest.mark.parametrize("extra", [
    {"diagnostic": {"evaluation_only": True}},
    {"meta": '{"source_split":"validation"}'},
    {"meta": {"training_eligible": False}},
    {"repair_provenance": {"requires_reaudit": True}},
    {"split": "test"},
    {"source_split": "train", "split": "test"},
])
def test_cli_train_flag_cannot_override_explicit_nontraining_source(extra):
    assert MODULE.source_training_issue({"id": "source", **extra})
    assert MODULE.source_training_issue({"meta": {"source_split": "train", "training_eligible": True}}) is None


def test_training_requires_content_matching_approved_lineage():
    source, _, _ = bound_audit()
    partition = {"id": source["id"], "split": "train", "partition_candidate": True,
                 "row_sha256": MODULE.content_sha256(source), "quarantine_reasons": []}
    assert MODULE.lineage_issue(source, {source["id"]: partition}, {}) is None
    assert MODULE.lineage_issue(source, {}, {}) == "lineage_partition_not_approved"
    source["system"] += "changed"
    assert MODULE.lineage_issue(source, {source["id"]: partition}, {}) == "lineage_source_content_mismatch"


def test_staged_bridge_needs_both_original_lineage_and_exact_candidate():
    source, _, _ = bound_audit()
    partition = {"id": source["id"], "split": "train", "partition_candidate": True,
                 "row_sha256": "original", "component_id": "group", "quarantine_reasons": []}
    source["candidate_provenance"] = {"source_row_sha256": "original", "component_id": "group"}
    staged = {source["id"]: {"candidate_output_sha256": MODULE.hashlib.sha256(source["raw_output"].encode()).hexdigest()}}
    assert MODULE.lineage_issue(source, {source["id"]: partition}, staged) is None
    source["raw_output"] += "changed"
    assert MODULE.lineage_issue(source, {source["id"]: partition}, staged) == "lineage_source_content_mismatch"


def test_lineage_must_cover_the_same_frozen_heldout(tmp_path):
    heldout = tmp_path / "heldout.json"
    heldout.write_text('[{"id":"test1"}]', encoding="utf-8")
    report = {"inputs": {"val": {"sha256": MODULE.sha256_file(heldout)}}}
    assert MODULE.lineage_heldout_issue(report, [heldout]) is None
    heldout.write_text('[{"id":"test2"}]', encoding="utf-8")
    assert MODULE.lineage_heldout_issue(report, [heldout]) == "lineage_heldout_content_mismatch"
    assert MODULE.lineage_heldout_issue({}, [heldout]) == "missing_lineage_heldout_provenance"


def test_training_cli_exports_only_freshly_reviewed_complete_answer(tmp_path, monkeypatch):
    source, judged, summary = bound_audit()
    source["task_type"] = "grounded_action_generation"
    judged["judgement"] = {
        "protocol": MODULE.PROTOCOL,
        "facts": [{"fact_index": 0, "support": "entailed", "question_relevance": "direct",
                   "checked_evidence_ids": ["E1"], "citation_complete": True}],
        "relations": [], "set_support": "complete", "missing_requirements": [],
        "critical_unsupported_claims": 0, "irrelevant_claims": 0,
        "context_sufficiency": "sufficient", "action_appropriateness": "appropriate",
    }
    files = {name: tmp_path / f"{name}.json" for name in ("predictions", "audit", "heldout", "lineage")}
    files["predictions"].write_text(json.dumps([source]), encoding="utf-8")
    files["audit"].write_text(json.dumps({"results": [judged], "summary": summary}), encoding="utf-8")
    files["heldout"].write_text(json.dumps([{
        "id": "heldout", "question": "完全不同的地点在哪里？",
        "conversations": [{"from": "human", "value": "question: 完全不同的地点在哪里？"}],
    }]), encoding="utf-8")
    lineage = {
        "protocol": "asa_training_lineage_audit_v1",
        "inputs": {"val": {"sha256": MODULE.sha256_file(files["heldout"])}},
        "records": [{"id": source["id"], "split": "train", "partition_candidate": True,
                     "row_sha256": MODULE.content_sha256(source), "quarantine_reasons": []}],
    }
    files["lineage"].write_text(json.dumps(lineage), encoding="utf-8")
    argv = [
        str(SCRIPT), "--predictions", str(files["predictions"]), "--audit", str(files["audit"]),
        "--heldout-json", str(files["heldout"]), "--lineage-audit", str(files["lineage"]),
        "--source-split", "train", "--out-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert MODULE.main() == 0
    accepted = json.loads((tmp_path / "out/sft.json").read_text())
    assert len(accepted) == 1
    assert json.loads(accepted[0]["meta"])["source_question"] == "谁完成任务？"
    partial = copy.deepcopy(judged)
    partial["judgement"]["set_support"] = "partial"
    partial["judgement"]["missing_requirements"] = ["尚有子问未答"]
    files["audit"].write_text(json.dumps({"results": [partial], "summary": summary}), encoding="utf-8")
    argv[-1] = str(tmp_path / "partial")
    assert MODULE.main() == 0
    repairs = json.loads((tmp_path / "partial/repair_candidates.json").read_text())
    assert len(repairs) == 1
    assert repairs[0]["conversations"][-1]["value"] == repairs[0]["raw_output"]
    assert repairs[0]["repair_provenance"]["requires_reaudit"] is True
    assert json.loads((tmp_path / "partial/sft.json").read_text()) == []
    files["audit"].write_text(json.dumps({"results": [judged], "summary": summary}), encoding="utf-8")
    # Even a valid but stale judge verdict cannot release a changed answer.
    source["raw_output"] = source["raw_output"].replace("甲完成", "乙完成")
    files["predictions"].write_text(json.dumps([source]), encoding="utf-8")
    lineage["records"][0]["row_sha256"] = MODULE.content_sha256(source)
    files["lineage"].write_text(json.dumps(lineage), encoding="utf-8")
    argv[-1] = str(tmp_path / "stale")
    assert MODULE.main() == 0
    assert json.loads((tmp_path / "stale/sft.json").read_text()) == []
    assert json.loads((tmp_path / "stale/report.json").read_text())["counts"]["excluded_audited_content_mismatch"] == 1


def test_export_preserves_story_provenance_for_later_lineage_audits():
    from audit_training_lineage import signals

    source, _, _ = bound_audit()
    source["meta"] = {"evidence_doc_ids": ["activities/story_a/scene#chunk1"]}
    source["diagnostic"] = {"evidence_id_to_corpus_id": {"E1": "obt/main/level_main_07-11#chunk2"}}
    record = MODULE.make_record(
        source, MODULE.parse_payload(source), tag=True, suffix="chosen",
        source_kind="relevance_verified", source_split="train", training_eligible=True,
    )
    assert signals(record, {})["story_footprints"] == {"activities/story_a", "obt/main/chapter_07"}
    assert json.loads(record["meta"])["audited_source_sha256"] == MODULE.content_sha256(source)


def test_secondary_review_can_only_veto_content_matching_teacher_verdict():
    source, _, _ = bound_audit()
    item = {"id": source["id"], "decision": "accept", "reason": "原文可核对",
            "audited_input_sha256": MODULE.audited_input_sha256(source)}
    assert MODULE.review_item_issue(source, {source["id"]: item}) is None
    assert MODULE.review_item_issue(source, {}) == "missing_secondary_review"
    assert MODULE.review_item_issue(source, {source["id"]: item | {"decision": "quarantine"}}) == "secondary_review_quarantine"
    source["raw_output"] += "changed"
    assert MODULE.review_item_issue(source, {source["id"]: item}) == "secondary_review_content_mismatch"


def test_prior_review_never_promotes_accepts_and_keeps_changed_rejections():
    source, _, _ = bound_audit()
    item = {"id": source["id"], "decision": "accept", "reason": "Reviewed",
            "audited_input_sha256": MODULE.audited_input_sha256(source)}
    assert MODULE.prior_review_veto(source, [{source["id"]: item}]) == []
    item["decision"] = "quarantine"
    veto = MODULE.prior_review_veto(source, [{source["id"]: item}])
    assert len(veto) == 1 and veto[0]["same_audited_content"] is True
    source["raw_output"] += "changed"
    veto = MODULE.prior_review_veto(source, [{source["id"]: item}])
    assert len(veto) == 1 and veto[0]["same_audited_content"] is False
    source["meta"] = {"source_id": source["id"]}
    source["id"] = "derived"
    assert len(MODULE.prior_review_veto(source, [{"source": item}])) == 1


@pytest.mark.parametrize("change", [None, "predictions", "audit", "item_content", "unknown_id", "duplicate"])
def test_prior_review_requires_its_original_frozen_inputs(tmp_path, change):
    source, _, _ = bound_audit()
    paths = [tmp_path / name for name in ("review.json", "predictions.json", "audit.json")]
    review_path, predictions_path, audit_path = paths
    predictions_path.write_text(json.dumps([source]), encoding="utf-8")
    audit_path.write_text("{}", encoding="utf-8")
    item = {"id": source["id"], "decision": "quarantine", "reason": "Missing requirement",
            "audited_input_sha256": MODULE.audited_input_sha256(source)}
    review = {
        "protocol": "asa_set_candidate_review_v1", "reviewer": "test",
        "predictions_sha256": MODULE.sha256_file(predictions_path),
        "audit_sha256": MODULE.sha256_file(audit_path), "results": [item],
    }
    if change in {"predictions", "audit"}:
        review[f"{change}_sha256"] = "stale"
    elif change == "item_content":
        item["audited_input_sha256"] = "stale"
    elif change == "unknown_id":
        item["id"] = "not-reviewed"
    elif change == "duplicate":
        review["results"].append(copy.deepcopy(item))
    review_path.write_text(json.dumps(review), encoding="utf-8")
    if change:
        with pytest.raises(ValueError):
            MODULE.load_bound_review(*paths)
    else:
        assert MODULE.load_bound_review(*paths) == {source["id"]: item}
