"""Seal completed old-source experiments and explicitly reject a tested patch."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts


def main():
    output = HERE / "checkpoint_v21_state.json"
    decisionpath = HERE / "prompt_priority_promotion_decision_v1.json"
    report = ROOT / "docs/CANONICAL_POINTER_AUDIT_AND_DEDUP_REJECTION_20260909_leU0LZ_v21.md"
    assert report.is_file() and not output.exists() and not decisionpath.exists()
    v20 = load(HERE / "checkpoint_v20_state.json")
    assert snapshot(SOURCE) == v20["runtime_source_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == v20["head"]
    prepared = load(HERE / "prompt_priority_prepared_v1.json")
    verified = load(HERE / "prompt_priority_candidate_verified_v1.json")
    assert verified["prepared_sha256"] == sha(HERE / "prompt_priority_prepared_v1.json")
    assert snapshot(SOURCE) == prepared["canonical_source_hashes"]
    assert verified["candidate_tests"]["counts"] == {
        "tests": 384, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(HERE / "prompt_priority_baseline_v1.xml") == {
        "tests": 22, "failures": 12, "errors": 0, "skipped": 0}
    assert all(sha(Path(p)) == h for p, h in prepared["verification_inputs_sha256"].items())
    assert not (HERE / "prompt_priority_preapply_v1.json").exists()
    assert not (HERE / "prompt_priority_applied_verified_v1.json").exists()
    assert not (ROOT / "release/ASA-ArknightStoryAgent/tests/test_prompt_priority_dedup.py").exists()
    postpaths = [
        HERE / "quote_pointer8b58_generation_postflight_v1.json",
        HERE / "quote_pointer8b58_blind_audits_postflight_v1.json",
        HERE / "susie_handoff4_postflight_v1.json",
    ]
    assert all(load(p)["protocol_verified"] for p in postpaths)
    for p in (postpaths[0], postpaths[2]):
        assert all(sha(Path(path)) == h for path, h in load(p)["artifacts_sha256"].items())
    audit = load(HERE / "quote_pointer8b58_blind_audits_v1.json")
    auditpost = load(postpaths[1])
    assert auditpost["audit_report_sha256"] == sha(HERE / "quote_pointer8b58_blind_audits_v1.json")
    assert auditpost["new_statuses"] == {"ok": 24, "error": 4}
    assert audit["summary"] == auditpost["summary"]
    assert audit["summary"]["quote_pointer8b_v1"]["semantics"] == {
        "pipeline_error": 24, "audit_error": 4, "no_answer_action": 20,
        "supported_complete": 5, "unsupported": 4, "supported_partial": 1}
    generationpost = load(postpaths[0])
    assert generationpost["states"] == 58 and generationpost["finish_reasons"] == {"stop": 58}
    four = load(postpaths[2])
    assert four["states"] == 4 and four["independent_questions"] == 1
    assert four["all_inputs_untruncated"] and four["all_output_budgets_unreduced"]
    baseline = load(HERE / "prompt_priority_handoff_baseline_v1.json")
    candidate = load(HERE / "prompt_priority_handoff_candidate_v1.json")
    assert len(baseline["conditions"]) == len(candidate["conditions"]) == 6
    regressions = [
        {"scoring_condition": a["scoring_condition"], "order_only_reserve6": a["order_only_reserve6"],
         "before_visible": a["target_phrase_visible"], "after_visible": b["target_phrase_visible"],
         "before_prompt_sha256": a["runtime_prompt_sha256"], "after_prompt_sha256": b["runtime_prompt_sha256"]}
        for a, b in zip(baseline["conditions"], candidate["conditions"])
        if a["target_phrase_visible"] and not b["target_phrase_visible"]]
    assert len(regressions) == 1 and regressions[0]["scoring_condition"] == "seed_retained_complete3072"
    assert regressions[0]["order_only_reserve6"] is True
    policies = load(HERE / "prompt_dedup_policies_v1.json")
    assert len(policies["conditions"]) == 18 and policies["canonical_source_unchanged"]
    seed_policies = [r for r in policies["conditions"] if r["scoring_condition"] == "seed_retained_complete3072"]
    assert len(seed_policies) == 6 and all(not r["target_visible"] for r in seed_policies)
    save(decisionpath, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "decision": "reject_promotion",
        "reason": "Target passage previously visible under a real development suffix condition is lost; unit tests do not establish answer quality.",
        "candidate_verified_sha256": sha(HERE / "prompt_priority_candidate_verified_v1.json"),
        "baseline_replay_sha256": sha(HERE / "prompt_priority_handoff_baseline_v1.json"),
        "candidate_replay_sha256": sha(HERE / "prompt_priority_handoff_candidate_v1.json"),
        "observed_regressions": regressions, "canonical_source_unchanged": True,
        "candidate_test_counts": verified["candidate_tests"]["counts"],
        "applied_to241": False, "synced_to248": False, "training_ready": False,
        "new_certified_correct_answers": 0,
    })
    names = [
        "checkpoint_v20_state.json", "quote_pointer8b58_generation_postflight_v1.json",
        "quote_pointer8b58_blind_audits_v1.json", "quote_pointer8b58_blind_audits_postflight_v1.json",
        "quote_pointer8b58_failure_diagnosis_v1.json", "quote_pointer8b58_selected_source_review_v1.json",
        "susie_seed_retention_v1.json", "explicit_speaker_reserve_v1.json",
        "susie_prompt_handoff_v1.json", "susie_prompt_dedup_trace_v1.json",
        "susie_handoff4_validation_v1.json", "susie_handoff4_postflight_v1.json",
        "prompt_priority_prepared_v1.json", "prompt_priority_baseline_v1.json",
        "prompt_priority_candidate_verified_v1.json", "prompt_priority_handoff_baseline_v1.json",
        "prompt_priority_handoff_candidate_v1.json", "prompt_priority_promotion_decision_v1.json",
        "prompt_dedup_policies_preflight_v1.json", "prompt_dedup_policies_v1.json",
    ]
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "head": head,
        "report": str(report), "report_sha256": sha(report), "goal_complete": False,
        "runtime_source_sha256": snapshot(SOURCE), "runtime_changed_since_v20": False,
        "canonical_tests": v20["canonical_tests"], "mirror_subset_tests_from_v20": v20["mirror_subset_tests"],
        "isolated_candidate_tests": verified["candidate_tests"]["counts"],
        "candidate_promotion": "rejected", "candidate_regressions": regressions,
        "pointer58_generation_complete": True, "pointer58_blind_audit_complete": True,
        "pointer_audit_summary": audit["summary"], "new_susie_generations": 4,
        "new_susie_independent_questions": 1, "susie_generation_summary": four["summary"],
        "new_chain_scores_since_v20": 15, "dedup_policy_replay_states": 18,
        "artifacts_sha256": {str(HERE / name): sha(HERE / name) for name in names},
        "checkpoint_script_sha256": sha(Path(__file__)),
        "training_ready": False, "old_labels_changed": False, "old_artifacts_overwritten": False,
        "models_or_config_changed": False, "services_restarted": False, "new_training_runs": 0,
        "new_full_rag_recall": None, "new_certified_answer_accuracy": None,
    })
    print(json.dumps({"checkpoint": str(output), "candidate_promotion": "rejected",
                      "canonical_source_changed": False, "goal_complete": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
