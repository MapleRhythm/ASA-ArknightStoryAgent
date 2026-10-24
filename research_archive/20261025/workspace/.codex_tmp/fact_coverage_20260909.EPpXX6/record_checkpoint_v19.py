"""Bind completed frozen experiments and the separately applied narrow guard."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts


def main():
    report = ROOT / "docs/CANONICAL_NATIVE8B_CONTEXT_AND_QUESTION_ECHO_20260909_leU0LZ_v19.md"
    destination = HERE / "checkpoint_v19_state.json"
    assert report.is_file() and not destination.exists()
    native = load(HERE / "qwen3_8b_blind_audits_postflight_v1.json")
    context = load(HERE / "without_reasoning_hints58_blind_audits_postflight_v1.json")
    projection = load(HERE / "without_reasoning_hints58_after_echo_audit_projection_v1.json")
    assert native["protocol_verified"] and context["protocol_verified"]
    assert native["new_calls"] == 18 and native["new_statuses"] == {"ok": 15, "error": 3}
    assert context["new_calls"] == 60 and context["new_statuses"] == {"ok": 57, "error": 3}
    assert projection["new_provider_calls"] == 0 and len(projection["changed_states"]) == 1
    for prefix in ("qwen3_8b", "without_reasoning_hints58"):
        gen = load(HERE / f"{prefix}_generation_postflight_v1.json")
        assert gen["protocol_verified"] and gen["states"] == 58
        assert gen["finish_reasons"] == {"stop": 58}
        assert all(sha(Path(p)) == h for p, h in gen["artifacts_sha256"].items())
        post = native if prefix == "qwen3_8b" else context
        assert post["audit_report_sha256"] == sha(HERE / f"{prefix}_blind_audits_v1.json")
    audit = load(HERE / "without_reasoning_hints58_blind_audits_v1.json")
    new_errors = [r for r in audit["results"]
                  if r["id"] in audit["new_request_ids"] and r["status"] != "ok"]
    assert len(new_errors) == 3
    limited = [r for r in new_errors if r.get("finish_reason") == "length"]
    assert len(limited) == 1
    assert limited[0]["response_model"] == "glm-5.3"
    assert limited[0]["usage"]["completion_tokens"] == 16384
    assert limited[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 15997
    assert limited[0]["raw_response"]  # Unlike the previous three empty-body length failures.
    try:
        json.loads(limited[0]["raw_response"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("expected preserved incomplete JSON body")
    applied = load(HERE / "question_echo_applied_verified_v1.json")
    assert applied["canonical_tests"] == {"tests": 348, "failures": 0, "errors": 0, "skipped": 0}
    assert all(sha(ROOT / p) == h for p, h in applied["applied_sha256"].items())
    assert snapshot(SOURCE) == load(HERE / "question_echo_preapply_v1.json")["candidate_source_hashes"]
    mirror = load(HERE / "mirror248_question_echo_delta/mirror_applied_verified_v1.json")
    assert mirror["full_runtime_python_matches241"] and mirror["four_dataset_scripts_match241"]
    assert counts(HERE / "mirror248_question_echo_delta/mirror_tests192_v1.xml") == {
        "tests": 192, "failures": 0, "errors": 0, "skipped": 0}
    actions = Counter(r.get("grounded", {}).get("next_action", "pipeline_error")
                      for r in load(HERE / "question_echo_no_hints58_replay_v1.json")["results"])
    assert actions == {"answer_directly": 30, "retrieve_more": 21, "abstain": 6, "pipeline_error": 1}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == "0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b"
    artifacts = [
        "checkpoint_v18_state.json", "partial_source_review_v2.json", "partial_source_review_v3.json",
        "native8b_all9_shown_source_review_v1.json", "native8b_rejection_taxonomy_v1.json",
        "without_reasoning_hints58_partial_source_review_v1.json",
        "without_reasoning_hints58_after_echo_audit_projection_v1.json",
        "question_echo_applied_verified_v1.json", "question_echo_candidate_verified_v1.json",
        "mirror248_question_echo_delta/mirror_applied_verified_v1.json",
        "artifact_hash_and_context_tests_v1.xml",
    ]
    artifacts += [f"{prefix}_{suffix}_postflight_v1.json"
                  for prefix in ("qwen3_8b", "without_reasoning_hints58")
                  for suffix in ("generation", "blind_audits")]
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": False, "training_ready": False, "head": head,
        "report": str(report), "report_sha256": sha(report),
        "native8b_summary": native["summary"], "context_pre_guard_summary": context["summary"],
        "context_post_guard_summary": projection["summary"],
        "context_post_guard_actions": dict(actions),
        "new_audits": {"native8b": native["new_statuses"], "context": context["new_statuses"]},
        "context_new_failures": [{k: r.get(k) for k in
                                 ("id", "error", "response_model", "finish_reason", "usage")}
                                for r in new_errors],
        "runtime_source_sha256": snapshot(SOURCE),
        "artifacts_sha256": {str(HERE / p): sha(HERE / p) for p in artifacts},
        "checkpoint_script_sha256": sha(Path(__file__)),
        "default_model_or_prompt_changed": False, "services_restarted": False,
        "old_outputs_or_labels_overwritten": False, "new_training_runs": 0,
        "new_certified_answer_accuracy": None, "new_retrieval_recall": None,
        "limits": ["All58 are development questions, not holdout.",
                   "Postflight source identity was checked before the later guard; saved reports preserved.",
                   "348 canonical regressions and192 mirror subset are not added together.",
                   "Removing one question echo is not adding one correct answer."],
    })
    print(json.dumps({"report": str(report), "checkpoint": str(destination),
                      "goal_complete": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
