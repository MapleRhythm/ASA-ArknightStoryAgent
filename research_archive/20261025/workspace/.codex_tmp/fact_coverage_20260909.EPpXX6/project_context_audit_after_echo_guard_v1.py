"""Project old frozen opinions onto unchanged delivered answers; no new calls."""
from datetime import datetime, timezone
import json

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import save
from audit_fact_coverage_pairs_v1 import prepare, summarize, audit_messages


def main():
    destination = HERE / "without_reasoning_hints58_after_echo_audit_projection_v1.json"
    assert not destination.exists()
    path = HERE / "without_reasoning_hints58_blind_audits_v1.json"
    postpath = HERE / "without_reasoning_hints58_blind_audits_postflight_v1.json"
    report, post = load(path), load(postpath)
    assert post["protocol_verified"] and post["audit_report_sha256"] == sha(path)
    applied = load(HERE / "question_echo_applied_verified_v1.json")
    assert applied["context_replay_changed_actions"] == 1
    originals = [r for r in load(HERE / "question_echo_paired116_replay_v1.json")["results"]
                 if r["renderer"] == "original_prompt_v1"]
    afterpath = HERE / "question_echo_no_hints58_replay_v1.json"
    cases, states = prepare({"results": originals + load(afterpath)["results"]})
    old_cases = {r["id"]: r for r in report["cases"]}
    old_results = {r["id"]: r for r in report["results"]}
    assert all(c["id"] in old_cases and audit_messages(c) == audit_messages(old_cases[c["id"]]) for c in cases)
    old_states = {r["id"]: r for r in report["states"]}
    changed_states = [s for s in states if s != old_states[s["id"]]]
    assert len(changed_states) == 1
    assert changed_states[0]["id"].startswith("relabel-04e53609ed1e65775ec14112::")
    assert changed_states[0]["state"] == "no_answer_action" and changed_states[0]["action"] == "retrieve_more"
    assert old_states[changed_states[0]["id"]]["state"] == "direct_answer"
    labeled, summary = summarize(states, [old_results[c["id"]] for c in cases])
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "new_provider_calls": 0,
        "old_audit_report_sha256": sha(path), "old_postflight_sha256": sha(postpath),
        "after_replay_sha256": sha(afterpath), "changed_states": changed_states,
        "all_surviving_direct_judge_inputs_identical": True,
        "summary": summary, "labeled_states": labeled,
        "limits": [
            "Reused automatic opinions, not human gold or a new generator experiment.",
            "Old31-direct audit is preserved; post-guard30-direct projection is a separate artifact.",
            "The removed question-echo answer becomes no-answer, never a newly correct answer.",
        ],
    })
    print(json.dumps({"new_provider_calls": 0, "changed_states": 1, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
