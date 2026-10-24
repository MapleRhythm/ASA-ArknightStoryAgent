from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared


def main():
    pre = shared.load(HERE / "preflight.json")
    report = shared.load(HERE / "audit_v1.json")
    assert report["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert len(report["actual_replays"]) == 32
    stats = report["actual_replay_summary_not_accuracy"]
    for kind in ("empty_objects", "original_objects", "visible_objects"):
        assert stats[f"{kind}:action:answer_directly"] == 20
        assert stats[f"{kind}:action:abstain"] == 12
        assert stats[f"{kind}:legacy_token_entries"] == 4
    for kind in ("original_objects", "visible_objects"):
        assert stats[f"{kind}:changed_action_vs_empty"] == 0
        assert stats[f"{kind}:changed_display_vs_empty"] == 0
    synthetic = report["synthetic_contract_probe"]
    assert synthetic["synthetic_only_not_story_or_model_output"]
    assert synthetic["required_anchors"] == {"augmented": "甲乙代号", "projected": ""}
    exact, reconstructed = synthetic["cases"]
    assert exact["label"] == "exact_fact"
    assert not exact["quote_issues"] and not exact["quote_warnings"]
    assert not reconstructed["quote_issues"] and reconstructed["quote_warnings"]
    for mode in ("quote", "grounded", "strict"):
        rejected = exact["conditions"][f"{mode}:augmented:complete"]
        accepted = reconstructed["conditions"][f"{mode}:augmented:complete"]
        assert rejected["result"]["next_action"] == "abstain"
        assert len(rejected["legacy_token_calls"]) == 1
        assert accepted["result"]["next_action"] == "answer_directly"
        assert accepted["legacy_token_calls"] == []
        assert accepted["result"]["answer"] == exact["conclusion"]["answer"]
        assert "另外补充无依据细节" not in accepted["result"]["answer"]
        assert exact["conditions"][f"{mode}:augmented:empty"]["result"]["next_action"] == "answer_directly"
        assert exact["conditions"][f"{mode}:projected:complete"]["result"]["next_action"] == "answer_directly"
    assert exact["conditions"]["evidence_id:augmented:complete"]["legacy_token_calls"] == []
    files = [HERE / name for name in (
        "preflight.json", "audit_v1.json", "audit_quote_fallthrough.py", "close_fallthrough.py",
    )]
    shared.save(HERE / "checkpoint_v38.json", {
        "version": "quote_legacy_fallthrough_closed_v38",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "actual_outputs": 32, "unique_questions": 8, "actual_output_action_or_display_changes": 0,
        "synthetic_branch_inconsistency_verified": True,
        "synthetic_extra_unsubstantiated_text_not_displayed": True,
        "promotion_decision": "no_validation_policy_change_implemented",
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": [
            "Actual 32-output replay is not an independent accuracy evaluation.",
            "Invented contract fixtures prove path inconsistency, not a measured production error rate.",
            "Do not remove validation gates or promote entity deletion by acceptance count.",
        ],
    })
    print({"actual_output_changes": 0, "synthetic_branch_inconsistency_verified": True})


if __name__ == "__main__":
    main()
