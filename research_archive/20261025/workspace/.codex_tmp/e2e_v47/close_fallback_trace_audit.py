"""Close the read-only trace-completeness diagnosis, not an answer fix."""
from datetime import datetime, timezone
from pathlib import Path
import run_fallback_trace_audit as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v47.json").exists()
    pre, result = shared.load(HERE / "preflight.json"), shared.load(HERE / "audit_v1.json")
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert result["actual_state_replays"] == len(result["rows"]) == 14
    assert result["summary_not_accuracy"] == {
        "fallback_states": 9, "fallback_states_missing_primary_raw": 9, "no_fallback_primary_raw_retained": 5,
    }
    for row in result["rows"]:
        assert row["canonical_replay_exact"]
        if row["canonical_fallback_called"]:
            assert not row["primary_raw_paths_in_final_round_trace"]
            assert row["fallback_raw_paths_in_final_round_trace"]
            assert row["v45_out_of_band_requests_preserved"] == 2
        else:
            assert row["primary_raw_paths_in_final_round_trace"]
    assert len(result["synthetic_failure_protocol_cases"]) == 2
    for case in result["synthetic_failure_protocol_cases"]:
        assert case["synthetic_only"] and case["actual_model_calls"] == 0
        assert case["fallback_exception_retained_in_trace"] is False
        assert "synthetic_final_direct_fallback_error_v47" in case["captured_stdout"]
    paths = [HERE / name for name in (
        "preflight.json", "prepare.log", "run_v1.log", "audit_v1.json",
        "run_fallback_trace_audit.py", "close_fallback_trace_audit.py",
    )]
    shared.save(HERE / "checkpoint_v47.json", {
        "version": "canonical_fallback_trace_completeness_closed_v47",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_state_replays": 14, "trace_summary_not_accuracy": result["summary_not_accuracy"],
        "synthetic_protocol_cases": 2, "actual_fallback_generation_failures_observed": 0,
        "v45_external_request_artifacts_intact": True,
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "production_changed": False, "training_ready": False,
        "decision": "prepare_isolated_observability_candidate_only",
        "limitations": [
            "The nine actual missing primary raws were preserved by the v45 experiment's separate ledger, not lost from those artifacts.",
            "Synthetic exceptions demonstrate a stdout-only fallback-error path, not actual failures in the23 successful v45 calls.",
            "Missing a complete primary attempt record does not mean every warning string disappears; overlapping warnings are separately measured.",
            "Trace preservation can improve auditability, not semantic correctness by itself.",
            "No production code, fallback policy, answer, validator, citation binding or previous artifact has been changed.",
        ],
    })
    print({"state_replays": 14, "actual_fallback_primary_raw_missing": 9,
           "synthetic_protocol_cases": 2, "production_changed": False}, flush=True)


if __name__ == "__main__":
    main()
