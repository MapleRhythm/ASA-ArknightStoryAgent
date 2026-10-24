from datetime import datetime, timezone
from pathlib import Path
import run_trace_retention_candidate as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v48.json").exists()
    pre, replay, baseline = [
        shared.load(HERE / name) for name in (
            "preflight.json", "replay_v1.json", "canonical_release_baseline_attempt02.json",
        )
    ]
    probe.verify(pre)
    assert replay["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert replay["unit_tests"] == 20 and replay["actual_state_pairs"] == 14
    assert replay["terminal_carry_unchanged"] == 10 and replay["fallback_primary_records_retained"] == 9
    assert replay["all_legacy_behavior_identical_after_sidecar_projection"]
    assert len(replay["synthetic_failure_protocol_cases"]) == 2
    assert baseline["exit_code"] == 0
    assert baseline["counts"] == {"tests": 306, "failures": 0, "errors": 0, "skipped": 0}
    assert all(shared.sha(Path(p)) == h for p, h in baseline["config_runner_retriever_tests_unchanged_sha256"].items())
    paths = [HERE / name for name in (
        "preflight.json", "prepare.log", "replay_v1.json", "run_v1.log",
        "unit_tests_v1.xml", "unit_tests_v1.log",
        "canonical_release_baseline_tests_v1.xml", "canonical_release_baseline_tests_v1.log",
        "canonical_release_baseline_tests_attempt02.xml", "canonical_release_baseline_tests_attempt02.log",
        "canonical_release_baseline_attempt02.json", "baseline_attempt02_runner.log",
        "answer_conclusion_generation_original.py", "answer_conclusion_generation_candidate.py",
        "test_fallback_trace_candidate.py", "run_trace_retention_candidate.py",
        "run_release_baseline_v2.py", "close_trace_retention_candidate.py",
    )]
    shared.save(HERE / "checkpoint_v48.json", {
        "version": "isolated_trace_retention_candidate_closed_v48",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_unit_tests": 20, "baseline_release_tests": 306,
        "baseline_test_first_attempt_collection_error_preserved": True,
        "actual_state_pairs": 14, "legacy_behavior_changes": 0, "terminal_carry_unchanged": 10,
        "fallback_primary_records_retained": 9, "synthetic_failure_pairs": 2,
        "original_source_sha256": pre["canonical_source_sha256"],
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
        "production_changed": False, "training_ready": False,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "decision": "eligible_for_guarded_trace_only_integration_with_postapply_tests",
        "limitations": [
            "306 tests run against the unchanged release baseline, not the isolated candidate.",
            "Twenty candidate unit tests and fourteen paired cached states are not326 candidate tests or a semantic correctness evaluation.",
            "Two generated-error fixtures are synthetic, not actual model failures.",
            "This stage does not install source or change the live service.",
        ],
    })
    print({"candidate_unit_tests": 20, "baseline_release_tests": 306,
           "actual_state_pairs": 14, "legacy_behavior_changes": 0, "production_changed": False}, flush=True)


if __name__ == "__main__":
    main()
