from datetime import datetime, timezone
from pathlib import Path
import subprocess
import run_trace_integration as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v49.json").exists()
    pre = shared.load(HERE / "preflight.json")
    result = shared.load(HERE / "installed_verification_v1.json")
    probe.verify(pre, installed=True)
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert result["tests"]["counts"] == {"tests": 326, "failures": 0, "errors": 0, "skipped": 0}
    assert result["tests"]["exit_code"] == 0
    assert result["actual_canonical_state_replays"] == len(result["rows"]) == 14
    assert result["terminal_carry_unchanged"] == 10 and result["legacy_behavior_changes"] == 0
    assert sum(bool(r["sidecars"]) for r in result["rows"]) == 9
    assert len(result["synthetic_failure_protocol_cases"]) == 2
    assert result["new_llm_calls"] == result["retrieval_calls"] == 0
    report = probe.ROOT / "docs/CANONICAL_FALLBACK_TRACE_RETENTION_INSTALLED_20260910_v49.md"
    paths = [HERE / name for name in (
        "preflight.json", "prepare.log", "preapply_guard_v1.json", "application_v1.log",
        "installed_verification_v1.json", "installed_check_v1.log",
        "release_tests_v1.xml", "release_tests_v1.log",
        "answer_conclusion_generation.py", "test_final_direct_trace_retention.py",
        "run_trace_integration.py", "close_trace_integration.py", "README.md",
    )]
    paths.extend([report, probe.SOURCE, probe.TEST])
    shared.save(HERE / "checkpoint_v49.json", {
        "version": "canonical_trace_retention_installed_closed_v49",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": True, "changed_targets": result["changed_targets"],
        "source_preimage_sha256": pre["baseline_source_sha256"],
        "source_installed_sha256": result["installed_source_sha256"],
        "canonical_preimage": pre["canonical_preimage"],
        "test_installed_sha256": result["installed_test_sha256"],
        "release_tests": 326, "actual_canonical_state_replays": 14,
        "legacy_behavior_changes": 0, "terminal_carry_unchanged": 10,
        "primary_records_retained_in_actual_fallback_states": 9,
        "synthetic_exception_fixtures": 2,
        "unrelated_legacy_source_model_config_tests_unchanged": True,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False, "training_ready": False,
        "services_restarted": False, "running_service_hot_reload_claimed": False,
        "commit_or_push": False, "semantic_accuracy_gain_claimed": False,
        "gpu_processes_at_close_read_only": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader",
        ], text=True),
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
        "important_manifest_note": result["important_manifest_note"],
        "decision": "trace_observability_fix_installed_no_answer_policy_promotion",
    })
    print({"installed_release_tests": 326, "canonical_state_replays": 14,
           "legacy_behavior_changes": 0, "production_trace_fix_installed": True}, flush=True)


if __name__ == "__main__":
    main()
