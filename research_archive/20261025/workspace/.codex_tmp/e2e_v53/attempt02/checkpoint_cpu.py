"""CPU-only handoff checkpoint, explicitly leaving GPU validation incomplete."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import check_cpu_v2 as cpu

HERE, shared = cpu.HERE, cpu.shared
V53 = cpu.wrapper.V53


def main():
    assert not (HERE / "checkpoint_cpu_v53.json").exists()
    pre = shared.load(HERE / "preflight.json")
    cpu.experiment.verify(pre)
    saved = shared.load(HERE / "cpu_integration_v1.json")
    actual = cpu.compute()
    assert all(saved[k] == v for k, v in actual.items())
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    unit_old, unit_new = (V53 / "unit_tests_v1.log").read_text(), (HERE / "unit_tests_v1.log").read_text()
    assert "Ran 28 tests" in unit_old and unit_old.rstrip().endswith("OK")
    assert "Ran 10 tests" in unit_new and unit_new.rstrip().endswith("OK")
    failure = (V53 / "run_v1.log").read_text()
    assert "GPU is not idle; no processes will be interrupted" in failure
    assert "AssertionError" in (V53 / "cpu_integration_v1.log").read_text()
    absent = []
    for directory in (V53, HERE):
        for name in ("generations.json", "generations.progress.jsonl", "idle_samples.json"):
            assert not (directory / name).exists(), str(directory / name)
            absent.append(str(directory / name))
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "-i", cpu.shared.GPU_UUID,
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True)
    files = [
        Path(__file__), HERE / "preflight.json", HERE / "inputs.json",
        HERE / "cpu_integration_v1.json", HERE / "cpu_integration_v1.log",
        HERE / "unit_tests_v1.log", HERE / "prepare.log",
        HERE / "CANONICAL_SPAN_SEMANTIC_SHADOW_20260910_v53.md",
        V53 / "run_v1.log", V53 / "cpu_integration_v1.log", V53 / "unit_tests_v1.log",
    ]
    shared.save(HERE / "checkpoint_cpu_v53.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prototype_cpu_verified_gpu_semantic_validation_incomplete",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "fresh_process_cpu_replay_equal": True, "span_bindings": actual["span_binding_count"],
        "base_unit_tests": 28, "additional_v2_unit_tests": 10,
        "invalid_reference_controls_rejected": 6,
        "synthetic_gate_checks": 12, "frozen_tasks_rebuilt_identically": 21,
        "new_llm_calls": 0, "semantic_effectiveness_tested": False,
        "model_outputs_and_idle_receipts_confirmed_absent": absent,
        "gpu_snapshot": gpu, "gpu2_compute_processes": processes,
        "production_changed": False, "training_ready": False, "human_gold": False,
        "background_wait_job_started": False,
        "remaining": "Fresh >=60s GPU2 idle guard, model run, replay, full source/output review; independent human holdout before promotion.",
    })
    print({"checkpoint": "CPU verified; semantic model run incomplete", "new_llm_calls": 0,
           "span_bindings": actual["span_binding_count"], "gpu2_compute_processes": processes.strip()}, flush=True)


if __name__ == "__main__":
    main()
