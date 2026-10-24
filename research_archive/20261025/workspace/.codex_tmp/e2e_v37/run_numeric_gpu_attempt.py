"""Attempt02: same frozen protocol; correct only GPU visibility representation."""
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "planner_entity_provenance_20260910.v37"
sys.path.insert(0, str(ORIGINAL))
import run_planner_capture as experiment

shared = experiment.shared
experiment.HERE = HERE


def mapping():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader",
    ], text=True)
    rows = [[part.strip() for part in line.split(",")] for line in text.splitlines()]
    target = next(row for row in rows if row[0] == "2")
    assert target[1] == shared.GPU_UUID
    assert sorted(rows, key=lambda row: row[2])[2] == target
    return rows


def prepare():
    old = shared.load(ORIGINAL / "preflight.json")
    failure = shared.load(ORIGINAL / "startup_failure_v1.json")
    assert failure["actual_generation_calls"] == 0
    original_save = shared.save
    gpu_mapping = mapping()

    def save(path, report):
        assert path == HERE / "preflight.json"
        assert report["rows"] == old["rows"]
        assert report["protocol"] == old["protocol"]
        for key in ("base_model", "adapter", "tokenizer", "runtime"):
            assert report[key] == old[key]
        report["attempt"] = 2
        report["original_failed_attempt"] = str(ORIGINAL)
        report["launch_environment"] = {
            "CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "expected_gpu_uuid": shared.GPU_UUID, "verified_nvidia_mapping": gpu_mapping,
        }
        for extra in (Path(__file__), ORIGINAL / "startup_failure_v1.json",
                      ORIGINAL / "preflight.json", ORIGINAL / "check_planner_equivalence.py"):
            report["sha256"][str(extra)] = shared.sha(extra)
        original_save(path, report)

    shared.save = save
    try:
        experiment.prepare()
    finally:
        shared.save = original_save


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert mapping() == shared.load(HERE / "preflight.json")["launch_environment"]["verified_nvidia_mapping"]
    experiment.run()


def method_check():
    import check_planner_equivalence as checker
    checker.HERE = HERE
    checker.main()


if __name__ == "__main__":
    operations = {
        "prepare": prepare, "run": run, "check": experiment.check, "method_check": method_check,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_numeric_gpu_attempt.py prepare|run|check|method_check")
    operations[sys.argv[1]]()
