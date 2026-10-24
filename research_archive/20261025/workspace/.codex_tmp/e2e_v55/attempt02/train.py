"""Single process, shared-A100 training. Never kills another process."""
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import prepare

HERE = Path(__file__).resolve().parent
GPU_UUID = "GPU-17dfd5ca-b6d1-7cd5-fdc7-b1e95d13d960"


def gpu_status():
    raw = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    uuid, total, used, utilization = [value.strip() for value in raw.strip().split(",")]
    assert uuid == GPU_UUID
    raw_processes = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    processes = []
    for line in raw_processes.splitlines():
        pid, memory = [value.strip() for value in line.split(",")]
        processes.append({"pid": int(pid), "memory_mib": int(memory)})
    return {"at_utc": prepare.now(), "uuid": uuid, "total_mib": int(total),
            "used_mib": int(used), "utilization": int(utilization), "processes": processes}


def run():
    assert HERE == prepare.TRAIN_RUN
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert not (HERE / "start.json").exists()
    with (HERE / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare.verify_snapshot()
        preflight = prepare.load(HERE / "preflight.json")
        assert preflight["snapshot_sha256"] == prepare.sha(HERE / "snapshot.json")
        assert preflight["encoded_rows_sha256"] == prepare.sha(HERE / "encoded_rows.json")
        for path, expected in preflight["code_sha256"].items():
            assert prepare.sha(path) == expected, path
        for kind, root in (("base", prepare.BASE248), ("parent", prepare.PARENT248)):
            for name, expected in prepare.load(HERE / "asset_hashes.json")[kind].items():
                assert prepare.sha(root / name) == expected, (kind, name)
        assert shutil.disk_usage(HERE).free >= 6 * 1024**3
        assert not (HERE / "model").exists()
        before = gpu_status()
        assert before["total_mib"] - before["used_mib"] >= 36000, before
        assert {p["pid"] for p in before["processes"]} <= {450914}, "unreviewed resident process"
        prepare.save(HERE / "start.json", {
            "at_utc": prepare.now(), "pid": os.getpid(), "gpu_before": before,
            "user_shared_gpu_authorization": True, "other_processes_interrupted": False,
            "allocator_fraction": 0.85, "snapshot_sha256": prepare.sha(HERE / "snapshot.json"),
            "preflight_sha256": prepare.sha(HERE / "preflight.json"),
        })
        import torch
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        from liger_kernel.transformers.model.qwen3_5 import lce_forward_for_multimodal
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        equivalence = prepare.load(HERE / "kernel_equivalence.json")
        assert equivalence["passed"] and equivalence["model_forward_gradient_cases"] == 2
        apply_liger_kernel_to_qwen3_5(
            rope=False, cross_entropy=False, fused_linear_cross_entropy=True,
            rms_norm=False, swiglu=False)
        assert Qwen3_5ForConditionalGeneration.forward is lce_forward_for_multimodal
        from transformers import TrainerCallback
        from llamafactory.train.tuner import run_exp
        import yaml

        class ReceiptCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                assert Qwen3_5ForConditionalGeneration.forward is lce_forward_for_multimodal
                prepare.save(HERE / "trainer_begin.json", {
                    "at_utc": prepare.now(), "max_steps": state.max_steps,
                    "num_train_epochs": state.num_train_epochs, "pid": os.getpid(),
                    "fused_linear_cross_entropy_only": True})

            def on_step_end(self, args, state, control, **kwargs):
                receipt = {"at_utc": prepare.now(), "step": state.global_step, "epoch": state.epoch,
                           "cuda_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                           "cuda_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
                           "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
                with (HERE / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(receipt) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if shutil.disk_usage(HERE).free < 3 * 1024**3:
                    raise RuntimeError("low disk: stopping only this training process")

            def on_log(self, args, state, control, logs=None, **kwargs):
                for name in ("loss", "grad_norm", "eval_loss"):
                    if logs and name in logs and not math.isfinite(float(logs[name])):
                        raise RuntimeError(f"nonfinite {name}")

        config = yaml.safe_load((HERE / "train.yaml").read_text())
        run_exp(config, callbacks=[ReceiptCallback()])
        assert (HERE / "model/adapter_model.safetensors").is_file()
        prepare.verify_snapshot()
        prepare.save(HERE / "complete.json", {
            "at_utc": prepare.now(), "exit_code": 0, "pid": os.getpid(),
            "adapter_sha256": prepare.sha(HERE / "model/adapter_model.safetensors"),
            "production_promoted": False, "canonical_return_pending": True,
        })


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        if not (HERE / "failure.json").exists():
            prepare.save(HERE / "failure.json", {
                "at_utc": prepare.now(), "error_type": type(error).__name__,
                "message": str(error)[:1000], "pid": os.getpid()})
        traceback.print_exc()
        sys.exit(1)
