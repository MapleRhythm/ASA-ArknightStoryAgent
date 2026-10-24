"""Resume the unchanged trainer with process-local caches on the experiment disk."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import annotation_protocol as p
import resume_targeted_pt_v2 as recovery
import targeted_pt_v2 as t
from audit_fresh_subset import require


CACHE_DIRS = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "cuda",
    "XDG_CACHE_HOME": "xdg",
    "TMPDIR": "tmp",
}


def verify_storage(out):
    require(out.is_absolute() and out.is_dir(), "existing absolute experiment path required")
    require(out.stat().st_dev != Path.home().stat().st_dev,
            "experiment must be on a separate filesystem from home")
    free = shutil.disk_usage(out).free
    require(free >= 20 * 1024 ** 3, "data filesystem needs at least 20 GiB free")
    return {"data_device": out.stat().st_dev, "free_bytes": free}


def cache_environment(out):
    """Never move/delete old caches; refuse symlinks and keep new paths private."""
    root = out / "runtime_cache_disk_v1"
    require(not root.is_symlink(), "cache root must not be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    result = {}
    for key, name in CACHE_DIRS.items():
        directory = root / name
        require(not directory.is_symlink(), "cache directory must not be a symlink")
        directory.mkdir(mode=0o700, exist_ok=True)
        require(directory.resolve().parent == root.resolve(), "cache escaped experiment")
        require(directory.stat().st_dev == out.stat().st_dev, "cache is on another filesystem")
        require(os.access(directory, os.W_OK), "cache directory is not writable")
        result[key] = str(directory.resolve())
    return result


def checkpoint_preflight(out):
    import torch
    from safetensors.torch import load_file

    merged = t.verify_merged(out / "merged_data")
    run_dir = out / "trained_v2"
    run = json.loads((run_dir / "run.json").read_text())
    require(not run["training_complete"], "training is already complete")
    require(run["data_sha256"] == merged["data_sha256"], "resume dataset changed")
    require(run["backbone_frozen"] and run["no_truncation"], "training invariants changed")
    state_path = run_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    step = state["step"]
    require(isinstance(step, int) and 0 < step < merged["train"], "invalid saved step")
    require(state.get("optimizer") and "torch_rng" in state and "cuda_rng" in state,
            "missing optimizer or RNG checkpoint")
    events = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    steps = [event for event in events if event.get("event") == "optimizer_step"]
    require([event["step"] for event in steps] == list(range(1, step + 1)),
            "logged and checkpoint optimizer steps do not match")
    rows = t.read_rows(out / "merged_data/train.jsonl")
    order = list(range(len(rows)))
    random.Random(20261012).shuffle(order)
    require(steps[-1]["id"] == rows[order[step - 1]]["id"], "checkpoint sample order mismatch")
    adapter = run_dir / f"checkpoint-{step}"
    config = json.loads((adapter / "adapter_config.json").read_text())
    require(config["source_sha256"] == run["source_sha256"], "checkpoint backbone mismatch")
    weights = load_file(str(adapter / "prompt_encoder.safetensors"), device="cpu")
    require(sum(value.numel() for value in weights.values()) == run["trainable_parameters"]
            and all(torch.isfinite(value).all().item() for value in weights.values()),
            "invalid prompt encoder checkpoint")
    for item in state["optimizer"]["state"].values():
        require(all(not torch.is_tensor(value) or torch.isfinite(value).all().item()
                    for value in item.values()), "nonfinite optimizer checkpoint")
    return dict(saved_optimizer_step=step, total_steps=merged["train"],
                next_sample_id=rows[order[step]]["id"], data_sha256=merged["data_sha256"],
                optimizer_state_sha256=p.file_sha(state_path),
                adapter_sha256=p.file_sha(adapter / "prompt_encoder.safetensors"),
                unchanged_trainer_sha256=p.file_sha(Path(t.__file__).with_name("train_pilot.py")))


def launch(out, gpu):
    storage = verify_storage(out)
    state = json.loads((out / "workflow.json").read_text())
    require(state["stage"] == "blocked_or_failed" and state["failed_stage"] == "training",
            "only resume an interrupted training stage")
    require(not recovery.job_alive(state.get("pid"), out), "workflow is still alive")
    checkpoint = checkpoint_preflight(out)
    free_mib = int(t.subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True).strip())
    require(free_mib >= 20500, "insufficient GPU headroom")
    env = cache_environment(out)
    stamp = str(time.time_ns())
    previous = out / "recovery.launch.json"
    if previous.exists():
        p.atomic_json(out / f"recovery.launch.before-disk-{stamp}.json",
                      json.loads(previous.read_text()))
    receipt_path = out / f"disk-recovery-{stamp}.json"
    receipt = dict(kind="unchanged_training_resume_with_data_disk_caches",
                   stage="preflight_passed", cache_environment=env, storage=storage,
                   checkpoint=checkpoint, gpu=gpu, free_gpu_mib=free_mib,
                   launcher_sha256=p.file_sha(__file__), old_caches_removed=False,
                   production_modified=False, data_or_optimizer_rewritten=False,
                   created_unix=time.time())
    p.atomic_json(receipt_path, receipt)
    os.environ.update(env)
    if not os.environ.get("BIGMODEL_API_KEY"):
        os.environ["BIGMODEL_API_KEY"] = t.read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    result = recovery.launch(out, gpu)
    receipt.update(stage="launched", recovery=result)
    p.atomic_json(receipt_path, receipt)
    return dict(result, disk_resume_receipt=str(receipt_path),
                saved_optimizer_step=checkpoint["saved_optimizer_step"],
                cache_environment=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    print(p.compact(launch(args.out.resolve(), args.gpu)), flush=True)


if __name__ == "__main__":
    main()
