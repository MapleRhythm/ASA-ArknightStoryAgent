"""Detached, bounded training followed by raw generation; records real child PIDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--memory-gate", required=True)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    path = root / f"{args.tag}.workflow.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True",
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    python = "${REMOTE_PYTHON}"
    if not args.worker:
        assert not path.exists() and not (root / args.tag).exists()
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", "1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).strip())
        assert free >= 20500, free
        gate = json.loads(Path(args.memory_gate).read_text())
        assert gate["completed"] and gate["activation_offload"]
        command = [python, "-u", str(Path(__file__)), "--data", args.data,
                   "--tag", args.tag, "--memory-gate", args.memory_gate, "--worker"]
        with (root / f"{args.tag}.workflow.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"workflow_pid": proc.pid, "gpu1_free_before_MiB": free,
                          "tag": args.tag, "production_modified": False,
                          "other_jobs_stopped": False}), flush=True)
        return
    record = {"workflow_pid": os.getpid(), "started_unix": time.time(),
              "stage": "starting", "production_modified": False, "other_jobs_stopped": False}
    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2))
        os.replace(temporary, path)
    save()
    commands = [
        ("training", [python, "-u", "train_pilot.py", "--data", args.data,
                      "--out", args.tag, "--epochs", "1", "--lr", "0.00002",
                      "--offload", "--validation-limit", "2",
                      "--gate", "full_gate_bfloat16_bounded_20261013.json",
                      "--memory-gate", args.memory_gate]),
        ("raw_generation", [python, "-u", "generate_eval.py", "--data", args.data,
                            "--run", args.tag, "--samples", "2",
                            "--out", f"{args.tag}/generated_heldout2.jsonl"]),
    ]
    for stage, command in commands:
        with (root / f"{args.tag}.{stage}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            record.update(stage=stage, child_pid=proc.pid, command=command)
            save()
            code = proc.wait()
        record[f"{stage}_exit_code"] = code
        if code:
            record.update(stage="failed", failed_stage=stage, finished_unix=time.time())
            save()
            return
    record.update(stage="completed", finished_unix=time.time(),
                  generated_answers_not_adjudicated=True)
    save()


if __name__ == "__main__":
    main()
