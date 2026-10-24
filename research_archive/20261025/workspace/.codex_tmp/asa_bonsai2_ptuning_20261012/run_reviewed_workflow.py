"""Isolated longest-context gate -> encoder training -> heldout raw generation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from audit_fresh_subset import verify, require, file_sha, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag) is not None, "unsafe tag")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    path = root / f"{args.tag}.workflow.json"
    gate_path = root / f"{args.tag}.memory_gate.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               PYTORCH_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false")
    python = sys.executable
    verified = verify(Path(args.data))
    selection = json.loads(Path(args.selection).read_text())
    val = {r["id"]: r for r in read_rows(Path(args.data) / "val.jsonl")}
    require(len(selection["ids"]) == len(set(selection["ids"])) == 8 and
            set(selection["ids"]) <= set(val), "invalid heldout selection")
    require({json.loads(val[i]["target"])["next_action"] for i in selection["ids"]} ==
            {"answer_directly", "abstain", "retrieve_more"}, "missing action stratum")
    if not args.worker:
        require(not path.exists() and not (root / args.tag).exists() and
                not gate_path.exists(), "refusing to overwrite experiment")
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True).strip())
        require(free >= 20500, f"GPU headroom insufficient: {free} MiB")
        command = [python, "-u", str(Path(__file__).resolve()), "--data", args.data,
                   "--tag", args.tag, "--selection", args.selection,
                   "--gpu", str(args.gpu), "--worker"]
        with (root / f"{args.tag}.workflow.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(workflow_pid=proc.pid, gpu=args.gpu,
                              gpu_free_before_MiB=free, tag=args.tag, review=verified,
                              production_modified=False, other_jobs_stopped=False)), flush=True)
        return
    record = dict(workflow_pid=os.getpid(), started_unix=time.time(), stage="starting",
                  gpu=args.gpu, review=verified, selection_sha256=file_sha(args.selection),
                  production_modified=False, other_jobs_stopped=False)

    def save():
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        os.replace(temp, path)

    save()
    commands = [
        ("memory_gate", [python, "-u", "memory_gate.py", "--data", args.data,
                         "--out", str(gate_path), "--offload"]),
        ("training", [python, "-u", "train_reviewed.py", "--data", args.data,
                      "--out", args.tag, "--epochs", "1", "--lr", "0.00002",
                      "--offload", "--validation-limit", "0",
                      "--gate", "full_gate_bfloat16_bounded_20261013.json",
                      "--memory-gate", str(gate_path)]),
        ("raw_generation", [python, "-u", "generate_reviewed.py", "--data", args.data,
                            "--run", args.tag, "--samples", "8", "--selection", args.selection,
                            "--out", f"{args.tag}/generated_heldout8.jsonl"]),
    ]
    try:
        for stage, command in commands:
            with (root / f"{args.tag}.{stage}.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
                record.update(stage=stage, child_pid=proc.pid, command=command)
                save()
                code = proc.wait()
            record[f"{stage}_exit_code"] = code
            if code:
                raise RuntimeError(f"{stage} exited with code {code}")
        record.update(stage="completed", finished_unix=time.time(),
                      generated_answers_not_adjudicated=True)
        save()
    except Exception as error:
        record.update(failed_stage=record["stage"], stage="failed", error=str(error),
                      finished_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
