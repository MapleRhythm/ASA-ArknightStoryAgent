"""Bounded read-only idle guard before one candidate evaluation.

Never stop or share another GPU process. Require 60 seconds continuously idle,
then let the inference runner recheck before allocation. No retries after launch.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


GPU = "GPU-90144416-2626-05e7-0b89-a4e45559ddf9"


def update_idle_since(*, busy, now, idle_since):
    if busy:
        return None
    return now if idle_since is None else idle_since


def main():
    run = Path(__file__).resolve().parent
    output = run / "strict9_native_exx_explicit_schema_ctx12288_v1"
    log_path = run / "strict9_native_exx_explicit_schema_ctx12288_v1.log"
    if output.exists() or log_path.exists():
        raise ValueError("existing run/log; refusing duplicate launch")
    deadline = time.monotonic() + 900
    idle_since = None
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    with log_path.open("x", buffering=1) as log:
        def record(data):
            data["at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            line = json.dumps(data)
            log.write(line + "\n")
            print(line, flush=True)
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=20,
            )
            rows = [line for line in result.stdout.splitlines() if line.split(",")[0].strip() == GPU]
            now = time.monotonic()
            idle_since = update_idle_since(busy=bool(rows), now=now, idle_since=idle_since)
            seconds = now - idle_since if idle_since is not None else 0
            record({"status": "waiting", "target_busy": bool(rows), "continuously_idle_seconds": round(seconds, 1)})
            if seconds >= 60:
                record({"status": "launching_once", "source": "quote_integrity_candidate_v1/src"})
                command = [
                    "${REMOTE_PYTHON}",
                    str(run / "run_canonical_vllm_eval_v3.py"),
                    "--input", str(run / "strict9_native_exx_explicit_schema_v1.matrix.json"), "--output-dir", str(output),
                    "--source-root", str(run / "quote_integrity_candidate_v1/src"),
                    "--gpu-uuid", GPU, "--max-new-tokens", "512", "--max-context", "12288",
                ]
                completed = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
                record({"status": "evaluation_exited", "returncode": completed.returncode})
                return completed.returncode
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        record({"status": "not_launched_no_stable_idle_window", "wait_limit_seconds": 900})
    return 2


if __name__ == "__main__":
    sys.exit(main())
