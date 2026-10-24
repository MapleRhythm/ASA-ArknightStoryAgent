"""Wait up to 15 minutes for a 60-second empty GPU window, launch once."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from wait_for_idle_context_probe import GPU, update_idle_since


def main():
    run = Path(__file__).resolve().parent
    folder = run / "same_evidence_retry_paired68_ctx12288_v1"
    log_path = run / "same_evidence_retry_paired68_v1.log"
    if folder.exists() or log_path.exists():
        raise ValueError("existing run/log; do not duplicate work")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    deadline = time.monotonic() + 900
    idle_since = None
    with log_path.open("x", buffering=1) as log:
        def record(data):
            data["at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            message = json.dumps(data)
            log.write(message + "\n")
            print(message, flush=True)
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=20,
            )
            busy = any(line.split(",")[0].strip() == GPU for line in result.stdout.splitlines())
            now = time.monotonic()
            idle_since = update_idle_since(busy=busy, now=now, idle_since=idle_since)
            duration = now - idle_since if idle_since is not None else 0
            record({"status": "waiting", "target_busy": busy, "idle_seconds": round(duration, 1)})
            if duration >= 60:
                record({"status": "launching_once", "paired_generations": 68})
                command = [
                    "${REMOTE_PYTHON}",
                    str(run / "run_paired_vllm_retry_v1.py"),
                    "--input", str(run / "same_evidence_retry_paired68_v1.matrix.json"),
                    "--output-dir", str(folder),
                    "--source-root", str(run / "context_guard_candidate_bytes_v1/src"),
                    "--gpu-uuid", GPU, "--max-new-tokens", "2048", "--max-context", "12288",
                ]
                completed = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
                record({"status": "evaluation_exited", "returncode": completed.returncode})
                return completed.returncode
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        record({"status": "not_launched_no_stable_idle_window"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
