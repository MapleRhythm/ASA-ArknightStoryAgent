"""Wait for a stable empty GPU and run the existing quote-SFT adapter once."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from wait_for_idle_context_probe import GPU, update_idle_since


def main():
    run = Path(__file__).resolve().parent
    log_path = run / "quote80_sft_adapter_probe9_v1.log"
    if log_path.exists() or (run / "quote80_sft_adapter_probe9_ctx12288_v1").exists():
        raise ValueError("Existing experiment; do not duplicate")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    deadline, idle_since = time.monotonic() + 900, None
    with log_path.open("x", buffering=1) as log:
        def record(data):
            data["at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            value = json.dumps(data)
            log.write(value + "\n")
            print(value, flush=True)
        while time.monotonic() < deadline:
            process = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=20,
            )
            busy = any(line.split(",")[0].strip() == GPU for line in process.stdout.splitlines())
            now = time.monotonic()
            idle_since = update_idle_since(busy=busy, now=now, idle_since=idle_since)
            elapsed = now - idle_since if idle_since is not None else 0
            record({"status": "waiting", "target_busy": busy, "idle_seconds": round(elapsed, 1)})
            if elapsed >= 60:
                record({"status": "launching_once", "cases": 9, "training": False})
                command = ["${REMOTE_PYTHON}",
                           str(run / "run_quote80_adapter_probe_v1.py"), "--gpu-uuid", GPU]
                result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
                record({"status": "evaluation_exited", "returncode": result.returncode})
                return result.returncode
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        record({"status": "not_launched_no_stable_idle_window"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
