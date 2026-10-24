"""Bounded isolated server launch; verify artifacts and never kill other jobs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

STAGE = Path("${REMOTE_ASA_ROOT}/bonsai2_27b_20261012")
MODEL = STAGE / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
EXPECTED = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"


def main():
    assert MODEL.stat().st_size == 5946648928
    with MODEL.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    assert actual == EXPECTED, actual
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    free, utilization = map(int, gpu.split(","))
    # Sharing GPU1 is intentional; do not reserve all VRAM or interfere with jobs.
    assert free > 14000, f"Insufficient safe headroom: {gpu}"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 18127))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["LD_LIBRARY_PATH"] = str(STAGE / "bin") + ":" + "/usr/local/cuda-12.9/lib64"
    command = [
        str(STAGE / "bin/llama-server"), "-m", str(MODEL),
        "--alias", "bonsai2-27b", "--host", "127.0.0.1", "--port", "18127",
        "-ngl", "99", "-fa", "on", "-c", "32768", "-np", "1",
        "-b", "512", "-ub", "256", "--jinja",
        "--reasoning", "auto", "--chat-template-kwargs", '{"reasoning_effort":"medium"}',
        "--no-context-shift", "-t", "8",
    ]
    with (STAGE / "server.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=STAGE, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": process.pid, "command": command, "model_sha256": actual,
        "prelaunch_gpu1": gpu, "other_jobs_stopped": False,
        "production_modified": False, "p_tuning_applied": False,
    }
    (STAGE / "server.launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
