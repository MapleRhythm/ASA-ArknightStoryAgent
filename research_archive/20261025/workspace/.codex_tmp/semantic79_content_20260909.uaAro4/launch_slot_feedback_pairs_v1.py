"""Launch once on one explicitly idle 241 GPU; no service or other process is stopped."""
from datetime import datetime, timezone
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"
GPU = "GPU-433d655c-5754-44f6-de4c-df1bc76db7e0"
MATRIX = HERE / "slot_feedback_paired8_v1.matrix.json"
OUTPUT = HERE / "slot_feedback_paired8_ctx12288_v1"
LOG = HERE / "slot_feedback_paired8_v1.log"
RECEIPT = HERE / "slot_feedback_launch_v1.json"
MAX_CONTEXT = 12288
MAX_NEW = 2048


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, timeout=20)


def code_snapshot():
    return {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}


def main():
    if any(path.exists() for path in (OUTPUT, LOG, RECEIPT)):
        raise ValueError("Existing launch artifact; never duplicate or overwrite")
    matrix = json.loads(MATRIX.read_text())
    assert matrix["evaluation_only"] and len(matrix["rows"]) == 8
    frozen = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    current = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    assert sha(frozen) == matrix["frozen_config_sha256"] and sha(current) == matrix["canonical_config_sha256"]
    effective = deepcopy(json.loads(frozen.read_text())["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = MAX_CONTEXT
    assert effective == json.loads(current.read_text())["generator"]
    assert matrix["config_context_override"]["explicit_max_context"] == MAX_CONTEXT
    config = json.loads(frozen.read_text())["generator"]["vllm"]
    base = ROOT / "release/ASA-ArknightStoryAgent" / config["base_model_path"]
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    lengths = []
    for row in matrix["rows"]:
        assert hashlib.sha256(row["runtime_prompt"].encode()).hexdigest() == row["runtime_prompt_sha256"]
        length = len(tokenizer.encode(row["runtime_prompt"], add_special_tokens=True))
        assert length + MAX_NEW <= MAX_CONTEXT, (row["id"], length)
        lengths.append({"id": row["id"], "input_tokens": length, "reserved_output": MAX_NEW, "fits_without_truncation": True})
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True, "training_ready": False,
        "gpu_uuid": GPU, "matrix_sha256": sha(MATRIX), "launcher_sha256": sha(Path(__file__)),
        "source_hashes": code_snapshot(), "context": MAX_CONTEXT, "rows": lengths,
        "harness_sha256": {name: sha(RUN / name) for name in (
            "run_frozen_seeded_pairs_v2.py", "run_canonical_vllm_eval_v3.py", "run_paired_vllm_retry_v1.py",
        )},
    }
    with (HERE / "slot_feedback_preflight_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, ensure_ascii=False, indent=2)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    lock_path = Path("/tmp/asa_slot_feedback_GPU-433d655c-5754-44f6-de4c-df1bc76db7e0.lock")
    with lock_path.open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def record(value):
            value["at_utc"] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(value)
            log.write(line + "\n")
            print(line, flush=True)

        started, idle_since = time.monotonic(), None
        while time.monotonic() - started < 180:
            processes = compute_processes()
            busy = any(line.split(",")[0].strip() == GPU for line in processes.splitlines())
            now = time.monotonic()
            idle_since = None if busy else (idle_since if idle_since is not None else now)
            idle_for = now - idle_since if idle_since is not None else 0
            record({"status": "waiting_for_stable_idle", "target_busy": busy, "idle_seconds": round(idle_for, 1)})
            if idle_for >= 60:
                assert code_snapshot() == preflight["source_hashes"]
                command = [
                    sys.executable, str(RUN / "run_frozen_seeded_pairs_v2.py"),
                    "--input", str(MATRIX), "--output-dir", str(OUTPUT),
                    "--source-root", str(SOURCE), "--gpu-uuid", GPU,
                    "--max-new-tokens", str(MAX_NEW), "--max-context", str(MAX_CONTEXT),
                ]
                record({"status": "launching_once", "paired_generations": 8})
                process = subprocess.Popen(command, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait()
                receipt = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_uuid": GPU, "process_pid": process.pid, "returncode": code,
                    "matrix_sha256": sha(MATRIX), "output": str(OUTPUT),
                    "source_unchanged": code_snapshot() == preflight["source_hashes"],
                    "post_run_compute_processes": compute_processes(), "training_ready": False,
                }
                with RECEIPT.open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, ensure_ascii=False, indent=2)
                record({"status": "evaluation_exited", "returncode": code, "source_unchanged": receipt["source_unchanged"]})
                return code
            time.sleep(10)
        record({"status": "not_launched_no_stable_idle_window"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
