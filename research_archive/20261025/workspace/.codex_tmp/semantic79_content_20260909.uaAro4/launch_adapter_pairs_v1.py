"""Launch the registered 12-row adapter comparison on one stably idle 4090."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"
GPU = "GPU-433d655c-5754-44f6-de4c-df1bc76db7e0"
MATRIX = HERE / "adapter_pairs12_v1.matrix.json"
OUTPUT = HERE / "adapter_pairs12_ctx12288_v1"
LOG = HERE / "adapter_pairs12_v1.log"
RECEIPT = HERE / "adapter_pairs_launch_v1.json"
PREFLIGHT = HERE / "adapter_pairs_preflight_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}


def processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, timeout=20)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not any(p.exists() for p in (OUTPUT, LOG, RECEIPT, PREFLIGHT))
    suites = ET.parse(HERE / "adapter_dispatch_tests_v1.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 7
    assert all(int(s.attrib.get("failures", 0)) == int(s.attrib.get("errors", 0)) == 0 for s in suites)
    matrix = json.loads(MATRIX.read_text())
    assert matrix["evaluation_only"] and len(matrix["rows"]) == 12
    frozen = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    current = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    assert sha(frozen) == matrix["frozen_config_sha256"]
    assert sha(current) == matrix["canonical_config_sha256"]
    effective = deepcopy(json.loads(frozen.read_text())["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == json.loads(current.read_text())["generator"]
    release = ROOT / "release/ASA-ArknightStoryAgent"
    engine = effective["vllm"]
    adapter = release / engine["lora_path"]
    base = release / engine["base_model_path"]
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").exists() else base
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    lengths, paired_ids = [], {}
    for row in matrix["rows"]:
        assert hashlib.sha256(row["runtime_prompt"].encode()).hexdigest() == row["runtime_prompt_sha256"]
        ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        assert len(ids) + 2048 <= 12288
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if row["parent_record_id"] in paired_ids:
            assert paired_ids[row["parent_record_id"]] == digest
        paired_ids[row["parent_record_id"]] = digest
        lengths.append({
            "id": row["id"], "input_tokens": len(ids), "prompt_token_ids_sha256": digest,
            "reserved_output": 2048, "fits_without_truncation": True,
        })
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "matrix_sha256": sha(MATRIX), "source_hashes": snapshot(),
        "tokenizer_path": str(tokenizer_path), "gpu_uuid": GPU, "rows": lengths,
        "max_context": 12288, "max_new_tokens": 2048,
        "script_sha256": {
            str(p): sha(p) for p in (
                Path(__file__), HERE / "run_adapter_pairs_v1.py",
                RUN / "run_canonical_vllm_eval_v3.py",
            )
        },
    }
    save(PREFLIGHT, preflight)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    with Path("/tmp/asa_slot_feedback_GPU-433d655c-5754-44f6-de4c-df1bc76db7e0.lock").open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def record(value):
            value["at_utc"] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(value)
            log.write(line + "\n")
            print(line, flush=True)

        start, idle_since = time.monotonic(), None
        while time.monotonic() - start < 180:
            current_processes = processes()
            busy = any(line.split(",")[0].strip() == GPU for line in current_processes.splitlines())
            now = time.monotonic()
            idle_since = None if busy else (idle_since if idle_since is not None else now)
            idle_for = now - idle_since if idle_since is not None else 0
            record({"status": "waiting_for_idle", "target_busy": busy, "idle_seconds": round(idle_for, 1)})
            if idle_for >= 60:
                assert snapshot() == preflight["source_hashes"]
                assert sha(MATRIX) == preflight["matrix_sha256"]
                command = [
                    sys.executable, str(HERE / "run_adapter_pairs_v1.py"),
                    "--input", str(MATRIX), "--output-dir", str(OUTPUT),
                    "--source-root", str(SOURCE), "--gpu-uuid", GPU,
                    "--max-new-tokens", "2048", "--max-context", "12288",
                ]
                record({"status": "launching_once", "generations": 12})
                process = subprocess.Popen(command, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait()
                receipt = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "process_pid": process.pid, "returncode": code, "gpu_uuid": GPU,
                    "source_unchanged": snapshot() == preflight["source_hashes"],
                    "matrix_sha256": sha(MATRIX), "post_compute_processes": processes(),
                    "evaluation_only": True, "training_ready": False,
                }
                save(RECEIPT, receipt)
                record({"status": "exited", "returncode": code, "source_unchanged": receipt["source_unchanged"]})
                return code
            time.sleep(10)
        record({"status": "not_launched_no_idle_window"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
