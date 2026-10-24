"""Preflight every input; use only a stably idle GPU for the frozen experiment."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, digest, load, sha

GPU = "GPU-433d655c-5754-44f6-de4c-df1bc76db7e0"
MATRIX = HERE / "fact_coverage_pairs116_v1.matrix.json"
OUTPUT = HERE / "paired116_ctx12288_v1"
PREFLIGHT = HERE / "preflight_v1.json"
RECEIPT = HERE / "launch_receipt_v1.json"
LOG = HERE / "generation_v1.log"
CURRENT_CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def snapshot():
    return {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}


def processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, timeout=20)


def main():
    assert not any(p.exists() for p in (OUTPUT, PREFLIGHT, RECEIPT, LOG))
    suites = ET.parse(HERE / "experiment_tests_v1.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 9
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    matrix = load(MATRIX)
    assert matrix["evaluation_only"] and len(matrix["rows"]) == 116
    assert sha(CURRENT_CONFIG) == matrix["canonical_config_sha256"]
    frozen = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    assert sha(frozen) == matrix["frozen_config_sha256"]
    effective = deepcopy(load(frozen)["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == load(CURRENT_CONFIG)["generator"]
    release = ROOT / "release/ASA-ArknightStoryAgent"
    engine = effective["vllm"]
    base, adapter = release / engine["base_model_path"], release / engine["lora_path"]
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").exists() else base
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True,
    )
    lengths = []
    for row in matrix["rows"]:
        assert digest(row["runtime_prompt"]) == row["runtime_prompt_sha256"]
        assert digest(row["evidence_brief"]) == row["evidence_sha256"]
        ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        lengths.append({
            "id": row["id"], "input_tokens": len(ids),
            "prompt_token_ids_sha256": digest(json.dumps(ids)),
            "reserved_output": 2048, "fits_without_truncation": len(ids) + 2048 <= 12288,
        })
    source_hashes = snapshot()
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "matrix_sha256": sha(MATRIX), "canonical_config_sha256": sha(CURRENT_CONFIG),
        "source_hashes": source_hashes, "tokenizer_path": str(tokenizer_path),
        "gpu_uuid": GPU, "max_context": 12288, "max_new_tokens": 2048, "rows": lengths,
        "script_sha256": {
            str(p): sha(p) for p in (
                Path(__file__), HERE / "build_fact_coverage_pairs_v1.py",
                HERE / "run_fact_coverage_pairs_v1.py",
                RUN / "run_canonical_vllm_eval_v3.py", RUN / "analyze_canonical_grounding_v4.py",
                RUN / "generate_exx_predictions_transformers.py",
                RUN / "run_canonical_quote_eval_v2.py",
            )
        },
        "all_inputs_fit": all(r["fits_without_truncation"] for r in lengths),
    }
    save(PREFLIGHT, preflight)
    print(json.dumps({
        "preflight": str(PREFLIGHT), "all_inputs_fit": preflight["all_inputs_fit"],
        "max_input_tokens": max(r["input_tokens"] for r in lengths),
    }), flush=True)
    assert preflight["all_inputs_fit"], "No truncation or selective row removal allowed"
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
        TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
    )
    with Path("/tmp/asa_slot_feedback_GPU-433d655c-5754-44f6-de4c-df1bc76db7e0.lock").open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def record(value):
            value["at_utc"] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(value)
            log.write(line + "\n")
            print(line, flush=True)

        start, idle_since = time.monotonic(), None
        while time.monotonic() - start < 180:
            current = processes()
            busy = any(line.split(",")[0].strip() == GPU for line in current.splitlines())
            now = time.monotonic()
            idle_since = None if busy else (idle_since if idle_since is not None else now)
            idle_for = now - idle_since if idle_since is not None else 0
            record({"status": "waiting_for_idle", "target_busy": busy, "idle_seconds": round(idle_for, 1)})
            if idle_for >= 60:
                assert snapshot() == source_hashes
                assert sha(CURRENT_CONFIG) == preflight["canonical_config_sha256"]
                assert sha(MATRIX) == preflight["matrix_sha256"]
                assert all(sha(Path(p)) == h for p, h in preflight["script_sha256"].items())
                command = [
                    sys.executable, str(HERE / "run_fact_coverage_pairs_v1.py"),
                    "--input", str(MATRIX), "--output-dir", str(OUTPUT),
                    "--source-root", str(SOURCE), "--gpu-uuid", GPU,
                    "--max-new-tokens", "2048", "--max-context", "12288",
                ]
                record({"status": "launching_once", "generations": 116})
                process = subprocess.Popen(command, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait()
                receipt = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "process_pid": process.pid, "returncode": code, "gpu_uuid": GPU,
                    "source_unchanged": snapshot() == source_hashes,
                    "config_unchanged": sha(CURRENT_CONFIG) == preflight["canonical_config_sha256"],
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
