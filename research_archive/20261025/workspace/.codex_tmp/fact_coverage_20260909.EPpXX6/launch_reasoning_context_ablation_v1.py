"""One same4B58 ablation on a fresh idle GPU, with frozen assets and full replay."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, RUN, ROOT, SOURCE, load, sha, digest
from build_reasoning_context_ablation_v1 import MATRIX, remove_reasoning_context
from verify_empty_quote_candidate_v1 import counts, save, snapshot
from run_native8b_v1 import GPU, observe

CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
PREFLIGHT = HERE / "without_reasoning_hints58_execution_preflight_v1.json"
OUTPUT = HERE / "without_reasoning_hints58_ctx12288_v1"
LOG = HERE / "without_reasoning_hints58_generation_v1.log"
RECEIPT = HERE / "without_reasoning_hints58_launch_receipt_v1.json"
VALIDATION = HERE / "without_reasoning_hints58_validation_v1.json"


def manifest(path):
    sys.path.insert(0, str(RUN))
    from generate_exx_predictions_transformers import model_asset_manifest
    return model_asset_manifest(path)


def dependencies():
    return {str(p): sha(p) for p in [
        Path(__file__), HERE / "build_reasoning_context_ablation_v1.py",
        HERE / "run_reasoning_context_ablation_v1.py", HERE / "run_fact_coverage_pairs_v1.py",
        HERE / "test_reasoning_context_ablation_v1.py", HERE / "run_native8b_v1.py",
        HERE / "build_fact_coverage_pairs_v1.py", HERE / "verify_empty_quote_candidate_v1.py",
        RUN / "run_canonical_vllm_eval_v3.py", RUN / "generate_exx_predictions_transformers.py",
        RUN / "run_canonical_quote_eval_v2.py", RUN / "analyze_canonical_grounding_v4.py",
    ]}


def unchanged(pre):
    assert sha(MATRIX) == pre["matrix_sha256"]
    assert sha(CONFIG) == pre["canonical_config_sha256"]
    assert snapshot(SOURCE) == pre["source_hashes"]
    assert dependencies() == pre["dependencies_sha256"]


def main():
    assert not any(p.exists() for p in (PREFLIGHT, OUTPUT, LOG, RECEIPT, VALIDATION))
    assert counts(HERE / "reasoning_context_tests_v1.xml") == {
        "tests": 7, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(HERE / "experiment_tests_v1.xml") == {
        "tests": 9, "failures": 0, "errors": 0, "skipped": 0}
    matrix = load(MATRIX)
    originals = [r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                 if r["renderer"] == "original_prompt_v1"]
    assert len(originals) == len(matrix["rows"]) == 58
    assert sha(HERE / "fact_coverage_pairs116_v1.matrix.json") == matrix["source_matrix_sha256"]
    effective = deepcopy(load(RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json")["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == load(CONFIG)["generator"]
    release = ROOT / "release/ASA-ArknightStoryAgent"
    base = release / effective["vllm"]["base_model_path"]
    adapter = release / effective["vllm"]["lora_path"]
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").is_file() else base
    old_metadata_path = HERE / "paired116_ctx12288_v1/metadata.json"
    old_metadata = load(old_metadata_path)
    base_assets, adapter_assets = manifest(base), manifest(adapter)
    assert base_assets == old_metadata["base_model_assets"]
    assert adapter_assets == old_metadata["adapter_assets"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    lengths = []
    for original, row in zip(originals, matrix["rows"]):
        assert row["user"] == remove_reasoning_context(original)[0]
        for key in ("question", "hypothesis", "evidence_brief", "prompt_evidence", "sampling_seed",
                    "system", "current_round", "max_retrieval_rounds", "assistant_prefix"):
            assert row[key] == original[key]
        assert render_qwen_chat_prompt(row["system"], row["user"],
                                      assistant_prefix=row["assistant_prefix"]) == row["runtime_prompt"]
        assert digest(row["runtime_prompt"]) == row["runtime_prompt_sha256"]
        ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        assert len(ids) + 2048 <= 12288, "do not truncate or drop rows"
        lengths.append({"id": row["id"], "input_tokens": len(ids),
                        "prompt_token_ids_sha256": digest(json.dumps(ids)), "seed": row["sampling_seed"]})
    pre = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "matrix_sha256": sha(MATRIX), "source_hashes": snapshot(SOURCE),
        "canonical_config_sha256": sha(CONFIG), "dependencies_sha256": dependencies(),
        "base_path": str(base), "adapter_path": str(adapter), "base_assets": base_assets,
        "adapter_assets": adapter_assets, "old116_metadata_sha256": sha(old_metadata_path),
        "all_inputs_fit": True, "rows": lengths, "gpu_uuid": GPU,
        "decoding": {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8,
                     "repeat_penalty": effective["repeat_penalty"], "max_context": 12288},
    }
    save(PREFLIGHT, pre)
    print(json.dumps({"preflight58": True, "max_input": max(r["input_tokens"] for r in lengths),
                      "base_and_adapter_match_prior116": True}), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    observations = []
    with Path("/tmp/asa_slot_feedback_" + GPU + ".lock").open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, idle_since = time.monotonic(), None
        while time.monotonic() - start < 180:
            observation = observe()
            now = time.monotonic()
            idle_since = None if not observation["idle"] else (now if idle_since is None else idle_since)
            observation.update(at_utc=datetime.now(timezone.utc).isoformat(),
                               idle_seconds=0 if idle_since is None else round(now - idle_since, 2))
            observations.append(observation)
            log.write(json.dumps(observation) + "\n")
            print(json.dumps(observation), flush=True)
            if idle_since is not None and now - idle_since >= 60:
                unchanged(pre)
                command = [
                    sys.executable, str(HERE / "run_reasoning_context_ablation_v1.py"),
                    "--input", str(MATRIX), "--output-dir", str(OUTPUT),
                    "--source-root", str(SOURCE), "--gpu-uuid", GPU,
                    "--max-new-tokens", "2048", "--max-context", "12288",
                ]
                child = subprocess.Popen(command, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                receipt = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": code, "child_pid": child.pid, "gpu_uuid": GPU,
                    "preflight_sha256": sha(PREFLIGHT), "idle_observations": observations,
                    "base_assets_after_equal": manifest(base) == base_assets,
                    "adapter_assets_after_equal": manifest(adapter) == adapter_assets,
                }
                save(RECEIPT, receipt)
                print(json.dumps({"generation_returncode": code}), flush=True)
                if code:
                    return code
                assert receipt["base_assets_after_equal"] and receipt["adapter_assets_after_equal"]
                unchanged(pre)
                subprocess.run([
                    sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
                    "--validator", "after", "--source-root", str(SOURCE),
                    "--matrix", str(MATRIX), "--predictions", str(OUTPUT / "predictions.json"),
                    "--output", str(VALIDATION),
                ], check=True, env=dict(env, CUDA_VISIBLE_DEVICES=""))
                return 0
            time.sleep(10)
        save(RECEIPT, {"launched": False, "reason": "no_sustained_idle_window",
                       "idle_observations": observations})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
