"""Fresh frozen preflight and one-shot native8B evaluation; never a live config."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, digest, load, sha
from verify_empty_quote_candidate_v1 import counts, save, snapshot

MODEL = ROOT / "model/qwen3-8b"
CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
MATRIX = HERE / "qwen3_8b_native58_v1.matrix.json"
PREFLIGHT = HERE / "qwen3_8b_execution_preflight_v1.json"
OUTPUT = HERE / "qwen3_8b_native58_ctx12288_v1"
GPU = "GPU-433d655c-5754-44f6-de4c-df1bc76db7e0"
LOG = HERE / "qwen3_8b_generation_v1.log"
RECEIPT = HERE / "qwen3_8b_launch_receipt_v1.json"
TEST_XML = HERE / "native8b_harness_tests_v1.xml"
VALIDATION = HERE / "qwen3_8b_validation_v1.json"


def assets():
    sys.path.insert(0, str(RUN))
    from generate_exx_predictions_transformers import model_asset_manifest
    return model_asset_manifest(MODEL)


def dependencies():
    return {
        str(p): sha(p) for p in [
            Path(__file__), HERE / "build_fact_coverage_pairs_v1.py",
            HERE / "verify_empty_quote_candidate_v1.py", HERE / "test_native8b_harness_v1.py",
            RUN / "generate_exx_predictions_transformers.py",
            RUN / "analyze_canonical_grounding_v4.py",
        ]
    }


def ensure_frozen(preflight):
    assert sha(MATRIX) == preflight["matrix_sha256"], "matrix changed"
    assert sha(CONFIG) == preflight["canonical_config_sha256"], "canonical config changed"
    assert snapshot(SOURCE) == preflight["source_hashes"], "canonical runtime changed"
    assert dependencies() == preflight["dependencies_sha256"], "harness changed"


def check_row(row, original, tokenizer):
    assert row["parent_record_id"] == original["parent_record_id"]
    for key in ("question", "system", "user", "evidence_brief", "evidence_sha256",
                "prompt_evidence", "hypothesis", "current_round", "max_retrieval_rounds", "sampling_seed"):
        assert row[key] == original[key], f"changed comparison input: {key}"
    assert type(row["sampling_seed"]) is int
    messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
    assert row["native_messages"] == messages and row["native_template_enable_thinking"] is False
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert row["assistant_prefix"] == "" and rendered == row["runtime_prompt"]
    assert row["template"] == "runtime_raw"
    assert digest(rendered) == row["runtime_prompt_sha256"]
    assert digest(row["evidence_brief"]) == row["evidence_sha256"]
    assert row["evidence_brief"] in rendered
    ids = tokenizer.encode(rendered, add_special_tokens=True)
    assert ids == tokenizer.encode(rendered, add_special_tokens=False)
    assert len(ids) + 2048 <= 12288, "do not truncate or drop an overflowing row"
    return {"id": row["id"], "input_tokens": len(ids), "prompt_token_ids_sha256": digest(json.dumps(ids))}


def idle_observation(gpu, process_csv, stats_csv):
    process_busy = any(line.split(",")[0].strip() == gpu for line in process_csv.splitlines())
    lines = [line.split(",") for line in stats_csv.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0][0].strip() == gpu, "ambiguous GPU identity"
    _, used, free, utilization = (item.strip() for item in lines[0])
    used, free, utilization = int(used), int(free), int(utilization)
    idle = not process_busy and used < 512 and free > 22500 and utilization == 0
    return {"gpu_uuid": gpu, "idle": idle, "compute_process_present": process_busy,
            "used_mib": used, "free_mib": free, "gpu_utilization": utilization}


def observe():
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader"], text=True, timeout=20)
    stats = subprocess.check_output([
        "nvidia-smi", "--id=" + GPU, "--query-gpu=uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    return idle_observation(GPU, processes, stats)


def requested_runner_kwargs():
    return dict(
        base_model_path=MODEL, lora_path=None, tensor_parallel_size=1,
        gpu_memory_utilization=0.90, max_model_len=12288, max_tokens=2048,
        temperature=0.1, top_p=0.8, repeat_penalty=1.05, dtype="bfloat16",
        max_num_batched_tokens=4096, enforce_eager=True,
    )


def preflight():
    assert not PREFLIGHT.exists() and not OUTPUT.exists()
    test_counts = counts(TEST_XML)
    assert test_counts == {"tests": 13, "failures": 0, "errors": 0, "skipped": 0}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    old = load(HERE / "qwen3_8b_native_preflight_v1.json")
    matrix = load(MATRIX)
    assert sha(MATRIX) == old["matrix_sha256"] and len(matrix["rows"]) == 58
    assert matrix["evaluation_only"] and matrix["training_ready"] is False
    originals = [r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                 if r["renderer"] == "original_prompt_v1"]
    assert len(originals) == 58 and len({r["id"] for r in matrix["rows"]}) == 58
    assert sha(HERE / "fact_coverage_pairs116_v1.matrix.json") == matrix["source_matrix_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=False, local_files_only=True)
    rows = [check_row(row, original, tokenizer) for row, original in zip(matrix["rows"], originals)]
    before = old["source_hashes_at_preflight"]
    current = snapshot(SOURCE)
    changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
    assert changed == [
        "asa_arknight_story_agent/inference/grounding/quote_validation.py",
        "asa_arknight_story_agent/inference/grounding/token_checks.py",
    ], "unexpected runtime change since the old CPU-only preflight"
    assert current == load(HERE / "checkpoint_v18_state.json")["runtime_source_sha256"]
    current_assets = assets()
    assert current_assets == old["model_asset_manifest"], "model assets changed"
    assert digest(tokenizer.chat_template) == old["native_chat_template_sha256"]
    assert load(CONFIG)["generator"]["repeat_penalty"] == 1.05
    kwargs = requested_runner_kwargs()
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "matrix_sha256": sha(MATRIX), "rows": rows,
        "canonical_config_sha256": sha(CONFIG), "source_hashes": current,
        "changed_since_cpu_preflight": changed, "model_assets": current_assets,
        "dependencies_sha256": dependencies(), "harness_tests": test_counts,
        "harness_test_xml_sha256": sha(TEST_XML),
        "runner_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
        "decoding": {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8,
                     "repetition_penalty": 1.05, "seed": "same per-row integer as original4B"},
        "limitations": [
            "Development frozen-evidence deployment-candidate comparison, not fresh RAG or E2E accuracy.",
            "Native8B model/tokenizer/template without LoRA differs from configured4B plus adapter.",
            "Memory utilization0.90/bfloat16 is explicit for8B; the live4B config is unchanged.",
            "No output repair, evidence truncation, ground-truth leakage, generation retry or default-model change.",
            "Matching local assets attests local byte identity, not official model provenance.",
        ],
    }
    save(PREFLIGHT, value)
    print(json.dumps({"preflight_rows": len(rows), "max_input_tokens": max(r["input_tokens"] for r in rows),
                      "refreshed_source_files": changed, "gpu_model_loaded": False}), flush=True)


def wrap_dispatch(real_generate, row, expected_ids, receipts):
    def capture(prompts, params, **kwargs):
        assert prompts == [row["runtime_prompt"]]
        assert kwargs.get("lora_request") is None, "base-only evaluation must not use LoRA"
        assert params.max_tokens == 2048 and params.temperature == 0.1 and params.top_p == 0.8
        assert params.repetition_penalty == 1.05
        assert type(row["sampling_seed"]) is int
        params.seed = row["sampling_seed"]
        receipt = {"id": row["id"], "seed": params.seed, "lora_request": None,
                   "prompt_sha256": digest(prompts[0]), "status": "requested",
                   "max_tokens": params.max_tokens, "temperature": params.temperature,
                   "top_p": params.top_p, "repetition_penalty": params.repetition_penalty}
        receipts.append(receipt)
        outputs = real_generate(prompts, params, **kwargs)
        assert len(outputs) == 1 and len(outputs[0].outputs) == 1
        request = outputs[0]
        assert list(request.prompt_token_ids) == expected_ids, "input truncation/tokenization mismatch"
        assert getattr(request, "lora_request", None) is None
        receipt.update(status="returned", returned_lora_echo=None,
                       prompt_token_ids_sha256=digest(json.dumps(expected_ids)))
        capture.output = request
        return outputs
    capture.output = None
    return capture


def run():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
    value = load(PREFLIGHT)
    ensure_frozen(value)
    assert observe()["idle"], "target GPU became busy"
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    metadata = {"evaluation_only": True, "training_ready": False, "completed": False,
                "preflight_sha256": sha(PREFLIGHT), "gpu_uuid": GPU,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "actual_engine_constructor": None, "requests": [],
                "describe_runtime_warning": "Static release_lora_artifact fields are not actual LoRA usage."}
    save(OUTPUT / "metadata.started.json", metadata)
    try:
        sys.path.insert(0, str(SOURCE))
        import vllm
        import torch
        from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
        assert observe()["idle"], "target GPU became busy before initialization"
        real_llm = vllm.LLM

        def tracked_llm(**kwargs):
            metadata["actual_engine_constructor"] = dict(kwargs)
            assert kwargs["model"] == kwargs["tokenizer"] == str(MODEL)
            assert kwargs["enable_lora"] is False and kwargs["max_model_len"] == 12288
            assert kwargs["gpu_memory_utilization"] == 0.90 and kwargs["dtype"] == "bfloat16"
            return real_llm(**kwargs)

        vllm.LLM = tracked_llm
        try:
            runner = VllmRunner(**requested_runner_kwargs())
            engine, adapter = runner._ensure_engine()
        finally:
            vllm.LLM = real_llm
        assert adapter is None
        metadata.update(vllm_version=vllm.__version__, torch_version=torch.__version__,
                        describe_runtime=runner.describe_runtime())
        tokenizer = engine.get_tokenizer()
        originals = [r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                     if r["renderer"] == "original_prompt_v1"]
        matrix = load(MATRIX)
        checked = [check_row(r, o, tokenizer) for r, o in zip(matrix["rows"], originals)]
        assert checked == value["rows"], "actual tokenizer differs from CPU preflight"
        real_generate = engine.generate
        predictions = []
        with (OUTPUT / "predictions.progress.jsonl").open("x", encoding="utf-8") as journal:
            for row in matrix["rows"]:
                ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
                capture = wrap_dispatch(real_generate, row, ids, metadata["requests"])
                engine.generate = capture
                begin = time.monotonic()
                text = runner.generate(row["runtime_prompt"], max_tokens=2048, temperature=0.1,
                                       top_p=0.8, repeat_penalty=1.05)
                elapsed = time.monotonic() - begin
                request = capture.output
                assert request is not None
                completion = request.outputs[0]
                result = {
                    "id": row["id"], "source_id": row["source_id"], "renderer": row["renderer"],
                    "template": row["template"],
                    "input_row_sha256": digest(json.dumps(row, ensure_ascii=False, sort_keys=True)),
                    "actual_prompt": row["runtime_prompt"], "actual_prompt_sha256": digest(row["runtime_prompt"]),
                    "raw_output": text, "raw_output_with_special_tokens": completion.text,
                    "prompt_token_count": len(ids), "prompt_token_ids_sha256": digest(json.dumps(ids)),
                    "generated_token_count": len(completion.token_ids),
                    "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                    "generation_seconds": elapsed, "reserved_output_over_context": False,
                    "input_token_ids_match_untruncated_prompt": True,
                    "runtime_generation_diagnostics": dict(runner.last_generation_diagnostics),
                }
                predictions.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"completed": len(predictions), "total": 58,
                                  "finish": completion.finish_reason, "seconds": round(elapsed, 2)}), flush=True)
        ensure_frozen(value)
        assert assets() == value["model_assets"], "model assets changed during evaluation"
        save(OUTPUT / "predictions.json", predictions)
        assert len(predictions) == len(metadata["requests"]) == 58
        metadata.update(completed=True, predictions_sha256=sha(OUTPUT / "predictions.json"),
                        model_assets_after_equal=True, canonical_source_after_equal=True,
                        finish_reasons=dict(Counter(r["finish_reason"] for r in predictions)))
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(OUTPUT / "metadata.final.json", metadata)


def launch():
    assert not any(p.exists() for p in (OUTPUT, LOG, RECEIPT, VALIDATION))
    pre = load(PREFLIGHT)
    ensure_frozen(pre)
    assert sha(TEST_XML) == pre["harness_test_xml_sha256"]
    observations = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    lock_path = Path("/tmp/asa_slot_feedback_" + GPU + ".lock")
    with lock_path.open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Rehash weights before a fresh idle window; no GPU allocation yet.
        assert assets() == pre["model_assets"]
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
                ensure_frozen(pre)
                child = subprocess.Popen([sys.executable, str(Path(__file__)), "run"],
                                         cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                receipt = {"child_pid": child.pid, "returncode": code,
                           "gpu_uuid": GPU, "idle_observations": observations,
                           "preflight_sha256": sha(PREFLIGHT),
                           "ended_at_utc": datetime.now(timezone.utc).isoformat()}
                save(RECEIPT, receipt)
                print(json.dumps({"generation_returncode": code}), flush=True)
                if code:
                    return code
                ensure_frozen(pre)
                subprocess.run([
                    sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
                    "--validator", "after", "--source-root", str(SOURCE),
                    "--matrix", str(MATRIX), "--predictions", str(OUTPUT / "predictions.json"),
                    "--output", str(VALIDATION),
                ], check=True, env=dict(env, CUDA_VISIBLE_DEVICES=""))
                return 0
            time.sleep(10)
        save(RECEIPT, {"launched": False, "reason": "no_sustained_idle_window",
                       "idle_observations": observations, "evaluation_only": True})
        return 2


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("preflight", "launch", "run")
    if sys.argv[1] == "preflight":
        preflight()
    elif sys.argv[1] == "run":
        run()
    else:
        raise SystemExit(launch())
