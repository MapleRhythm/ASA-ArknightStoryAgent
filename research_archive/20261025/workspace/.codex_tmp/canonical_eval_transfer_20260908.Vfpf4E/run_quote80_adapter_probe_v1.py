"""Existing-adapter-only probe with exact control prompts, tokens and seeds."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from generate_exx_predictions_transformers import model_asset_manifest
from run_canonical_quote_eval_v2 import assert_gpu_idle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def token_sha(ids):
    return text_sha(json.dumps(list(ids)))


def assert_control_input(row, token_ids):
    if token_sha(token_ids) != row["control_prompt_token_ids_sha256"]:
        raise ValueError("Candidate tokenizer changed actual input tokens relative to control")
    if len(token_ids) != row["control_prompt_token_count"]:
        raise ValueError("Control input token count mismatch")
    if text_sha(row["runtime_prompt"]) != row["runtime_prompt_sha256"]:
        raise ValueError("Frozen prompt hash mismatch")
    if text_sha(row["evidence_brief"]) != row["evidence_sha256"]:
        raise ValueError("Frozen evidence hash mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    source = run / "inferred_integrity_candidate_v2/src"
    manifest_path = run / "quote80_sft_adapter_probe_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    matrix_path = run / "quote80_sft_adapter_probe9_v1.matrix.json"
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["rows"]
    assert matrix["evaluation_only"] and len(rows) == len({r["id"] for r in rows}) == 9
    assert matrix["manifest_sha256"] == sha(manifest_path)
    output_dir = run / "quote80_sft_adapter_probe9_ctx12288_v1"
    if output_dir.exists():
        parser.error("Output exists; do not overwrite or duplicate")
    adapter = Path(manifest["selected_adapter"])
    base = Path(manifest["base_model"])
    assert sha(adapter / "adapter_model.safetensors") == manifest["adapter_weights_sha256"]
    base_assets = model_asset_manifest(base)
    assert base_assets == manifest["control_base_assets"], "Base model differs from control"
    gpu_before = assert_gpu_idle(args.gpu_uuid)
    config_path = run / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    config = json.loads(config_path.read_text())["generator"]
    engine_config = config["vllm"]
    metadata = {
        "version": "quote80_adapter_probe_v1", "evaluation_only": True, "training_ready": False,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_matrix_sha256": sha(matrix_path), "probe_manifest_sha256": sha(manifest_path),
        "base_model": str(base), "adapter": str(adapter),
        "base_model_assets": base_assets, "adapter_assets": model_asset_manifest(adapter),
        "frozen_default_config_sha256": sha(config_path),
        "explicit_override": "Only evaluated adapter; production config/pointer is not changed.",
        "context": 12288, "output_budget": 2048, "temperature": 0.1, "top_p": 0.8,
        "repeat_penalty": config["repeat_penalty"], "seed": "Per-row seed matched to frozen control",
        "gpu_before": gpu_before, "runtime_source_root": str(source),
        "runtime_runner_sha256": sha(source / "asa_arknight_story_agent/inference/model_runtime/vllm_runner.py"),
        "probe_runner_sha256": sha(Path(__file__)),
        "environment": {key: os.environ.get(key) for key in
                        ("CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "TOKENIZERS_PARALLELISM")},
    }
    output_dir.mkdir()
    with (output_dir / "metadata.started.json").open("x") as output:
        output.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    import vllm
    import torch
    assert_gpu_idle(args.gpu_uuid)
    runner = VllmRunner(
        base_model_path=base, lora_path=adapter,
        tensor_parallel_size=engine_config["tensor_parallel_size"],
        gpu_memory_utilization=engine_config["gpu_memory_utilization"],
        max_model_len=12288, max_tokens=config["max_tokens"],
        temperature=config["temperature"], top_p=config["top_p"], repeat_penalty=config["repeat_penalty"],
        dtype=engine_config["dtype"], max_num_batched_tokens=engine_config["max_num_batched_tokens"],
        enforce_eager=engine_config["enforce_eager"],
    )
    metadata.update(vllm=vllm.__version__, torch=torch.__version__, runtime=runner.describe_runtime())
    try:
        engine, _ = runner._ensure_engine()
    except Exception as exc:
        metadata.update(status="engine_initialization_failed", error=f"{type(exc).__name__}: {exc}")
        with (output_dir / "metadata.failed.json").open("x") as output:
            output.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        raise
    tokenizer = engine.get_tokenizer()
    token_ids = [tokenizer.encode(row["runtime_prompt"], add_special_tokens=True) for row in rows]
    for row, ids in zip(rows, token_ids):
        assert_control_input(row, ids)
        assert len(ids) + 2048 <= 12288
    original_generate = engine.generate
    calls, captured = [], []

    def seeded_capture(prompts, params, **kwargs):
        row = rows[len(calls)]
        assert prompts == [row["runtime_prompt"]]
        params.seed = row["sampling_seed"]
        calls.append({"id": row["id"], "seed": params.seed})
        output = original_generate(prompts, params, **kwargs)
        captured.extend(output)
        return output

    engine.generate = seeded_capture
    predictions = []
    with (output_dir / "predictions.progress.jsonl").open("x") as journal:
        for row, ids in zip(rows, token_ids):
            captured.clear()
            begin = time.monotonic()
            text = runner.generate(row["runtime_prompt"], max_tokens=2048, temperature=0.1,
                                   top_p=0.8, repeat_penalty=config["repeat_penalty"])
            seconds = time.monotonic() - begin
            assert len(captured) == 1 and len(captured[0].outputs) == 1
            request, completion = captured[0], captured[0].outputs[0]
            assert list(request.prompt_token_ids) == ids
            result = {
                "id": row["id"], "source_id": row["source_id"], "renderer": row["renderer"],
                "template": row["template"],
                "input_row_sha256": text_sha(json.dumps(row, ensure_ascii=False, sort_keys=True)),
                "actual_prompt": row["runtime_prompt"], "actual_prompt_sha256": text_sha(row["runtime_prompt"]),
                "raw_output": text, "raw_output_with_special_tokens": completion.text,
                "prompt_token_count": len(ids), "prompt_token_ids_sha256": token_sha(ids),
                "generated_token_count": len(completion.token_ids), "generation_seconds": seconds,
                "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                "sampling_seed": row["sampling_seed"],
                "input_token_ids_match_untruncated_prompt": True, "input_token_ids_match_control": True,
                "reserved_output_over_context": False,
                "runtime_generation_diagnostics": dict(runner.last_generation_diagnostics),
            }
            predictions.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({"completed": len(predictions), "total": 9, "id": row["id"],
                              "finish": completion.finish_reason, "tokens": len(completion.token_ids)}), flush=True)
    assert sha(adapter / "adapter_model.safetensors") == manifest["adapter_weights_sha256"]
    with (output_dir / "predictions.json").open("x") as output:
        output.write(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
    metadata.update(
        status="completed", completed_rows=len(predictions), input_truncated=False,
        predictions_sha256=sha(output_dir / "predictions.json"), seed_calls=calls,
        all_input_token_ids_match_control=True,
    )
    with (output_dir / "metadata.json").open("x") as output:
        output.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed_rows": len(predictions), "production_model_changed": False}))


if __name__ == "__main__":
    main()
