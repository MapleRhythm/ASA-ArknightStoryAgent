"""Frozen-evidence replay through 241's actual VllmRunner and configured engine.

No retrieval or service restart. The runner's actual request output is retained,
including finish reason and input token IDs, instead of guessed from text.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from generate_exx_predictions_transformers import model_asset_manifest, sha256_file
from run_canonical_quote_eval_v2 import assert_gpu_idle, text_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=0)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    source_root = args.source_root or run / "after/src"
    config_path = run / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    config = json.loads(config_path.read_text())["generator"]
    engine_config = config["vllm"]
    release = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent")
    base_path = release / engine_config["base_model_path"]
    adapter_path = release / engine_config["lora_path"]
    matrix = json.loads(args.input.read_text())
    rows = matrix["rows"]
    if (not matrix.get("evaluation_only") or not rows
            or len({r["id"] for r in rows}) != len(rows)
            or any(r["template"] != "runtime_raw" for r in rows)):
        parser.error("expected unique runtime-raw evaluation rows")
    if args.output_dir.exists():
        parser.error("output directory exists; use a new artifact")
    args.output_dir.mkdir(parents=True)
    gpu_before = assert_gpu_idle(args.gpu_uuid)
    metadata = {
        "version": "canonical_vllm_configured_v3", "evaluation_only": True, "training_ready": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_matrix_sha256": sha256_file(args.input), "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(run / "manifest.json"),
        "runner_sha256": sha256_file(Path(__file__)), "gpu_before": gpu_before,
        "base_model": str(base_path), "adapter": str(adapter_path),
        "base_model_assets": model_asset_manifest(base_path), "adapter_assets": model_asset_manifest(adapter_path),
        "requested_context_override": args.max_context or None,
        "runtime_source_root": str(source_root),
        "runtime_runner_sha256": sha256_file(
            source_root / "asa_arknight_story_agent/inference/model_runtime/vllm_runner.py"
        ),
        "explicit_environment": {
            name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "TOKENIZERS_PARALLELISM",
            )
        },
        "decoding": {"temperature": 0.1, "top_p": 0.8, "repeat_penalty": config["repeat_penalty"],
                     "max_new_tokens": args.max_new_tokens, "seed": "runtime engine default"},
        "limitations": [
            "Real configured vLLM generation on frozen evidence, not fresh retrieval or production E2E.",
            "Sampling differs from the greedy Transformers experiment; do not merge their denominators.",
            "No prompt truncation requested and actual input token counts are checked.",
        ],
    }
    started_path = args.output_dir / "metadata.started.json"
    with started_path.open("x") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    sys.path.insert(0, str(source_root))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    import vllm
    import torch
    assert_gpu_idle(args.gpu_uuid)
    runner = VllmRunner(
        base_model_path=base_path, lora_path=adapter_path,
        tensor_parallel_size=engine_config["tensor_parallel_size"],
        gpu_memory_utilization=engine_config["gpu_memory_utilization"],
        max_model_len=args.max_context or engine_config["max_model_len"],
        max_tokens=config["max_tokens"], temperature=config["temperature"],
        top_p=config["top_p"], repeat_penalty=config["repeat_penalty"],
        dtype=engine_config["dtype"], max_num_batched_tokens=engine_config["max_num_batched_tokens"],
        enforce_eager=engine_config["enforce_eager"],
    )
    metadata.update(vllm=vllm.__version__, torch=torch.__version__, runtime=runner.describe_runtime())
    try:
        llm, _ = runner._ensure_engine()
    except Exception as exc:
        metadata.update(status="engine_initialization_failed", error=f"{type(exc).__name__}: {exc}")
        with (args.output_dir / "metadata.failed.json").open("x") as f:
            f.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        raise
    tokenizer = llm.get_tokenizer()
    original_generate = llm.generate
    captured = []
    def capture_generate(*call_args, **kwargs):
        outputs = original_generate(*call_args, **kwargs)
        captured.extend(outputs)
        return outputs
    llm.generate = capture_generate
    results = []
    with (args.output_dir / "predictions.progress.jsonl").open("x") as journal:
        for row in rows:
            prompt = row["runtime_prompt"]
            if text_sha(prompt) != row["runtime_prompt_sha256"] or text_sha(row["evidence_brief"]) != row["evidence_sha256"]:
                raise ValueError("frozen prompt hash mismatch")
            expected_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(expected_ids) >= runner.max_model_len:
                raise ValueError("input itself exhausts context; no truncation permitted")
            captured.clear()
            begin = time.monotonic()
            text = runner.generate(
                prompt, max_tokens=args.max_new_tokens,
                temperature=0.1, top_p=0.8, repeat_penalty=config["repeat_penalty"],
            )
            elapsed = time.monotonic() - begin
            if len(captured) != 1 or len(captured[0].outputs) != 1:
                raise ValueError("unexpected vLLM request/output count")
            request = captured[0]
            completion = request.outputs[0]
            if list(request.prompt_token_ids) != list(expected_ids):
                raise ValueError("vLLM input token IDs differ from exact untruncated tokenizer input")
            result = {
                "id": row["id"], "source_id": row["source_id"],
                "renderer": row["renderer"], "template": row["template"],
                "input_row_sha256": text_sha(json.dumps(row, ensure_ascii=False, sort_keys=True)),
                "actual_prompt": prompt, "actual_prompt_sha256": text_sha(prompt),
                "raw_output": text, "raw_output_with_special_tokens": completion.text,
                "prompt_token_count": len(request.prompt_token_ids),
                "prompt_token_ids_sha256": hashlib.sha256(json.dumps(list(request.prompt_token_ids)).encode()).hexdigest(),
                "generated_token_count": len(completion.token_ids),
                "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                "generation_seconds": elapsed,
                "reserved_output_over_context": len(expected_ids) + args.max_new_tokens > runner.max_model_len,
                "input_token_ids_match_untruncated_prompt": True,
                "runtime_generation_diagnostics": dict(getattr(runner, "last_generation_diagnostics", {})),
            }
            results.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({"completed": len(results), "total": len(rows), "id": row["id"],
                              "finish": completion.finish_reason, "tokens": len(completion.token_ids),
                              "seconds": round(elapsed, 2)}), flush=True)
    with (args.output_dir / "predictions.json").open("x") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    metadata.update(
        status="completed", completed_rows=len(results),
        predictions_sha256=sha256_file(args.output_dir / "predictions.json"),
        length_outputs=sum(r["finish_reason"] == "length" for r in results),
        input_truncated=False,
    )
    with (args.output_dir / "metadata.json").open("x") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": len(results), "predictions_sha256": metadata["predictions_sha256"]}))


if __name__ == "__main__":
    main()
