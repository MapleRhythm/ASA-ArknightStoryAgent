#!/usr/bin/env python3
"""Isolated single-GPU nonthinking runner with streaming, content-bound outputs."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import generate_exx_predictions_vllm as common
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import generate_exx_predictions_vllm as common


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reader", required=True, type=common.validate_reader_name)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist; preserve failed attempts")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = open(f"/tmp/asa_scope_gpu_{args.gpu}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = subprocess.run(["nvidia-smi", "-i", str(args.gpu),
                               "--query-gpu=uuid,name,memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    gpu_fields = [part.strip() for part in preflight.stdout.strip().split(",")]
    if int(gpu_fields[-2]) > 200 or int(gpu_fields[-1]) > 5:
        raise RuntimeError(f"GPU not idle; no sharing authorized by this runner: {preflight.stdout}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    started = time.monotonic()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = json.loads(args.input.read_text())
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("duplicate input IDs")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    prompts = [common.render_prompt(row, tokenizer, enable_thinking=False) for row in rows]
    lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False).input_ids]
    if not all("<think>\n\n</think>" in prompt for prompt in prompts):
        raise ValueError("missing canonical empty nonthinking block")
    if max(lengths) + args.max_tokens > args.max_model_len:
        raise ValueError("input budget exceeded; never truncate evidence")
    save(args.output / "config.json", {
        "reader": args.reader, "input_sha256": common.sha256_file(args.input),
        "selected_rows": len(rows), "started_at": datetime.datetime.now().astimezone().isoformat(),
        "runner_sha256": common.sha256_file(Path(__file__)), "common_runner_sha256": common.sha256_file(Path(common.__file__)),
        "model": str(args.model), "base_model_assets": common.model_asset_manifest(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": common.sha256_file(args.adapter / "adapter_model.safetensors") if args.adapter else None,
        "adapter_config_sha256": common.sha256_file(args.adapter / "adapter_config.json") if args.adapter else None,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "peft")},
        "gpu_preflight": preflight.stdout, "nonthinking": True, "input_truncation": False,
        "prompt_token_min": min(lengths), "prompt_token_max": max(lengths),
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "temperature": 0, "seed": 42, "batch_size": args.batch_size,
        "dtype": "bfloat16", "gpu_memory_utilization": 0.65, "enforce_eager": True,
    })
    with (args.output / "rendered_prompts.jsonl").open("w") as handle:
        for row, prompt, length in zip(rows, prompts, lengths, strict=True):
            handle.write(json.dumps({"id": row["id"], "prompt": prompt, "sha256": digest(prompt),
                                     "tokens": length}, ensure_ascii=False) + "\n")
    kwargs = {}
    if args.adapter:
        kwargs = {"enable_lora": True, "max_lora_rank": 64, "max_loras": 1}
    engine = LLM(model=str(args.model), tokenizer=str(args.model), dtype="bfloat16",
                 max_model_len=args.max_model_len, gpu_memory_utilization=0.65,
                 max_num_seqs=args.batch_size, max_num_batched_tokens=8192,
                 enforce_eager=True, seed=42, disable_log_stats=True, **kwargs)
    sampling = SamplingParams(temperature=0, top_p=1, repetition_penalty=1,
                              max_tokens=args.max_tokens, stop=["<|im_end|>", "<|endoftext|>"],
                              skip_special_tokens=False)
    generation_args = {"lora_request": LoRARequest(args.reader, 1, str(args.adapter))} if args.adapter else {}
    predictions = []
    with (args.output / "predictions.jsonl").open("w") as stream:
        for offset in range(0, len(rows), args.batch_size):
            outputs = engine.generate(prompts[offset:offset + args.batch_size], sampling,
                                      use_tqdm=False, **generation_args)
            for row, prompt, expected_length, output in zip(
                rows[offset:offset + args.batch_size], prompts[offset:offset + args.batch_size],
                lengths[offset:offset + args.batch_size], outputs, strict=True
            ):
                candidate = output.outputs[0]
                if len(output.prompt_token_ids) != expected_length:
                    raise ValueError("engine/tokenizer token count disagreement")
                pred = common.prediction_row(row=row, text=candidate.text,
                                              prompt_token_count=expected_length,
                                              generated_token_count=len(candidate.token_ids),
                                              finish_reason=candidate.finish_reason)
                pred.update({"reader": args.reader, "rendered_prompt_sha256": digest(prompt),
                             "raw_output_sha256": digest(candidate.text),
                             "generated_at": datetime.datetime.now().astimezone().isoformat()})
                predictions.append(pred)
                stream.write(json.dumps(pred, ensure_ascii=False) + "\n")
            stream.flush()
            save(args.output / "progress.json", {"completed": len(predictions), "total": len(rows),
                                                "elapsed_seconds": time.monotonic() - started})
            print(f"completed {len(predictions)}/{len(rows)}", flush=True)
    save(args.output / "predictions.json", predictions)
    save(args.output / "completion.json", {
        "status": "complete", "rows": len(predictions), "elapsed_seconds": time.monotonic() - started,
        "truncated_rows": sum(row["hit_max_new_tokens"] for row in predictions),
        "predictions_sha256": common.sha256_file(args.output / "predictions.json"),
        "completed_at": datetime.datetime.now().astimezone().isoformat(),
    })


if __name__ == "__main__":
    main()
