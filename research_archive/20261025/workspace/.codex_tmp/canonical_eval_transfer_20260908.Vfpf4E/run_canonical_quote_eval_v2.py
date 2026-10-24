#!/usr/bin/env python3
"""Run frozen canonical quote pairs with independently controlled tokenizer.

The original v1 runner and its completed artifacts remain frozen on 241.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from generate_exx_predictions_transformers import (
    generation_hit_token_limit, model_asset_manifest, sha256_file,
)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_input(row: dict[str, Any], tokenizer: Any) -> str:
    if text_sha(row["runtime_prompt"]) != row["runtime_prompt_sha256"]:
        raise ValueError("runtime prompt hash mismatch")
    if text_sha(row["evidence_brief"]) != row["evidence_sha256"]:
        raise ValueError("evidence hash mismatch")
    if row["template"] == "runtime_raw":
        return row["runtime_prompt"]
    if row["template"] != "canonical_non_thinking":
        raise ValueError("unknown template")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]},
         {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def assert_gpu_idle(gpu_uuid: str) -> str:
    if not gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("require CUDA_VISIBLE_DEVICES to match the explicit GPU UUID")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
    )
    if any(line.split(",")[0].strip() == gpu_uuid for line in result.stdout.splitlines()):
        raise ValueError("target GPU has a compute process; refusing to share or stop it")
    return subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}",
         "--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu",
         "--format=csv"], check=True, capture_output=True, text=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--reader-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-context", type=int, default=10000)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be absent or empty")
    matrix = json.loads(args.input.read_text(encoding="utf-8"))
    rows = matrix["rows"]
    if (not matrix.get("evaluation_only") or not rows
            or len({row["id"] for row in rows}) != len(rows)):
        parser.error("expected unique, nonempty evaluation-only rows")
    if args.adapter and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("adapter weights do not exist")
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(20260908)
    tokenizer_path = args.tokenizer_path or (
        args.adapter if args.adapter and (args.adapter / "tokenizer_config.json").exists()
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True,
    )
    prompts = [render_input(row, tokenizer) for row in rows]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "input_sha256": sha256_file(args.input),
        "tokenizer_path": str(tokenizer_path),
        "max_context": args.max_context, "requested_output_tokens": args.max_new_tokens,
        "max_input_tokens": max(lengths),
        "rows": [{"id": row["id"], "input_tokens": length,
                  "fits": length + args.max_new_tokens <= args.max_context}
                 for row, length in zip(rows, lengths)],
    }
    with (args.output_dir / "preflight.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"preflight_rows": len(rows), "max_input_tokens": max(lengths),
                      "over_budget": sum(not row["fits"] for row in preflight["rows"]),
                      "tokenizer_path": str(tokenizer_path)}), flush=True)
    if args.preflight_only:
        return 0
    if max(lengths) + args.max_new_tokens > args.max_context:
        parser.error("prompt plus output exceeds context; refusing to truncate evidence")
    gpu_before = assert_gpu_idle(args.gpu_uuid)
    print(gpu_before, flush=True)
    metadata = {
        "version": "asa_canonical_quote_inference_v2", "evaluation_only": True,
        "input_sha256": sha256_file(args.input),
        "runner_sha256": sha256_file(Path(__file__)),
        "helper_sha256": sha256_file(Path(__file__).with_name("generate_exx_predictions_transformers.py")),
        "base_model": str(args.base_model),
        "base_model_assets": model_asset_manifest(args.base_model),
        "reader_name": args.reader_name,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_assets": model_asset_manifest(args.adapter) if args.adapter else None,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_assets": {
            p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(tokenizer_path.iterdir())
            if p.is_file() and ("token" in p.name or p.name in {"vocab.json", "merges.txt", "chat_template.jinja"})
        },
        "gpu_before": gpu_before, "gpu_uuid": args.gpu_uuid,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "decoding": {"do_sample": False, "repetition_penalty": args.repetition_penalty,
                     "max_new_tokens": args.max_new_tokens, "dtype": "bfloat16",
                     "seed": 20260908, "max_context": args.max_context},
        "rows": len(rows), "max_input_tokens": max(lengths),
        "model_input_truncated": False,
        "limitations": ["Frozen evidence, no fresh retrieval.",
                        "Greedy Transformers controlled run, not deployed vLLM sampling E2E.",
                        "Runtime prompt and configured base/adapter/tokenizer paths are explicit."],
    }
    (args.output_dir / "metadata.started.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Recheck after asset hashing, before allocating model weights.
    assert_gpu_idle(args.gpu_uuid)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model), trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cuda", local_files_only=True,
    )
    if args.adapter:
        import peft
        model = peft.PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        metadata["peft"] = peft.__version__
    model.eval()
    predictions = []
    with (args.output_dir / "predictions.progress.jsonl").open("x", encoding="utf-8") as journal:
        for index, (row, prompt) in enumerate(zip(rows, prompts), 1):
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            completion = generated[0, prompt_tokens:]
            limited = generation_hit_token_limit(
                completion, max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(completion, skip_special_tokens=True).strip()
            result = {
                "id": row["id"], "source_id": row["source_id"],
                "renderer": row["renderer"], "template": row["template"],
                "input_row_sha256": text_sha(json.dumps(row, ensure_ascii=False, sort_keys=True)),
                "actual_prompt": prompt, "actual_prompt_sha256": text_sha(prompt),
                "raw_output": output,
                "raw_output_with_special_tokens": tokenizer.decode(completion, skip_special_tokens=False),
                "prompt_token_count": prompt_tokens, "generated_token_count": int(completion.shape[-1]),
                "finish_reason": "length" if limited else "stop",
                "generation_seconds": elapsed, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            predictions.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({
                "completed": index, "total": len(rows), "id": row["id"],
                "tokens": result["generated_token_count"],
                "seconds": round(elapsed, 2), "finish": result["finish_reason"],
            }), flush=True)
    output_path = args.output_dir / "predictions.json"
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata.update({
        "output_sha256": sha256_file(output_path),
        "completed_rows": len(predictions),
        "length_rows": sum(row["finish_reason"] == "length" for row in predictions),
        "generation_seconds_sum": sum(row["generation_seconds"] for row in predictions),
    })
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": len(predictions), "output_sha256": metadata["output_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
