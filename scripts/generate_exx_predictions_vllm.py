#!/usr/bin/env python3
"""Generate fixed-evidence Exx predictions for one or more LoRA adapters.

The input is a LLaMA-Factory JSON array. Only grounded_action_generation
rows are evaluated. Prompts exactly follow the qwen3_nothink formatter used
during SFT: system and user messages followed by an empty assistant prefix,
without a thinking preamble.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(row: dict) -> str:
    conversations = row.get("conversations") or []
    if not conversations:
        raise ValueError(f"row {row.get('id')} has no conversations")
    user = str(conversations[0].get("value") or "")
    system = str(row.get("system") or "")
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must have NAME=/absolute/path form")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_absolute():
        raise argparse.ArgumentTypeError("adapter must have NAME=/absolute/path form")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", action="append", required=True, type=parse_adapter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {args.output_dir}")
    if not args.gold.is_file() or not args.base_model.is_dir():
        parser.error("gold or base model path does not exist")
    for name, path in args.adapter:
        if not (path / "adapter_model.safetensors").is_file():
            parser.error(f"missing adapter_model.safetensors for {name}: {path}")

    rows = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("task_type") == "grounded_action_generation"]
    if args.limit > 0:
        rows = rows[: args.limit]
    prompts = [render_prompt(row) for row in rows]
    if not prompts:
        parser.error("no grounded_action_generation rows selected")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    prompt_lengths = [len(tokens) for tokens in tokenizer(prompts, add_special_tokens=False).input_ids]
    longest = max(prompt_lengths)
    if longest + args.max_tokens > args.max_model_len:
        parser.error(
            f"longest prompt {longest} + max_tokens {args.max_tokens} exceeds "
            f"max_model_len {args.max_model_len}; evidence will not be truncated"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "gold": str(args.gold),
        "gold_sha256": sha256_file(args.gold),
        "base_model": str(args.base_model),
        "adapters": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path / "adapter_model.safetensors"),
            }
            for name, path in args.adapter
        ],
        "selected_rows": len(rows),
        "selection": "task_type == grounded_action_generation; source order preserved",
        "prompt_template": "qwen3_nothink",
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_tokens": args.max_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
        "prompt_tokens": {
            "min": min(prompt_lengths),
            "max": longest,
            "mean": sum(prompt_lengths) / len(prompt_lengths),
        },
        "engine": {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "enforce_eager": args.enforce_eager,
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    started = time.monotonic()
    llm = LLM(
        model=str(args.base_model),
        tokenizer=str(args.base_model),
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )
    engine_seconds = time.monotonic() - started
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
        skip_special_tokens=False,
    )

    timings = {"engine_init_seconds": engine_seconds, "adapters": {}}
    for adapter_id, (name, path) in enumerate(args.adapter, start=1):
        adapter_started = time.monotonic()
        outputs = llm.generate(
            prompts,
            sampling,
            use_tqdm=True,
            lora_request=LoRARequest(name, adapter_id, str(path)),
        )
        predictions = []
        truncated_rows = 0
        for row, prompt, output in zip(rows, prompts, outputs, strict=True):
            text = output.outputs[0].text if output.outputs else ""
            finish_reason = output.outputs[0].finish_reason if output.outputs else "empty"
            hit_max_tokens = finish_reason == "length"
            truncated_rows += int(hit_max_tokens)
            predictions.append(
                {
                    "id": row.get("id"),
                    "task_type": row.get("task_type"),
                    "system": row.get("system"),
                    "conversations": [
                        row["conversations"][0],
                        {"from": "gpt", "value": text},
                    ],
                    "raw_output": text,
                    "prompt_token_count": len(
                        tokenizer(prompt, add_special_tokens=False).input_ids
                    ),
                    "generated_token_count": len(output.outputs[0].token_ids)
                    if output.outputs
                    else 0,
                    "finish_reason": finish_reason,
                    "hit_max_new_tokens": hit_max_tokens,
                }
            )
        output_path = args.output_dir / f"{name}.predictions.json"
        output_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        timings["adapters"][name] = {
            "seconds": time.monotonic() - adapter_started,
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "truncated_rows": truncated_rows,
        }
        print(json.dumps({"completed": name, **timings["adapters"][name]}, ensure_ascii=False))

    timings["total_seconds"] = time.monotonic() - started
    (args.output_dir / "timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(timings, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
