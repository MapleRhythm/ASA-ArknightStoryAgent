#!/usr/bin/env python3
"""Generate fixed-evidence Exx predictions for base and/or LoRA readers.

The input is a LLaMA-Factory JSON array. Only grounded_action_generation
rows are evaluated. Prompts are rendered by the checkpoint tokenizer's
canonical chat template with an explicit thinking mode. Qwen3.5's
non-thinking template includes an empty ``<think>...</think>`` block; merely
omitting a thinking preamble is not equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


READER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_asset_manifest(model_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in model_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"base model directory has no files: {model_dir}")
    records = []
    aggregate = hashlib.sha256()
    for path in files:
        digest = sha256_file(path)
        size = path.stat().st_size
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        records.append({"name": path.name, "bytes": size, "sha256": digest})
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = row.get("conversations") or []
    if not conversations:
        raise ValueError(f"row {row.get('id')} has no conversations")
    return [
        {"role": "system", "content": str(row.get("system") or "")},
        {"role": "user", "content": str(conversations[0].get("value") or "")},
    ]


def render_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    enable_thinking: bool = False,
) -> str:
    return str(
        tokenizer.apply_chat_template(
            chat_messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )


def validate_reader_name(name: str) -> str:
    if not READER_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "reader name must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return name


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must have NAME=/absolute/path form")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    validate_reader_name(name)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("adapter must have NAME=/absolute/path form")
    return name, path


def reader_specs(
    base_reader: str | None,
    adapters: list[tuple[str, Path]],
) -> list[tuple[str, Path | None]]:
    readers: list[tuple[str, Path | None]] = []
    if base_reader:
        readers.append((validate_reader_name(base_reader), None))
    readers.extend(adapters)
    names = [name for name, _ in readers]
    if len(names) != len(set(names)):
        raise ValueError("reader names must be unique")
    return readers


def prediction_row(
    *,
    row: dict[str, Any],
    text: str,
    prompt_token_count: int,
    generated_token_count: int,
    finish_reason: str,
) -> dict[str, Any]:
    result = {
        "id": row.get("id"),
        "task_type": row.get("task_type"),
        "system": row.get("system"),
        "conversations": [
            row["conversations"][0],
            {"from": "gpt", "value": text},
        ],
        "raw_output": text,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "finish_reason": finish_reason,
        "hit_max_new_tokens": finish_reason == "length",
    }
    if isinstance(row.get("diagnostic"), dict):
        result["diagnostic"] = row["diagnostic"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument(
        "--base-reader",
        type=validate_reader_name,
        help="Also evaluate the base model without a LoRA, using this output name.",
    )
    parser.add_argument("--adapter", action="append", default=[], type=parse_adapter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Explicitly enable the model's thinking template and reasoning mode.",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {args.output_dir}")
    if not args.gold.is_file() or not args.base_model.is_dir():
        parser.error("gold or base model path does not exist")
    try:
        readers = reader_specs(args.base_reader, args.adapter)
    except ValueError as exc:
        parser.error(str(exc))
    if not readers:
        parser.error("at least one --base-reader or --adapter is required")
    for name, path in args.adapter:
        if not (path / "adapter_model.safetensors").is_file():
            parser.error(f"missing adapter_model.safetensors for {name}: {path}")

    rows = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("task_type") == "grounded_action_generation"]
    if args.limit > 0:
        rows = rows[: args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    prompts = [
        render_prompt(row, tokenizer, enable_thinking=args.enable_thinking)
        for row in rows
    ]
    if not prompts:
        parser.error("no grounded_action_generation rows selected")
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
        "base_model_assets": model_asset_manifest(args.base_model),
        "base_reader": args.base_reader,
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
        "prompt_template": "tokenizer.apply_chat_template",
        "thinking": {
            "enabled": args.enable_thinking,
            "explicit": True,
        },
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
    engine_kwargs = dict(
        model=str(args.base_model),
        tokenizer=str(args.base_model),
        trust_remote_code=True,
        enable_lora=bool(args.adapter),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )
    if args.adapter:
        engine_kwargs.update(max_lora_rank=64, max_loras=1)
    llm = LLM(**engine_kwargs)
    engine_seconds = time.monotonic() - started
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
        skip_special_tokens=False,
    )

    timings = {"engine_init_seconds": engine_seconds, "readers": {}}
    adapter_ids = {
        name: adapter_id
        for adapter_id, (name, _) in enumerate(args.adapter, start=1)
    }
    for name, path in readers:
        adapter_started = time.monotonic()
        generation_kwargs = {}
        if path is not None:
            generation_kwargs["lora_request"] = LoRARequest(
                name, adapter_ids[name], str(path)
            )
        outputs = llm.generate(prompts, sampling, use_tqdm=True, **generation_kwargs)
        predictions = []
        truncated_rows = 0
        for row, prompt, output in zip(rows, prompts, outputs, strict=True):
            text = output.outputs[0].text if output.outputs else ""
            finish_reason = output.outputs[0].finish_reason if output.outputs else "empty"
            hit_max_tokens = finish_reason == "length"
            truncated_rows += int(hit_max_tokens)
            predictions.append(
                prediction_row(
                    row=row,
                    text=text,
                    prompt_token_count=len(
                        tokenizer(prompt, add_special_tokens=False).input_ids
                    ),
                    generated_token_count=(
                        len(output.outputs[0].token_ids) if output.outputs else 0
                    ),
                    finish_reason=finish_reason,
                )
            )
        output_path = args.output_dir / f"{name}.predictions.json"
        output_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        timings["readers"][name] = {
            "seconds": time.monotonic() - adapter_started,
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "truncated_rows": truncated_rows,
        }
        print(json.dumps({"completed": name, **timings["readers"][name]}, ensure_ascii=False))

    timings["total_seconds"] = time.monotonic() - started
    (args.output_dir / "timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(timings, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
