#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into the base model.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(args.dtype)
    device_map = args.device if args.device in {"cpu", "auto"} else "auto"

    print(f"Loading base model from {args.base_model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter from {args.lora_path}", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(args.lora_path))
    print("Merging adapter into base model", flush=True)
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}", flush=True)
    merged_model.save_pretrained(
        str(args.output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output_dir))

    for optional_name in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
        "video_preprocessor_config.json",
    ):
        source = args.base_model / optional_name
        if source.exists():
            target = args.output_dir / optional_name
            if not target.exists():
                target.write_bytes(source.read_bytes())

    print("Merge completed.", flush=True)


if __name__ == "__main__":
    main()
