#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
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


def model_module_names(model) -> set[str]:
    return {name for name, _ in model.named_modules()}


def should_normalize_language_model_prefix(model, adapter_dir: Path) -> bool:
    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        return False

    module_names = model_module_names(model)
    if "model.layers.0" not in module_names or "model.language_model.layers.0" in module_names:
        return False

    with safe_open(adapter_model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

    return any(".language_model.layers." in key for key in keys)


def normalize_adapter_language_model_prefix(adapter_dir: Path, tmp_root: Path) -> Path:
    tmp_adapter_dir = tmp_root / adapter_dir.name
    shutil.copytree(adapter_dir, tmp_adapter_dir, ignore=shutil.ignore_patterns("adapter_model.safetensors"))

    source_model_path = adapter_dir / "adapter_model.safetensors"
    target_model_path = tmp_adapter_dir / "adapter_model.safetensors"
    tensors = load_file(source_model_path)
    normalized_tensors = {
        key.replace("base_model.model.model.language_model.", "base_model.model.model."): tensor
        for key, tensor in tensors.items()
    }
    save_file(normalized_tensors, target_model_path, metadata={"format": "pt"})
    return tmp_adapter_dir


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
    with tempfile.TemporaryDirectory(prefix="asa_lora_merge_") as tmp_dir:
        lora_path = args.lora_path
        if should_normalize_language_model_prefix(base_model, args.lora_path):
            print("Normalizing multimodal language_model LoRA prefix for text-only merge", flush=True)
            lora_path = normalize_adapter_language_model_prefix(args.lora_path, Path(tmp_dir))

        print(f"Loading LoRA adapter from {lora_path}", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, str(lora_path))
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
        "generation_config.json",
    ):
        source = args.base_model / optional_name
        if source.exists():
            target = args.output_dir / optional_name
            target.write_bytes(source.read_bytes())

    # AutoModelForCausalLM may save a text-only Qwen3.5 config. Keep that config
    # instead of copying the top-level multimodal config from the original base:
    # a text-only checkpoint with a multimodal config makes vLLM look for
    # missing visual.* weights.
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") == "qwen3_5_text":
            for optional_name in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
                optional_path = args.output_dir / optional_name
                if optional_path.exists():
                    optional_path.unlink()

    readme_source = args.base_model / "README.md"
    if readme_source.exists():
        shutil.copy2(readme_source, args.output_dir / "README.base.md")

    print("Merge completed.", flush=True)


if __name__ == "__main__":
    main()
