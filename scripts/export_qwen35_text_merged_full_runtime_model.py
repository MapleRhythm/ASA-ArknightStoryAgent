#!/usr/bin/env python3
"""Export a vLLM-compatible full Qwen3.5 runtime model from a text-only merge.

Some local Qwen3.5 LoRA merges are saved as text-only CausalLM checkpoints with
model_type=qwen3_5_text. The local vLLM runtime expects the original full
Qwen3.5 multimodal shell (Qwen3_5ForConditionalGeneration), whose weights are
laid out as model.visual.* and model.language_model.*. This script copies the
text merged language weights into the original full base model and keeps the
original visual/config files so vLLM can load it without a LoRA adapter.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-full-base", type=Path, default=PROJECT_ROOT / "model" / "qwen3.5-4b")
    parser.add_argument("--text-merged-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="cpu", help="Use cpu for safest export, or auto/cuda with enough memory.")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("original full base", args.original_full_base),
        ("text merged model", args.text_merged_model),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)


def transplant_language_weights(full_model, text_model) -> tuple[int, list[str], list[str]]:
    text_state = text_model.state_dict()
    full_state = full_model.state_dict()
    copied = 0
    missing_in_text: list[str] = []
    unexpected_shape: list[str] = []

    for full_key, full_tensor in full_state.items():
        if full_key.startswith("model.language_model."):
            text_key = "model." + full_key[len("model.language_model.") :]
        elif full_key == "lm_head.weight":
            text_key = "lm_head.weight"
        else:
            continue

        text_tensor = text_state.get(text_key)
        if text_tensor is None:
            missing_in_text.append(f"{full_key} <= {text_key}")
            continue
        if tuple(text_tensor.shape) != tuple(full_tensor.shape):
            unexpected_shape.append(f"{full_key}: full={tuple(full_tensor.shape)} text={tuple(text_tensor.shape)}")
            continue
        full_tensor.copy_(text_tensor.to(device=full_tensor.device, dtype=full_tensor.dtype))
        copied += 1

    return copied, missing_in_text, unexpected_shape


def save_tokenizer_and_processor(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(args.original_full_base), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output_dir))
    try:
        processor = AutoProcessor.from_pretrained(str(args.original_full_base), trust_remote_code=True)
        processor.save_pretrained(str(args.output_dir))
    except Exception as exc:
        print(f"[warn] AutoProcessor save skipped: {type(exc).__name__}: {exc}", flush=True)

    for optional_name in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
    ):
        source = args.original_full_base / optional_name
        if source.exists():
            shutil.copy2(source, args.output_dir / optional_name)


def write_export_meta(args: argparse.Namespace, copied: int, missing: list[str], bad_shapes: list[str]) -> None:
    meta = {
        "original_full_base": str(args.original_full_base),
        "text_merged_model": str(args.text_merged_model),
        "copied_language_tensors": copied,
        "missing_language_tensors": missing,
        "shape_mismatches": bad_shapes,
    }
    (args.output_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    torch_dtype = resolve_dtype(args.dtype)

    print(f"[1/4] Loading text merged model: {args.text_merged_model}", flush=True)
    text_model = AutoModelForCausalLM.from_pretrained(
        str(args.text_merged_model),
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )

    print(f"[2/4] Loading original full base shell: {args.original_full_base}", flush=True)
    full_model = AutoModelForImageTextToText.from_pretrained(
        str(args.original_full_base),
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )

    print("[3/4] Transplanting language weights", flush=True)
    copied, missing, bad_shapes = transplant_language_weights(full_model, text_model)
    if missing or bad_shapes:
        raise RuntimeError(
            "Language weight transplant was incomplete. "
            f"copied={copied} missing={len(missing)} shape_mismatches={len(bad_shapes)}"
        )
    print(f"Copied {copied} language tensors.", flush=True)

    del text_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Saving full runtime model: {args.output_dir}", flush=True)
    full_model.save_pretrained(
        str(args.output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    save_tokenizer_and_processor(args)
    write_export_meta(args, copied, missing, bad_shapes)
    print("Export completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
