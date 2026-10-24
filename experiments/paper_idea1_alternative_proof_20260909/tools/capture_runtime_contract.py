#!/usr/bin/env python3
"""Capture the exact non-thinking prompt contract without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")

    import torch
    import transformers
    from transformers import AutoTokenizer

    rows = json.loads(args.matrix.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model), trust_remote_code=True
    )
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": str(row.get("system") or "")},
            {
                "role": "user",
                "content": str(row["conversations"][0].get("value") or ""),
            },
        ]
        prompts.append(
            str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        )
    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    assistant_suffixes = [
        prompt.rsplit("<|im_start|>assistant", 1)[-1] for prompt in prompts
    ]
    empty_think_present = [
        "<think>\n\n</think>" in suffix for suffix in assistant_suffixes
    ]
    if not all(empty_think_present):
        raise ValueError("canonical non-thinking empty think block is absent")
    if max(lengths) + args.max_tokens > args.max_model_len:
        raise ValueError("prompt plus output budget would require truncation")

    result = {
        "schema_version": "asa_nonthinking_runtime_contract_v1",
        "matrix": {
            "path": str(args.matrix),
            "sha256": sha256_file(args.matrix),
            "rows": len(rows),
        },
        "base_model": str(args.base_model),
        "runner": {
            "path": str(args.runner),
            "sha256": sha256_file(args.runner),
        },
        "prompt_rendering": {
            "api": "tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "empty_think_block_present_for_all_rows": all(empty_think_present),
            "prompt_sha256": [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in prompts
            ],
            "prompt_tokens": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "evidence_or_prompt_truncation_required": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

