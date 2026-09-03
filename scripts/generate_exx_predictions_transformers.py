#!/usr/bin/env python3
"""Generate deterministic fixed-evidence Exx predictions with Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = row.get("conversations") or []
    if not conversations:
        raise ValueError(f"row {row.get('id')} has no conversations")
    return [
        {"role": "system", "content": str(row.get("system") or "")},
        {"role": "user", "content": str(conversations[0].get("value") or "")},
    ]


def generation_hit_token_limit(
    completion_ids: Any, *, max_new_tokens: int, eos_token_id: int | list[int] | None
) -> bool:
    generated = int(completion_ids.shape[-1])
    if generated < max_new_tokens or generated == 0:
        return False
    eos_ids = (
        set(eos_token_id)
        if isinstance(eos_token_id, list)
        else ({eos_token_id} if eos_token_id is not None else set())
    )
    return int(completion_ids[-1].item()) not in eos_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if not args.gold.is_file() or not args.base_model.is_dir():
        parser.error("gold or base model path does not exist")
    adapter_model = args.adapter / "adapter_model.safetensors"
    if not adapter_model.is_file():
        parser.error(f"missing adapter weights: {adapter_model}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    rows = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("task_type") == "grounded_action_generation"]
    if args.limit:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model), trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()

    predictions = []
    truncated_rows = 0
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        messages = chat_messages(row)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completion_ids = generated[0, prompt_tokens:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        hit_max_new_tokens = generation_hit_token_limit(
            completion_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        truncated_rows += int(hit_max_new_tokens)
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
                "prompt_token_count": prompt_tokens,
                "generated_token_count": int(completion_ids.shape[-1]),
                "finish_reason": "length" if hit_max_new_tokens else "stop",
                "hit_max_new_tokens": hit_max_new_tokens,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(json.dumps({"completed": index, "total": len(rows)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "gold": str(args.gold),
        "gold_sha256": sha256_file(args.gold),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(adapter_model),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(predictions),
        "truncated_rows": truncated_rows,
        "seconds": time.monotonic() - started,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "evidence_truncated": False,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
