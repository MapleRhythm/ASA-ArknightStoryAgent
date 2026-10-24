#!/usr/bin/env python3
"""Generate deterministic fixed-evidence Exx predictions with Transformers.

The same entry point supports either the untouched base reader or one LoRA
adapter.  This is the low-dependency fallback for hosts where vLLM is not
installed.
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


def prediction_row(
    *,
    row: dict[str, Any],
    text: str,
    prompt_token_count: int,
    generated_token_count: int,
    hit_max_new_tokens: bool,
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
        "finish_reason": "length" if hit_max_new_tokens else "stop",
        "hit_max_new_tokens": hit_max_new_tokens,
    }
    for name in ("diagnostic", "state_provenance", "meta"):
        if name in row:
            result[name] = row[name]
    result["input_row_sha256"] = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional LoRA adapter. Omit it to evaluate the untouched base reader.",
    )
    parser.add_argument(
        "--reader-name",
        help="Stable reader label recorded in metadata.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--gpu-uuid", help="Refuse to run unless this explicitly selected GPU is idle.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if not args.gold.is_file() or not args.base_model.is_dir():
        parser.error("gold or base model path does not exist")
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if metadata_path.exists():
        parser.error(f"refusing to overwrite metadata: {metadata_path}")
    adapter_model = (
        args.adapter / "adapter_model.safetensors"
        if args.adapter is not None
        else None
    )
    if adapter_model is not None and not adapter_model.is_file():
        parser.error(f"missing adapter weights: {adapter_model}")

    if args.max_context < 1 or args.max_new_tokens < 1:
        parser.error("context and output budgets must be positive")
    gpu_before = assert_gpu_idle(args.gpu_uuid) if args.gpu_uuid else None
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    rows = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("task_type") == "grounded_action_generation"]
    if args.limit:
        rows = rows[: args.limit]
    if not rows or len({row.get("id") for row in rows}) != len(rows) or any(not row.get("id") for row in rows):
        parser.error("rows must be nonempty with unique IDs")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            chat_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for row in rows
    ]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if max(lengths) + args.max_new_tokens > args.max_context:
        parser.error("prompt plus output exceeds context; refusing to truncate evidence")
    assets = model_asset_manifest(args.base_model)
    if args.gpu_uuid:
        assert_gpu_idle(args.gpu_uuid)
    torch.manual_seed(20260908)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model), trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(args.adapter), is_trainable=False
        )
    model.eval()

    predictions = []
    truncated_rows = 0
    started = time.monotonic()
    for index, (row, actual_prompt) in enumerate(zip(rows, prompts), start=1):
        encoded = tokenizer(actual_prompt, add_special_tokens=False, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                repetition_penalty=1.0,
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
            prediction_row(
                row=row,
                text=text,
                prompt_token_count=prompt_tokens,
                generated_token_count=int(completion_ids.shape[-1]),
                hit_max_new_tokens=hit_max_new_tokens,
            )
        )
        predictions[-1]["actual_prompt"] = actual_prompt
        predictions[-1]["actual_prompt_sha256"] = hashlib.sha256(actual_prompt.encode()).hexdigest()
        predictions[-1]["raw_output_with_special_tokens"] = tokenizer.decode(completion_ids, skip_special_tokens=False)
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(json.dumps({"completed": index, "total": len(rows)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "gold": str(args.gold),
        "gold_sha256": sha256_file(args.gold),
        "base_model": str(args.base_model),
        "base_model_assets": assets,
        "reader_name": args.reader_name
        or ("lora_adapter" if args.adapter is not None else "base"),
        "reader_type": "lora_adapter" if args.adapter is not None else "base",
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "adapter_sha256": (
            sha256_file(adapter_model) if adapter_model is not None else None
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(predictions),
        "truncated_rows": truncated_rows,
        "seconds": time.monotonic() - started,
        "decoding": {
            "do_sample": False, "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context, "repetition_penalty": 1.0,
            "dtype": "bfloat16", "enable_thinking": False, "seed": 20260908,
        },
        "gpu_uuid": args.gpu_uuid, "gpu_before": gpu_before,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "runner_sha256": sha256_file(Path(__file__)),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "max_input_tokens": max(lengths),
        "evidence_truncated": False,
        "diagnostic_metadata_preserved": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
