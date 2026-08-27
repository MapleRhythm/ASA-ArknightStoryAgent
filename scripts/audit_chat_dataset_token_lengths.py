#!/usr/bin/env python3
"""Audit exact chat-template token lengths for frozen ShareGPT JSON datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "observation": "tool",
    "function_call": "assistant",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(row.get("system") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("conversations is not a list")
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("conversation item is not an object")
        role = ROLE_MAP.get(str(message.get("from") or ""), str(message.get("from") or ""))
        content = message.get("value")
        if not role or not isinstance(content, str):
            raise ValueError("conversation item has invalid role or content")
        messages.append({"role": role, "content": content})
    return messages


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    report: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "model": str(args.model.resolve()),
        "cutoff": args.cutoff,
        "enable_thinking": args.enable_thinking,
        "splits": {},
    }
    violation_count = 0
    for split in ("train", "val", "test"):
        path = args.dataset_dir / f"{split}.json"
        if not path.exists():
            continue
        lengths: list[tuple[int, str, str]] = []
        for row in read_rows(path):
            encoded = tokenizer.apply_chat_template(
                chat_messages(row),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=args.enable_thinking,
                return_dict=True,
            )
            length = len(encoded["input_ids"])
            lengths.append((length, str(row.get("id") or ""), str(row.get("task_type") or "")))
        lengths.sort()
        values = [item[0] for item in lengths]
        over_cutoff = sum(value > args.cutoff for value in values)
        violation_count += over_cutoff
        report["splits"][split] = {
            "rows": len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0),
            "over_cutoff": over_cutoff,
            "longest": [
                {"tokens": length, "id": row_id, "task_type": task_type}
                for length, row_id, task_type in reversed(lengths[-20:])
            ],
        }
    report["total_over_cutoff"] = violation_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if violation_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
