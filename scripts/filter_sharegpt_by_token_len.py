#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))

from transformers import AutoTokenizer


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    system = str(record.get("system") or "").strip()
    if system:
        parts.append(system)
    for message in record.get("conversations") or []:
        if not isinstance(message, dict):
            continue
        parts.append(str(message.get("from") or ""))
        parts.append(str(message.get("value") or ""))
    return "\n".join(parts)


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter ShareGPT records by tokenizer length.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tokenizer", default="model/qwen3.5-4b")
    parser.add_argument("--cutoff-len", type=int, required=True)
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    tokenizer_path = resolve_path(args.tokenizer)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in [*splits, "dataset_info.json", "summary.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    summary: dict[str, Any] = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "tokenizer": str(tokenizer_path),
        "cutoff_len": args.cutoff_len,
        "splits": {},
        "dropped": {},
        "actions": {},
    }
    action_counts: Counter[str] = Counter()

    for split in splits:
        records = read_json(input_dir / split)
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for record in records:
            token_len = len(tokenizer.encode(record_text(record), add_special_tokens=False))
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(record.get("conversations", [])[-1].get("value", "{}"))
            except Exception:
                payload = {}
            if token_len <= args.cutoff_len:
                kept_record = dict(record)
                kept_record["token_len"] = token_len
                kept.append(kept_record)
                action = str(payload.get("next_action") or payload.get("a") or "unknown")
                action_counts[action] += 1
            else:
                dropped.append({"id": record.get("id"), "token_len": token_len})
        write_json(output_dir / split, kept)
        summary["splits"][split] = {"source": len(records), "kept": len(kept), "dropped": len(dropped)}
        summary["dropped"][split] = dropped[:200]

    summary["actions"] = dict(action_counts)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
