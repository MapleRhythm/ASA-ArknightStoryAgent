#!/usr/bin/env python3
"""Extract current-schema SFT records from API-verifier KTO data.

Input is a verifier dataset built by scripts/build_soda_api_verifier_dataset.py.
Only chosen records (kto_tag=true) are kept by default. The output is a
ShareGPT SFT dataset without kto_tag, compatible with LLaMA-Factory SFT.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_jsonish(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def assistant_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    last = conversations[-1]
    if not isinstance(last, dict):
        return None
    return parse_jsonish(str(last.get("value") or ""))


def dataset_info(dataset_name: str, split_files: list[str]) -> dict[str, Any]:
    return {
        f"{dataset_name}_{split.removesuffix('.json')}": {
            "file_name": split,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }
        for split in split_files
    }


def clone_as_sft_record(record: dict[str, Any], *, source_dataset: str, split: str) -> dict[str, Any]:
    output = json.loads(json.dumps(record, ensure_ascii=False))
    output.pop("kto_tag", None)
    meta = output.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["source_api_verifier_dataset"] = source_dataset
        meta["source_api_verifier_split"] = split
        meta["sft_extracted_from_kto_tag"] = True
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract chosen API-verifier KTO records into an SFT dataset.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--task-types", default="user_question_hypothesis_generation,conclusion_generation")
    parser.add_argument("--chosen-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    split_files = [item.strip() for item in args.splits.split(",") if item.strip()]
    allowed_task_types = {item.strip() for item in args.task_types.split(",") if item.strip()}
    stats: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for name in [*split_files, "dataset_info.json", "summary.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    for split in split_files:
        source_path = input_dir / split
        if not source_path.exists():
            write_json(output_dir / split, [])
            stats[f"missing_split:{split}"] += 1
            continue
        source_records = read_json(source_path)
        if not isinstance(source_records, list):
            raise SystemExit(f"Input split is not a list: {source_path}")
        output_records: list[dict[str, Any]] = []
        for record in source_records:
            if not isinstance(record, dict):
                continue
            stats["records_total"] += 1
            task_type = str(record.get("task_type") or "")
            if task_type not in allowed_task_types:
                stats[f"skip_task:{task_type or '<empty>'}"] += 1
                continue
            if args.chosen_only and "kto_tag" in record and not bool(record.get("kto_tag")):
                stats["skip_rejected"] += 1
                continue
            payload = assistant_payload(record)
            if task_type == "conclusion_generation":
                action = str((payload or {}).get("next_action") or "<parse_error>")
                stats[f"conclusion_action:{action}"] += 1
            output_records.append(clone_as_sft_record(record, source_dataset=input_dir.name, split=split))
            stats[f"output_split:{split}"] += 1
        write_json(output_dir / split, output_records)

    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name, split_files))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "splits": {split: len(read_json(output_dir / split)) for split in split_files},
        "task_types": sorted(allowed_task_types),
        "chosen_only": args.chosen_only,
        "stats": dict(stats),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
