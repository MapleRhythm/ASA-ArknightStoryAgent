#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/llama_factory/action_target_hard_negative_kto_v1"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict[str, Any], source: str, index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"{source}[{index}] id={record.get('id', '<missing>')}"
    if not isinstance(record.get("kto_tag"), bool):
        errors.append(f"{prefix}: kto_tag must be boolean")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        errors.append(f"{prefix}: conversations must contain exactly one user turn and one assistant turn")
        return errors
    expected = [("human", "value"), ("gpt", "value")]
    for turn, (role, content_key) in zip(conversations, expected, strict=True):
        if not isinstance(turn, dict):
            errors.append(f"{prefix}: conversation turn must be object")
            continue
        if turn.get("from") != role:
            errors.append(f"{prefix}: expected role {role}, got {turn.get('from')!r}")
        content = turn.get(content_key)
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{prefix}: {role} content must be non-empty string")
    assistant = conversations[-1].get("value") if isinstance(conversations[-1], dict) else ""
    try:
        payload = json.loads(assistant)
    except Exception as exc:
        errors.append(f"{prefix}: assistant value must be compact JSON: {exc}")
        return errors
    if not isinstance(payload, dict):
        errors.append(f"{prefix}: assistant JSON must be object")
        return errors
    if payload.get("next_action") != "answer_directly":
        errors.append(f"{prefix}: next_action must be answer_directly")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append(f"{prefix}: answer must be non-empty")
    return errors


def validate_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.exists():
        return {}, [f"missing {dataset_info_path}"]
    dataset_info = load_json(dataset_info_path)
    if not isinstance(dataset_info, dict):
        return {}, ["dataset_info.json must be object"]
    stats: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "files": {},
        "kto_tags": Counter(),
        "targets": Counter(),
        "preferences": Counter(),
    }
    for dataset_name, entry in dataset_info.items():
        file_name = entry.get("file_name") if isinstance(entry, dict) else None
        if not isinstance(file_name, str):
            errors.append(f"{dataset_name}: file_name missing")
            continue
        columns = entry.get("columns") if isinstance(entry, dict) else {}
        if columns.get("messages") != "conversations":
            errors.append(f"{dataset_name}: columns.messages must map to conversations")
        if columns.get("kto_tag") != "kto_tag":
            errors.append(f"{dataset_name}: columns.kto_tag must map to kto_tag")
        data_path = dataset_dir / file_name
        if not data_path.exists():
            errors.append(f"{dataset_name}: missing {data_path}")
            continue
        records = load_json(data_path)
        if not isinstance(records, list):
            errors.append(f"{dataset_name}: {file_name} must be a JSON list")
            continue
        stats["files"][file_name] = len(records)
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{file_name}[{index}]: record must be object")
                continue
            errors.extend(validate_record(record, file_name, index))
            stats["kto_tags"][str(record.get("kto_tag"))] += 1
            meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
            stats["targets"][str(meta.get("target"))] += 1
            stats["preferences"][str(meta.get("preference"))] += 1
    for key in ("kto_tags", "targets", "preferences"):
        stats[key] = dict(stats[key])
    if stats["kto_tags"].get("True", 0) == 0 or stats["kto_tags"].get("False", 0) == 0:
        errors.append("dataset must contain both True and False kto_tag examples")
    return stats, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate action-target KTO dataset format.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats, errors = validate_dataset(resolve_path(args.dataset_dir))
    print(json.dumps({"stats": stats, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
