#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import normalize_task_type  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy dataset task labels to the current task-type names."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory, e.g. data/processed/sft_data/teacher_v2",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def normalize_distribution(value: Any) -> tuple[dict[str, int], bool]:
    if not isinstance(value, dict):
        return value, False
    output: dict[str, int] = {}
    changed = False
    for key, raw_count in value.items():
        canonical = normalize_task_type(key)
        if canonical != key:
            changed = True
        output[canonical] = output.get(canonical, 0) + int(raw_count)
    return output, changed


def update_sample_record(record: dict[str, Any]) -> bool:
    changed = False
    task_type = record.get("task_type")
    canonical = normalize_task_type(task_type)
    if canonical != task_type:
        record["task_type"] = canonical
        changed = True

    meta = record.get("meta")
    if canonical in {
        "user_question_hypothesis_generation",
        "follow_up_hypothesis_generation",
        "conclusion_generation",
    } and isinstance(meta, dict):
        task_family = (
            "conclusion_generation"
            if canonical == "conclusion_generation"
            else "hypothesis_generation"
        )
        if meta.get("task_family") != task_family:
            meta["task_family"] = task_family
            changed = True
        if canonical != "conclusion_generation" and "decision_case" in meta and meta["decision_case"] is not None:
            meta["decision_case"] = None
            changed = True
    return changed


def update_json_payload(payload: Any) -> bool:
    changed = False

    if isinstance(payload, dict):
        if "task_type" in payload:
            canonical = normalize_task_type(payload.get("task_type"))
            if canonical != payload.get("task_type"):
                payload["task_type"] = canonical
                changed = True

        for key in ("task_type_distribution", "target_task_distribution"):
            if key in payload:
                normalized, key_changed = normalize_distribution(payload[key])
                if key_changed:
                    payload[key] = normalized
                    changed = True

        if "stats" in payload and isinstance(payload["stats"], dict):
            if update_json_payload(payload["stats"]):
                changed = True

    return changed


def migrate_dataset(dataset_dir: Path) -> dict[str, Any]:
    before = Counter()
    after = Counter()
    jsonl_files_changed = 0
    json_files_changed = 0

    jsonl_paths = sorted(dataset_dir.rglob("*.jsonl"))
    for path in jsonl_paths:
        records = load_jsonl(path)
        changed = False
        for record in records:
            task_type = record.get("task_type")
            if task_type:
                before[str(task_type)] += 1
            if update_sample_record(record):
                changed = True
            task_type = record.get("task_type")
            if task_type:
                after[str(task_type)] += 1
        if changed:
            save_jsonl(path, records)
            jsonl_files_changed += 1

    json_paths = sorted(dataset_dir.rglob("*.json"))
    for path in json_paths:
        payload = load_json(path)
        if update_json_payload(payload):
            save_json(path, payload)
            json_files_changed += 1

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "jsonl_files_changed": jsonl_files_changed,
        "json_files_changed": json_files_changed,
        "task_type_distribution_before": dict(before),
        "task_type_distribution_after": dict(after),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve() if not args.dataset_dir.is_absolute() else args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")
    summary = migrate_dataset(dataset_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
