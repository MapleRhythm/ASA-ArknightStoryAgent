#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import dedupe_samples, split_samples  # noqa: E402
from scripts.merge_sft_datasets import (  # noqa: E402
    DEFAULT_BASE_DIR,
    _normalize_record,
    bucket_of,
    load_jsonl,
    save_bucket_splits,
    save_json,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair old teacher_v2 dataset in-place or into a new output directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Existing teacher_v2 dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dataset directory. Default: overwrite input dir in-place.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def backup_manifest_if_needed(input_dir: Path, output_dir: Path) -> None:
    if input_dir != output_dir:
        return
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return
    backup_path = input_dir / "manifest.pre_repair.json"
    if not backup_path.exists():
        backup_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()

    backup_manifest_if_needed(input_dir, output_dir)

    raw_records = load_jsonl(input_dir / "all.jsonl")
    normalization_stats: Counter[str] = Counter()
    repaired_records = [
        normalized
        for record in raw_records
        if (
            normalized := _normalize_record(
                record,
                source_name=input_dir.name,
                stats=normalization_stats,
            )
        )
        is not None
    ]
    repaired_records = dedupe_samples(repaired_records)

    splits = split_samples(
        repaired_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_jsonl(output_dir / "all.jsonl", repaired_records)
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    save_bucket_splits(output_dir, splits)

    task_distribution = Counter(record.get("task_type") or "unknown" for record in repaired_records)
    category_distribution = Counter(bucket_of(record) for record in repaired_records)

    manifest = {
        "generator": "repair_teacher_v2_dataset",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stats": {
            "raw_total": len(raw_records),
            "repaired_total_after_cleaning": len(repaired_records),
            "split_sizes": {name: len(records) for name, records in splits.items()},
            "task_type_distribution": dict(task_distribution),
            "category_distribution": dict(category_distribution),
            "normalization": dict(normalization_stats),
        },
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "stats.json", manifest["stats"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
