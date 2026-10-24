#!/usr/bin/env python3
"""Merge exported BGE pairs with provider priority and split-leakage checks.

DeepSeek-labelled query groups are kept as the primary source.  GLM pairs are
used only when a complete ``(source_name, query)`` group is absent from the
DeepSeek export.  The script operates on compact pair files, not the original
120-candidate teacher records.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-primary", type=Path, required=True)
    parser.add_argument("--train-fallback", type=Path, required=True)
    parser.add_argument(
        "--train-primary-groups",
        type=Path,
        help="Raw primary teacher shard; blocks fallback for every labelled query, even if pair export skipped it.",
    )
    parser.add_argument("--val-primary", type=Path, required=True)
    parser.add_argument("--val-fallback", type=Path, required=True)
    parser.add_argument(
        "--val-primary-groups",
        type=Path,
        help="Raw primary teacher shard; blocks fallback for every labelled query, even if pair export skipped it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def query_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("source_name") or ""), str(record.get("query") or "")


def pair_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    source_name, query = query_key(record)
    return (
        source_name,
        query,
        str(record.get("positive") or ""),
        str(record.get("negative") or ""),
    )


def scan_query_keys(path: Path) -> set[tuple[str, str]]:
    return {query_key(record) for record in iter_jsonl(path)}


def merge_split(
    primary: Path,
    fallback: Path,
    output: Path,
    *,
    primary_groups: Path | None = None,
) -> dict[str, Any]:
    exported_primary_query_keys = scan_query_keys(primary)
    primary_query_keys = (
        scan_query_keys(primary_groups) if primary_groups else exported_primary_query_keys
    )
    seen_pairs: set[tuple[str, str, str, str]] = set()
    output_query_keys: set[tuple[str, str]] = set()
    story_families: set[str] = set()
    source_rows: Counter[str] = Counter()
    duplicate_rows: Counter[str] = Counter()
    skipped_fallback_rows = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for provider, path in (("primary", primary), ("fallback", fallback)):
            for record in iter_jsonl(path):
                key = query_key(record)
                if provider == "fallback" and key in primary_query_keys:
                    skipped_fallback_rows += 1
                    continue
                current_pair_key = pair_key(record)
                if current_pair_key in seen_pairs:
                    duplicate_rows[provider] += 1
                    continue
                seen_pairs.add(current_pair_key)
                output_query_keys.add(key)
                story_family = str(record.get("story_family") or "")
                if story_family:
                    story_families.add(story_family)
                source_rows[provider] += 1
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "primary": str(primary),
        "primary_groups": str(primary_groups) if primary_groups else None,
        "fallback": str(fallback),
        "output": str(output),
        "rows": sum(source_rows.values()),
        "queries": len(output_query_keys),
        "story_families": len(story_families),
        "source_rows": dict(sorted(source_rows.items())),
        "duplicate_rows_skipped": dict(sorted(duplicate_rows.items())),
        "fallback_rows_skipped_due_to_primary_query": skipped_fallback_rows,
        "primary_labelled_queries": len(primary_query_keys),
        "primary_exported_queries": len(exported_primary_query_keys),
        "query_keys": output_query_keys,
        "story_family_values": story_families,
    }


def serializable(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"query_keys", "story_family_values"}
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    train_summary = merge_split(
        args.train_primary.resolve(),
        args.train_fallback.resolve(),
        output_dir / "train_pairs.jsonl",
        primary_groups=(
            args.train_primary_groups.resolve() if args.train_primary_groups else None
        ),
    )
    val_summary = merge_split(
        args.val_primary.resolve(),
        args.val_fallback.resolve(),
        output_dir / "val_pairs.jsonl",
        primary_groups=(
            args.val_primary_groups.resolve() if args.val_primary_groups else None
        ),
    )

    query_overlap = train_summary["query_keys"] & val_summary["query_keys"]
    story_family_overlap = (
        train_summary["story_family_values"] & val_summary["story_family_values"]
    )
    # Some historical exports omitted story_family from compact pair rows even
    # though the split was constructed by family upstream.  An empty set is
    # therefore "not checked", never evidence that the split is leak-free.
    story_family_check_available = bool(
        train_summary["story_family_values"] and val_summary["story_family_values"]
    )
    manifest = {
        "merge_policy": "primary_query_group_then_fallback",
        "train": serializable(train_summary),
        "val": serializable(val_summary),
        "leakage": {
            "query_key_overlap_count": len(query_overlap),
            "query_key_overlap_examples": sorted(query_overlap)[:10],
            "story_family_check_available": story_family_check_available,
            "story_family_overlap_count": (
                len(story_family_overlap) if story_family_check_available else None
            ),
            "story_family_overlap_examples": sorted(story_family_overlap)[:10],
        },
    }
    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not story_family_check_available:
        raise RuntimeError(
            "story_family is missing from one or both splits; outputs were written for audit but must not be trained"
        )
    if query_overlap or story_family_overlap:
        raise RuntimeError(
            "train/validation leakage detected; outputs were written for audit but must not be trained"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
