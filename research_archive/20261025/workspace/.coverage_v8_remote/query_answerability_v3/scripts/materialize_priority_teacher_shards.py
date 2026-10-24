#!/usr/bin/env python3
"""Materialize a complete teacher shard by whole-query provider priority.

Teacher inputs are ordered from highest to lowest priority.  A complete row is
selected from the first shard containing each frozen ``(source_name, query)``
key.  Candidate pools must exactly match the frozen query-only input.  Source
teacher shards are read-only and are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row must be an object at {path}:{line_number}")
            yield line_number, row


def query_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source_name") or ""), str(row.get("query") or "")


def pool_fingerprint(row: dict[str, Any]) -> str:
    candidates = row.get("candidates") or []
    if not isinstance(candidates, list) or any(
        not isinstance(item, dict) for item in candidates
    ):
        raise ValueError("candidates must be a list of objects")
    stable = sorted(
        (
            str(item.get("candidate_id") or ""),
            str(item.get("kind") or ""),
            str(item.get("text") or ""),
        )
        for item in candidates
    )
    payload = json.dumps(stable, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    frozen_rows: list[dict[str, Any]] = []
    frozen_fingerprints: dict[tuple[str, str], str] = {}
    for line_number, row in iter_jsonl(args.input):
        key = query_key(row)
        if not all(key):
            raise ValueError(f"empty frozen query key at {args.input}:{line_number}")
        if key in frozen_fingerprints:
            raise ValueError(f"duplicate frozen query key at {args.input}:{line_number}: {key}")
        if len(row.get("candidates") or []) != args.expected_candidates:
            raise ValueError(
                f"candidate_count={len(row.get('candidates') or [])} at "
                f"{args.input}:{line_number}"
            )
        frozen_rows.append(row)
        frozen_fingerprints[key] = pool_fingerprint(row)

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_path: dict[tuple[str, str], str] = {}
    rows_by_path: Counter[str] = Counter()
    selected_by_path: Counter[str] = Counter()
    duplicates_by_path: Counter[str] = Counter()
    for path in args.teacher:
        local_seen: set[tuple[str, str]] = set()
        for line_number, row in iter_jsonl(path):
            rows_by_path[str(path)] += 1
            key = query_key(row)
            if key in local_seen:
                raise ValueError(f"duplicate key within teacher shard at {path}:{line_number}: {key}")
            local_seen.add(key)
            if key not in frozen_fingerprints:
                raise ValueError(f"teacher key is not in frozen input at {path}:{line_number}: {key}")
            if len(row.get("candidates") or []) != args.expected_candidates:
                raise ValueError(f"candidate_count mismatch at {path}:{line_number}: {key}")
            if pool_fingerprint(row) != frozen_fingerprints[key]:
                raise ValueError(f"candidate pool drift at {path}:{line_number}: {key}")
            if key in selected:
                duplicates_by_path[str(path)] += 1
                continue
            selected[key] = row
            selected_path[key] = str(path)
            selected_by_path[str(path)] += 1

    missing = [query_key(row) for row in frozen_rows if query_key(row) not in selected]
    manifest = {
        "policy": "first_complete_query_group_wins",
        "frozen_input": str(args.input),
        "teacher_priority": [str(path) for path in args.teacher],
        "output": str(args.output),
        "expected_queries": len(frozen_rows),
        "selected_queries": len(selected),
        "missing_queries": len(missing),
        "missing_query_keys": missing,
        "rows_by_path": dict(rows_by_path),
        "selected_by_path": dict(selected_by_path),
        "lower_priority_duplicates_skipped": dict(duplicates_by_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise RuntimeError(
            f"teacher coverage is incomplete: {len(missing)} frozen queries are missing"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for frozen_row in frozen_rows:
            sink.write(
                json.dumps(selected[query_key(frozen_row)], ensure_ascii=False) + "\n"
            )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
