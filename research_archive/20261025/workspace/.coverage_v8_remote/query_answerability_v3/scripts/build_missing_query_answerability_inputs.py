#!/usr/bin/env python3
"""Materialize only unlabeled query groups without changing old checkpoints.

The frozen query-only input remains authoritative.  Completed teacher shards
are read in priority order and must contain exactly the same candidate pool for
any matching ``(source_name, query)`` key.  The output is a fresh resumable
input shard; existing teacher JSONL files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--completed", type=Path, action="append", default=[])
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


def load_frozen(path: Path, expected_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in iter_jsonl(path):
        key = query_key(row)
        if not all(key):
            raise ValueError(f"empty source/query at {path}:{line_number}")
        if key in seen:
            raise ValueError(f"duplicate frozen query key at {path}:{line_number}: {key}")
        seen.add(key)
        candidates = row.get("candidates") or []
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"candidate_count={len(candidates)} at {path}:{line_number}; "
                f"expected={expected_candidates}"
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    frozen_rows = load_frozen(args.input, args.expected_candidates)
    frozen_by_key = {query_key(row): row for row in frozen_rows}
    completed_keys: set[tuple[str, str]] = set()
    duplicate_completed = 0
    completed_rows = 0
    for path in args.completed:
        for line_number, row in iter_jsonl(path):
            completed_rows += 1
            key = query_key(row)
            if key not in frozen_by_key:
                raise ValueError(f"completed key not in frozen input at {path}:{line_number}: {key}")
            if pool_fingerprint(row) != pool_fingerprint(frozen_by_key[key]):
                raise ValueError(f"candidate pool drift at {path}:{line_number}: {key}")
            if key in completed_keys:
                duplicate_completed += 1
            completed_keys.add(key)

    missing_rows = [row for row in frozen_rows if query_key(row) not in completed_keys]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in missing_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "frozen_input": str(args.input),
        "completed_shards": [str(path) for path in args.completed],
        "frozen_queries": len(frozen_rows),
        "completed_rows": completed_rows,
        "completed_unique_queries": len(completed_keys),
        "duplicate_completed_rows": duplicate_completed,
        "missing_queries": len(missing_rows),
        "output": str(args.output),
        "missing_query_keys": [query_key(row) for row in missing_rows],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
