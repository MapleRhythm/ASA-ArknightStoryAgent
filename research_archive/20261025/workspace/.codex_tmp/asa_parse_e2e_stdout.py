#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def parse_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    position = 0
    records: list[dict] = []
    while True:
        start = text.find("{", position)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(payload, dict) and "question" in payload and "answer" in payload:
            records.append(payload)
        position = end
    return records


def main() -> int:
    args = parse_args()
    records = parse_objects(args.input.read_text(encoding="utf-8", errors="replace"))
    durations = [float(item) for item in re.findall(r"^\[done\]\s+([0-9.]+)s$", args.log.read_text(), re.M)]
    if len(records) != len(durations):
        raise SystemExit(f"record/duration mismatch: {len(records)} != {len(durations)}")
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    abstain = sum("证据不足" in str(record.get("answer") or "") for record in records)
    two_round = sum(len(record.get("retrieval_trace") or []) == 2 for record in records)
    summary = {
        "count": len(records),
        "durations_sec": durations,
        "mean_sec": round(sum(durations) / len(durations), 3),
        "min_sec": min(durations),
        "max_sec": max(durations),
        "over_20s": sum(value > 20 for value in durations),
        "two_round_count": two_round,
        "abstain_like_count": abstain,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
