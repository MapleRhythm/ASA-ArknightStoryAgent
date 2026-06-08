#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge evaluate_trace_replay_prompt_gold_recall shard JSON files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def cumulative_counts(ranks: list[int | None], top_ks: list[int]) -> dict[str, Any]:
    return {
        f"@{top_k}": {
            "count": sum(rank is not None and rank <= top_k for rank in ranks),
            "ratio": ratio(sum(rank is not None and rank <= top_k for rank in ranks), len(ranks)),
        }
        for top_k in top_ks
    } | {
        "missed": {
            "count": sum(rank is None for rank in ranks),
            "ratio": ratio(sum(rank is None for rank in ranks), len(ranks)),
        }
    }


def weighted_mean(rows: list[dict[str, Any]], field: str, weight_field: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in rows:
        weight = int(row.get(weight_field) or 0)
        numerator += float(row.get(field) or 0.0) * weight
        denominator += weight
    return round(numerator / denominator, 2) if denominator else 0.0


def merge_mode(mode_payloads: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    for payload in mode_payloads:
        prompt_rows.extend(payload.get("prompt_rows") or [])

    gold_unit_ranks: list[int | None] = []
    prompt_any_ranks: list[int | None] = []
    prompt_all_ranks: list[int | None] = []
    by_round: dict[str, dict[str, list[Any]]] = {}
    questions: set[str] = set()

    for row in prompt_rows:
        questions.add(str(row.get("question") or ""))
        unit_ranks = list(row.get("unit_ranks") or [])
        gold_unit_ranks.extend(unit_ranks)
        prompt_any_ranks.append(row.get("any_rank"))
        prompt_all_ranks.append(row.get("all_rank"))
        round_id = str(row.get("round") or "")
        bucket = by_round.setdefault(round_id, {"unit": [], "any": [], "all": [], "coverage": []})
        bucket["unit"].extend(unit_ranks)
        bucket["any"].append(row.get("any_rank"))
        bucket["all"].append(row.get("all_rank"))
        bucket["coverage"].append(float(row.get("coverage") or 0.0))

    coverages = [float(row.get("coverage") or 0.0) for row in prompt_rows]
    scope_records = sum(int((payload.get("counts") or {}).get("scope_records") or 0) for payload in mode_payloads)
    scoped_rows = []
    for payload in mode_payloads:
        row = dict(payload.get("scoped_local_stats") or {})
        row["scope_records"] = int((payload.get("counts") or {}).get("scope_records") or 0)
        scoped_rows.append(row)
    return {
        "counts": {
            "prompts": len(prompt_rows),
            "questions": len({question for question in questions if question}),
            "gold_units": len(gold_unit_ranks),
            "scope_records": scope_records,
        },
        "gold_unit_cumulative": cumulative_counts(gold_unit_ranks, top_ks),
        "prompt_any_gold_cumulative": cumulative_counts(prompt_any_ranks, top_ks),
        "prompt_all_gold_cumulative": cumulative_counts(prompt_all_ranks, top_ks),
        "prompt_gold_coverage": {
            "mean": round(statistics.mean(coverages), 4) if coverages else 0.0,
            "p50": round(statistics.median(coverages), 4) if coverages else 0.0,
            "full_coverage_count": sum(value >= 1.0 for value in coverages),
            "zero_coverage_count": sum(value <= 0.0 for value in coverages),
        },
        "rounds": {
            round_id: {
                "gold_unit_cumulative": cumulative_counts(bucket["unit"], top_ks),
                "prompt_any_gold_cumulative": cumulative_counts(bucket["any"], top_ks),
                "prompt_all_gold_cumulative": cumulative_counts(bucket["all"], top_ks),
                "coverage_mean": round(statistics.mean(bucket["coverage"]), 4) if bucket["coverage"] else 0.0,
                "prompts": len(bucket["coverage"]),
            }
            for round_id, bucket in sorted(by_round.items())
        },
        "scoped_local_stats": {
            "dense_mean": weighted_mean(scoped_rows, "dense_mean", "scope_records"),
            "sparse_mean": weighted_mean(scoped_rows, "sparse_mean", "scope_records"),
            "dense_max": max((int(row.get("dense_max") or 0) for row in scoped_rows), default=0),
            "sparse_max": max((int(row.get("sparse_max") or 0) for row in scoped_rows), default=0),
        },
        "prompt_rows": prompt_rows,
    }


def main() -> int:
    args = parse_args()
    shards = [read_json(path) for path in args.inputs]
    if not shards:
        raise SystemExit("No inputs")
    top_ks = list(shards[0].get("settings", {}).get("top_ks") or [1, 3, 5, 8, 10, 12])
    mode_names = sorted({mode for shard in shards for mode in (shard.get("modes") or {})})
    merged = {
        "settings": {
            **(shards[0].get("settings") or {}),
            "inputs": [str(path) for path in args.inputs],
            "merged_shards": len(shards),
        },
        "modes": {},
    }
    for mode in mode_names:
        mode_payloads = [
            shard["modes"][mode]
            for shard in shards
            if isinstance(shard.get("modes"), dict) and mode in shard["modes"]
        ]
        merged["modes"][mode] = merge_mode(mode_payloads, top_ks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
