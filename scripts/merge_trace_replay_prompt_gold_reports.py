#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_soda_gold_evidence_topk import cumulative_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded trace replay prompt-gold recall reports.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_mean(items: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in items)
    if not total_weight:
        return 0.0
    return round(sum(value * weight for value, weight in items) / total_weight, 2)


def merge_mode(mode_reports: list[tuple[Path, dict[str, Any]]], top_ks: list[int]) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    scope_records = 0
    dense_means: list[tuple[float, int]] = []
    sparse_means: list[tuple[float, int]] = []
    dense_max = 0
    sparse_max = 0

    for report_path, payload in mode_reports:
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        scoped_stats = payload.get("scoped_local_stats") if isinstance(payload.get("scoped_local_stats"), dict) else {}
        shard_scope_records = int(counts.get("scope_records") or 0)
        scope_records += shard_scope_records
        dense_means.append((float(scoped_stats.get("dense_mean") or 0.0), shard_scope_records))
        sparse_means.append((float(scoped_stats.get("sparse_mean") or 0.0), shard_scope_records))
        dense_max = max(dense_max, int(scoped_stats.get("dense_max") or 0))
        sparse_max = max(sparse_max, int(scoped_stats.get("sparse_max") or 0))

        for row in payload.get("prompt_rows") or []:
            if "unit_ranks" not in row:
                raise ValueError(f"{report_path} lacks prompt_rows[].unit_ranks; rerun shards with the updated evaluator.")
            merged_row = dict(row)
            merged_row["source_report"] = str(report_path)
            prompt_rows.append(merged_row)

    gold_unit_ranks: list[int | None] = []
    prompt_any_ranks: list[int | None] = []
    prompt_all_ranks: list[int | None] = []
    by_round: dict[str, dict[str, list[Any]]] = {}
    for row in prompt_rows:
        ranks = list(row.get("unit_ranks") or [])
        gold_unit_ranks.extend(ranks)
        prompt_any_ranks.append(row.get("any_rank"))
        prompt_all_ranks.append(row.get("all_rank"))
        round_id = str(row.get("round") or "")
        bucket = by_round.setdefault(round_id, {"unit": [], "any": [], "all": [], "coverage": []})
        bucket["unit"].extend(ranks)
        bucket["any"].append(row.get("any_rank"))
        bucket["all"].append(row.get("all_rank"))
        bucket["coverage"].append(float(row.get("coverage") or 0.0))

    coverages = [float(row.get("coverage") or 0.0) for row in prompt_rows]
    return {
        "counts": {
            "prompts": len(prompt_rows),
            "questions": len({row.get("question") for row in prompt_rows}),
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
                "gold_unit_cumulative": cumulative_counts(payload["unit"], top_ks),
                "prompt_any_gold_cumulative": cumulative_counts(payload["any"], top_ks),
                "prompt_all_gold_cumulative": cumulative_counts(payload["all"], top_ks),
                "coverage_mean": round(statistics.mean(payload["coverage"]), 4) if payload["coverage"] else 0.0,
                "prompts": len(payload["coverage"]),
            }
            for round_id, payload in sorted(by_round.items())
        },
        "scoped_local_stats": {
            "dense_mean": weighted_mean(dense_means),
            "sparse_mean": weighted_mean(sparse_means),
            "dense_max": dense_max,
            "sparse_max": sparse_max,
        },
        "prompt_rows": prompt_rows,
    }


def main() -> int:
    args = parse_args()
    reports = [(path, read_json(path)) for path in args.inputs]
    if not reports:
        raise ValueError("no input reports")

    top_ks = list(reports[0][1].get("settings", {}).get("top_ks") or [1, 3, 5, 8, 10, 12])
    modes = sorted(set().union(*(set(report.get("modes", {}).keys()) for _, report in reports)))
    output = {
        "settings": {
            "inputs": [str(path) for path, _ in reports],
            "top_ks": top_ks,
            "match_threshold": reports[0][1].get("settings", {}).get("match_threshold"),
        },
        "modes": {},
    }
    for mode in modes:
        mode_reports: list[tuple[Path, dict[str, Any]]] = []
        for report_path, report in reports:
            mode_payload = report.get("modes", {}).get(mode)
            if mode_payload is not None:
                mode_reports.append((report_path, mode_payload))
        output["modes"][mode] = merge_mode(mode_reports, top_ks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({mode: output["modes"][mode]["prompt_gold_coverage"] for mode in output["modes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
