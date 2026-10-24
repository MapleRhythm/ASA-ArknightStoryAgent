#!/usr/bin/env python3
"""Evaluate a BGE cross-encoder on query-answerability teacher pools.

The input may be passed multiple times (for example DeepSeek and GLM shards).
Every record is a complete, query-grouped candidate pool.  Metrics deliberately
focus on the online cutoffs used by ASA rather than on pairwise validation loss.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


GAIN = {
    "direct_answer": 3.0,
    "partial_answer": 2.0,
    "supporting_context": 1.0,
    "hard_context": 0.0,
    "irrelevant": 0.0,
}


def read_jsonl(
    paths: list[Path], *, duplicate_policy: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicates_by_path: Counter[str] = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("source_name") or ""), str(row.get("query") or ""))
                if key in seen:
                    if duplicate_policy == "error":
                        raise ValueError(f"duplicate query key at {path}:{line_number}: {key}")
                    duplicates_by_path[str(path)] += 1
                    continue
                seen.add(key)
                rows.append(row)
    return rows, {
        "policy": duplicate_policy,
        "kept_query_groups": len(rows),
        "duplicates_skipped": sum(duplicates_by_path.values()),
        "duplicates_skipped_by_path": dict(sorted(duplicates_by_path.items())),
    }


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def dcg(labels: list[str]) -> float:
    return sum(GAIN[label] / math.log2(index + 2) for index, label in enumerate(labels))


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"records": len(rows)}
    metric_names = sorted({key for row in rows for key in row if "@" in key})
    for metric in metric_names:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if not values:
            continue
        if metric.startswith(("D_hit", "DP_hit", "DPS_hit")):
            aggregate[metric] = sum(bool(value) for value in values) / len(values)
        else:
            aggregate[metric] = statistics.fmean(float(value) for value in values)
    return aggregate


class Scorer:
    def __init__(self, model_path: Path, device: str, max_length: int, batch_size: int) -> None:
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), trust_remote_code=True
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, documents: list[str]) -> list[float]:
        output: list[float] = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output.extend(
                float(value)
                for value in self.model(**encoded).logits.view(-1).float().cpu().tolist()
            )
        return output


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cutoff", type=int, action="append", default=[])
    parser.add_argument("--warmup-records", type=int, default=1)
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first"),
        default="error",
        help="With 'first', earlier --input paths have query-group priority.",
    )
    args = parser.parse_args()

    cutoffs = sorted(set(args.cutoff or [1, 5, 10, 32]))
    rows, duplicate_summary = read_jsonl(
        args.input, duplicate_policy=args.duplicate_policy
    )
    scorer = Scorer(args.model, args.device, args.max_length, args.batch_size)
    if rows and args.warmup_records:
        for row in rows[: args.warmup_records]:
            scorer.score(str(row["query"]), [str(item["text"]) for item in row["candidates"][:8]])

    latency: list[float] = []
    usable = 0
    skipped_ambiguous = 0
    skipped_no_positive = 0
    over_broad_positive_gt5 = 0
    label_counts: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        decision = str(row.get("teacher", {}).get("query_decision", {}).get("status") or "answerable")
        ranking = row.get("teacher", {}).get("ranking") or []
        by_id = {str(item["candidate_id"]): item for item in row["candidates"]}
        labels = {str(item["candidate_id"]): str(item["label"]) for item in ranking}
        if decision == "ambiguous":
            skipped_ambiguous += 1
            continue
        positive_ids = {cid for cid, label in labels.items() if label in {"direct_answer", "partial_answer", "supporting_context"}}
        if not positive_ids:
            skipped_no_positive += 1
            continue
        positive_count = len(positive_ids)
        label_quality = "over_broad_positive_gt5" if positive_count > 5 else "trainable_positive_le5"
        if positive_count > 5:
            over_broad_positive_gt5 += 1
        candidate_ids = list(by_id)
        synchronize(args.device)
        started = time.perf_counter()
        scores = scorer.score(str(row["query"]), [str(by_id[cid]["text"]) for cid in candidate_ids])
        synchronize(args.device)
        latency.append(time.perf_counter() - started)
        ordered_ids = [candidate_ids[position] for position in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
        ordered_labels = [labels[cid] for cid in ordered_ids]
        label_counts.update(ordered_labels)
        best_rank: dict[str, int | None] = {}
        for label_set_name, label_set in {
            "D": {"direct_answer"},
            "DP": {"direct_answer", "partial_answer"},
            "DPS": {"direct_answer", "partial_answer", "supporting_context"},
        }.items():
            ranks = [rank for rank, label in enumerate(ordered_labels, 1) if label in label_set]
            best_rank[label_set_name] = min(ranks) if ranks else None
        ideal_labels = sorted(ordered_labels, key=lambda label: GAIN[label], reverse=True)
        row_metrics: dict[str, Any] = {
            "query": row["query"],
            "query_type": str(row.get("query_type") or "unknown"),
            "source_name": row.get("source_name", ""),
            "teacher_provider": row.get("teacher_provider", ""),
            "positive_count": positive_count,
            "label_quality": label_quality,
            "latency_seconds": latency[-1],
            "best_rank": best_rank,
        }
        for cutoff in cutoffs:
            top = ordered_labels[:cutoff]
            ideal = ideal_labels[:cutoff]
            denom = dcg(ideal)
            row_metrics[f"ndcg@{cutoff}"] = dcg(top) / denom if denom else 0.0
            row_metrics[f"x_rate@{cutoff}"] = top.count("hard_context") / max(len(top), 1)
            for short, accepted in {
                "D": {"direct_answer"},
                "DP": {"direct_answer", "partial_answer"},
                "DPS": {"direct_answer", "partial_answer", "supporting_context"},
            }.items():
                total = sum(1 for label in ordered_labels if label in accepted)
                found = sum(1 for label in top if label in accepted)
                row_metrics[f"{short}_recall@{cutoff}"] = found / total if total else None
                row_metrics[f"{short}_hit@{cutoff}"] = bool(found)
        details.append(row_metrics)
        usable += 1
        if index % 10 == 0:
            print(f"[{index}/{len(rows)}] usable={usable}", flush=True)

    aggregate = aggregate_metrics(details)
    details_by_query_type: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        details_by_query_type.setdefault(str(row["query_type"]), []).append(row)
    aggregate_by_query_type = {
        query_type: aggregate_metrics(type_rows)
        for query_type, type_rows in sorted(details_by_query_type.items())
    }
    details_by_provider: dict[str, list[dict[str, Any]]] = {}
    details_by_label_quality: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        details_by_provider.setdefault(str(row["teacher_provider"] or "unknown"), []).append(row)
        details_by_label_quality.setdefault(str(row["label_quality"]), []).append(row)
    aggregate_by_teacher_provider = {
        provider: aggregate_metrics(provider_rows)
        for provider, provider_rows in sorted(details_by_provider.items())
    }
    aggregate_by_label_quality = {
        quality: aggregate_metrics(quality_rows)
        for quality, quality_rows in sorted(details_by_label_quality.items())
    }
    report = {
        "model": str(args.model),
        "inputs": [str(path) for path in args.input],
        "duplicate_query_groups": duplicate_summary,
        "records": len(rows),
        "usable_records": usable,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_no_positive": skipped_no_positive,
        "over_broad_positive_gt5": over_broad_positive_gt5,
        "cutoffs": cutoffs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "aggregate": aggregate,
        "aggregate_by_query_type": aggregate_by_query_type,
        "aggregate_by_teacher_provider": aggregate_by_teacher_provider,
        "aggregate_by_label_quality": aggregate_by_label_quality,
        "candidate_label_counts": dict(label_counts),
        "latency_seconds": {
            "mean": statistics.fmean(latency) if latency else 0.0,
            "p50": percentile(latency, 0.50),
            "p95": percentile(latency, 0.95),
            "max": max(latency, default=0.0),
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
