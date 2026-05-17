#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_records(records: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    eval_size = int(len(shuffled) * eval_ratio)
    if eval_ratio > 0 and eval_size == 0 and len(shuffled) > 1:
        eval_size = 1
    if eval_size <= 0:
        return shuffled, []
    return shuffled[eval_size:], shuffled[:eval_size]


class RerankerScorer:
    def __init__(self, model_path: Path, *, device: str, max_length: int, batch_size: int) -> None:
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, queries: list[str], docs: list[str]) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(docs), self.batch_size):
            batch_queries = queries[offset : offset + self.batch_size]
            batch_docs = docs[offset : offset + self.batch_size]
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits.view(-1).detach().cpu().float().tolist()
            scores.extend(float(item) for item in logits)
        return scores


def summarize_pairwise(records: list[dict[str, Any]], scorer: RerankerScorer) -> dict[str, Any]:
    pos_scores = scorer.score([item["query"] for item in records], [item["positive"] for item in records])
    neg_scores = scorer.score([item["query"] for item in records], [item["negative"] for item in records])
    correct = 0
    ties = 0
    margins: list[float] = []
    by_negative_type: dict[str, dict[str, float]] = {}
    by_query_type: dict[str, dict[str, float]] = {}

    def update_bucket(bucket: dict[str, dict[str, float]], key: str, is_correct: bool, margin: float) -> None:
        stats = bucket.setdefault(key, {"count": 0, "correct": 0, "margin_sum": 0.0})
        stats["count"] += 1
        stats["correct"] += int(is_correct)
        stats["margin_sum"] += margin

    for record, pos_score, neg_score in zip(records, pos_scores, neg_scores, strict=True):
        margin = pos_score - neg_score
        is_correct = margin > 0
        correct += int(is_correct)
        ties += int(margin == 0)
        margins.append(margin)
        update_bucket(by_negative_type, str(record.get("negative_type") or "unknown"), is_correct, margin)
        update_bucket(by_query_type, str(record.get("query_type") or "unknown"), is_correct, margin)

    def finalize_bucket(bucket: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        finalized: dict[str, dict[str, float]] = {}
        for key, stats in sorted(bucket.items()):
            count = int(stats["count"])
            finalized[key] = {
                "count": count,
                "accuracy": round(float(stats["correct"]) / count, 4) if count else 0.0,
                "mean_margin": round(float(stats["margin_sum"]) / count, 4) if count else 0.0,
            }
        return finalized

    count = len(records)
    sorted_margins = sorted(margins)
    return {
        "count": count,
        "accuracy": round(correct / count, 4) if count else 0.0,
        "ties": ties,
        "mean_margin": round(sum(margins) / count, 4) if count else 0.0,
        "min_margin": round(min(margins), 4) if margins else 0.0,
        "p05_margin": round(sorted_margins[int(0.05 * (count - 1))], 4) if count else 0.0,
        "p50_margin": round(sorted_margins[int(0.50 * (count - 1))], 4) if count else 0.0,
        "by_negative_type": finalize_bucket(by_negative_type),
        "by_query_type": finalize_bucket(by_query_type),
    }


def summarize_listwise(records: list[dict[str, Any]], scorer: RerankerScorer) -> dict[str, Any]:
    flat_queries: list[str] = []
    flat_docs: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(flat_docs)
        for candidate in record["candidates"]:
            flat_queries.append(record["query"])
            flat_docs.append(candidate["text"])
        spans.append((start, len(flat_docs)))

    flat_scores = scorer.score(flat_queries, flat_docs)
    top1 = 0
    mrr = 0.0
    gold_margins: list[float] = []
    by_query_type: dict[str, dict[str, float]] = {}

    def update_bucket(key: str, is_top1: bool, reciprocal_rank: float, margin: float) -> None:
        stats = by_query_type.setdefault(key, {"count": 0, "top1": 0, "rr_sum": 0.0, "margin_sum": 0.0})
        stats["count"] += 1
        stats["top1"] += int(is_top1)
        stats["rr_sum"] += reciprocal_rank
        stats["margin_sum"] += margin

    for record, (start, end) in zip(records, spans, strict=True):
        scores = flat_scores[start:end]
        candidates = record["candidates"]
        gold_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.get("label") == "positive" or candidate.get("type") == "gold"
        ]
        if not gold_indices:
            continue
        gold_index = gold_indices[0]
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        rank = ranked_indices.index(gold_index) + 1
        reciprocal_rank = 1.0 / rank
        is_top1 = rank == 1
        top1 += int(is_top1)
        mrr += reciprocal_rank
        negative_scores = [score for index, score in enumerate(scores) if index != gold_index]
        margin = scores[gold_index] - max(negative_scores)
        gold_margins.append(margin)
        update_bucket(str(record.get("query_type") or "unknown"), is_top1, reciprocal_rank, margin)

    count = len(records)
    return {
        "count": count,
        "top1_accuracy": round(top1 / count, 4) if count else 0.0,
        "mrr": round(mrr / count, 4) if count else 0.0,
        "mean_gold_vs_best_negative_margin": round(sum(gold_margins) / count, 4) if count else 0.0,
        "by_query_type": {
            key: {
                "count": int(stats["count"]),
                "top1_accuracy": round(float(stats["top1"]) / int(stats["count"]), 4),
                "mrr": round(float(stats["rr_sum"]) / int(stats["count"]), 4),
                "mean_margin": round(float(stats["margin_sum"]) / int(stats["count"]), 4),
            }
            for key, stats in sorted(by_query_type.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate evidence-chain reranker pairwise/listwise metrics.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairwise-file", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v1_strict/reranker_pairwise.jsonl"))
    parser.add_argument("--listwise-file", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v1_strict/reranker_listwise.jsonl"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--split", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_path(args.model)
    pairwise_file = resolve_path(args.pairwise_file)
    listwise_file = resolve_path(args.listwise_file)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    pairwise_records = read_jsonl(pairwise_file)
    listwise_records = read_jsonl(listwise_file)
    if args.split != "all":
        pair_train, pair_eval = split_records(pairwise_records, eval_ratio=args.eval_ratio, seed=args.seed)
        list_train, list_eval = split_records(listwise_records, eval_ratio=args.eval_ratio, seed=args.seed)
        pairwise_records = pair_train if args.split == "train" else pair_eval
        listwise_records = list_train if args.split == "train" else list_eval

    scorer = RerankerScorer(model_path, device=device, max_length=args.max_length, batch_size=args.batch_size)
    summary = {
        "model": str(model_path),
        "device": device,
        "split": args.split,
        "pairwise_file": str(pairwise_file),
        "listwise_file": str(listwise_file),
        "max_length": args.max_length,
        "pairwise": summarize_pairwise(pairwise_records, scorer),
        "listwise": summarize_listwise(listwise_records, scorer),
    }

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
