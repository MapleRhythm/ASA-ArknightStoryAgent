#!/usr/bin/env python3
"""Independent evaluation for the binding verifier reranker.

The training builder intentionally excludes suspected missed positives.  This
evaluator reconstructs them from the raw hard-negative audit and measures:

* held-out clean pairwise accuracy (the 44 eval pairs);
* suspected-missed-positive vs confirmed-negative accuracy;
* per-query-type accuracy and mean margin;
* ROC-AUC without requiring sklearn.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auc(scores: list[float], labels: list[int]) -> float:
    positives = sorted(float(s) for s, y in zip(scores, labels) if y == 1)
    negatives = sorted(float(s) for s, y in zip(scores, labels) if y == 0)
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


class Scorer:
    def __init__(self, model_path: Path, device: str, max_length: int, batch_size: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, queries: list[str], docs: list[str]) -> list[float]:
        result: list[float] = []
        for offset in range(0, len(queries), self.batch_size):
            inputs = self.tokenizer(
                queries[offset : offset + self.batch_size],
                docs[offset : offset + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            result.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        return [float(value) for value in result]


def summarize(
    name: str, records: list[dict[str, Any]], scorer: Scorer
) -> dict[str, Any]:
    if not records:
        return {"name": name, "count": 0}
    queries = [str(row["query"]) for row in records]
    positives = [str(row["positive"]) for row in records]
    negatives = [str(row["negative"]) for row in records]
    pos_scores = scorer.score(queries, positives)
    neg_scores = scorer.score(queries, negatives)
    margins = [p - n for p, n in zip(pos_scores, neg_scores)]
    by_type: dict[str, list[float]] = defaultdict(list)
    for row, margin in zip(records, margins):
        by_type[str(row.get("query_type") or "unknown")].append(margin)
    return {
        "name": name,
        "count": len(records),
        "accuracy": sum(m > 0 for m in margins) / len(margins),
        "ties": sum(m == 0 for m in margins),
        "mean_margin": sum(margins) / len(margins),
        "p05_margin": sorted(margins)[max(0, int(0.05 * (len(margins) - 1)))],
        "auc": auc(pos_scores + neg_scores, [1] * len(pos_scores) + [0] * len(neg_scores)),
        "by_query_type": {
            key: {
                "count": len(values),
                "accuracy": sum(value > 0 for value in values) / len(values),
                "mean_margin": sum(values) / len(values),
            }
            for key, values in sorted(by_type.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-pairs", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    heldout = read_jsonl(args.eval_pairs)
    hardneg = read_jsonl(args.hardneg)
    suspected: list[dict[str, Any]] = []
    for source in hardneg:
        confirmed = [
            item
            for item in source.get("hard_negatives") or []
            if str(item.get("glm_verdict") or "") == "unsupported"
            and str(item.get("text") or "").strip()
        ]
        for missed in source.get("suspected_missed_positives") or []:
            if str(missed.get("glm_verdict") or "") != "supported":
                continue
            positive = str(missed.get("text") or "").strip()
            if not positive:
                continue
            # Keep every confirmed negative: this evaluates ordering against
            # the same distractor pool used by the hard-negative audit.
            for negative in confirmed:
                suspected.append(
                    {
                        "query": str(source.get("claim") or source.get("query") or ""),
                        "query_type": source.get("query_type"),
                        "positive": positive,
                        "negative": str(negative.get("text") or "").strip(),
                    }
                )

    scorer = Scorer(args.model, args.device, args.max_length, args.batch_size)
    summary = {
        "model": str(args.model),
        "device": args.device,
        "max_length": args.max_length,
        "heldout_clean": summarize("heldout_clean", heldout, scorer),
        "suspected_missed_positive": summarize(
            "suspected_missed_positive", suspected, scorer
        ),
        "raw_counts": {
            "heldout_pairs": len(heldout),
            "hardneg_records": len(hardneg),
            "suspected_pairs": len(suspected),
            "suspected_positive_records": sum(
                1
                for source in hardneg
                for missed in source.get("suspected_missed_positives") or []
                if str(missed.get("glm_verdict") or "") == "supported"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
