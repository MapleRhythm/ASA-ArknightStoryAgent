#!/usr/bin/env python3
"""Join GLM and 4B judgements and compute agreement metrics and strata."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    y = [int(bool(r["glm_supports"])) for r in rows]
    p = [int(bool(r["fourb_supports"])) for r in rows]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    agreement = (tp + tn) / n
    po = agreement
    py = sum(y) / n
    pp = sum(p) / n
    pe = py * pp + (1 - py) * (1 - pp)
    kappa = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else (1.0 if po == 1 else 0.0)
    scores = [float(r.get("fourb_support_score", r["fourb_supports"])) for r in rows]
    auc: float | None
    if len(set(y)) < 2:
        auc = None
    else:
        order = sorted(range(n), key=lambda i: scores[i])
        rank_sum = 0.0
        rank = 1
        i = 0
        while i < n:
            j = i
            while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg_rank = (rank + rank + (j - i)) / 2
            rank_sum += sum(y[order[k]] for k in range(i, j + 1)) * avg_rank
            rank += j - i + 1
            i = j + 1
        positives = sum(y)
        negatives = n - positives
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {
        "n": n,
        "agreement": agreement,
        "kappa": kappa,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "acc": agreement,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "auc": auc,
        "glm_positive_rate": py,
        "fourb_positive_rate": pp,
        "mean_support_score": mean(scores),
        "score_methods": dict(Counter(str(r.get("fourb_score_method")) for r in rows)),
    }


def group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {name: metric_rows(group) for name, group in sorted(groups.items())}


def group_metrics_query_type_side(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query_type = str(row.get("query_type") or "unknown")
        side = str(row.get("side") or "unknown")
        groups.setdefault(f"{query_type}::{side}", []).append(row)
    return {name: metric_rows(group) for name, group in sorted(groups.items())}


def glm_gold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    matches = sum(int(bool(r["glm_supports"])) == int(r["gold_label"]) for r in rows)
    positives = [r for r in rows if int(r["gold_label"]) == 1]
    negatives = [r for r in rows if int(r["gold_label"]) == 0]
    return {
        "n": len(rows),
        "agreement": matches / len(rows),
        "positive_gold_glm_supported_rate": mean(int(bool(r["glm_supports"])) for r in positives),
        "positive_gold_glm_unsupported_rate": mean(
            int(not bool(r["glm_supports"])) for r in positives
        ),
        "negative_gold_glm_unsupported_confirmation_rate": mean(
            int(not bool(r["glm_supports"])) for r in negatives
        ),
        "confusion_gold_vs_glm": {
            "gold_positive_glm_supported": sum(bool(r["glm_supports"]) for r in positives),
            "gold_positive_glm_unsupported": sum(not bool(r["glm_supports"]) for r in positives),
            "gold_negative_glm_supported": sum(bool(r["glm_supports"]) for r in negatives),
            "gold_negative_glm_unsupported": sum(not bool(r["glm_supports"]) for r in negatives),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--glm", required=True)
    parser.add_argument("--fourb", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = {str(row["pair_id"]): row for row in load_jsonl(args.pairs)}
    glm = {str(row["pair_id"]): row for row in load_jsonl(args.glm)}
    fourb = {str(row["pair_id"]): row for row in load_jsonl(args.fourb)}
    joined: list[dict[str, Any]] = []
    excluded = Counter()
    for pair_id, pair in pairs.items():
        g = glm.get(pair_id)
        b = fourb.get(pair_id)
        if not g:
            excluded["missing_glm"] += 1
            continue
        if g.get("glm_verdict") in {"skip", "error"} or g.get("glm_supports") is None:
            excluded[str(g.get("glm_verdict") or "glm_missing")] += 1
            continue
        if not b or b.get("fourb_supports") is None:
            excluded["missing_fourb"] += 1
            continue
        row = dict(pair)
        row.update(
            {
                "glm_verdict": g["glm_verdict"],
                "glm_supports": bool(g["glm_supports"]),
                "glm_reason": g.get("glm_reason", ""),
                "glm_raw_content": g.get("glm_raw_content", ""),
                "fourb_verdict": b.get("fourb_verdict"),
                "fourb_supports": bool(b["fourb_supports"]),
                "fourb_text": b.get("fourb_text", ""),
                "fourb_support_score": float(b.get("fourb_support_score", b["fourb_supports"])),
                "fourb_score_method": b.get("fourb_score_method"),
            }
        )
        joined.append(row)
    result = {
        "input_counts": {
            "pairs": len(pairs),
            "glm_rows": len(glm),
            "fourb_rows": len(fourb),
            "joined_usable": len(joined),
            "excluded": dict(excluded),
        },
        "overall": metric_rows(joined),
        "by_query_type": group_metrics(joined, "query_type"),
        "by_side": group_metrics(joined, "side"),
        "by_query_type_side": group_metrics_query_type_side(joined),
        "glm_vs_gold_prior": glm_gold_metrics(joined),
        "metadata": {
            "pairs_file": args.pairs,
            "glm_file": args.glm,
            "fourb_file": args.fourb,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
