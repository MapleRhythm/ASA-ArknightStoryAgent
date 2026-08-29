#!/usr/bin/env python3
"""Blindly compare aligned Exx prediction files with one GLM call per row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from glm_exx_semantic_reward import (
    GlmEvidenceJudge,
    extract_judge_context,
    judgement_counts,
    parse_json_object,
    payload_is_judge_eligible,
)
from score_exx_predictions_with_glm import completion_value, prompt_value, read_rows


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=/absolute/path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_absolute():
        raise argparse.ArgumentTypeError("prediction must be NAME=/absolute/path")
    return name, path


def aligned_rows(inputs: list[tuple[str, Path]]) -> list[tuple[str, list[dict[str, Any]]]]:
    if len(inputs) < 2:
        raise ValueError("at least two prediction files are required")
    loaded = [(name, read_rows(path)) for name, path in inputs]
    expected_ids = [str(row.get("id")) for row in loaded[0][1]]
    for name, rows in loaded[1:]:
        ids = [str(row.get("id")) for row in rows]
        if ids != expected_ids:
            raise ValueError(f"prediction rows are not ID-aligned: {name}")
    return loaded


def blind_order(row_id: str, names: list[str], seed: int) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    result = list(names)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument(
        "--endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    )
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--ca-bundle", type=Path)
    args = parser.parse_args()

    names = [name for name, _ in args.prediction]
    if len(set(names)) != len(names):
        parser.error("prediction names must be unique")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")
    loaded = aligned_rows(args.prediction)
    row_count = len(loaded[0][1])
    if args.limit > 0:
        row_count = min(row_count, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    judge = GlmEvidenceJudge(
        endpoint=args.endpoint,
        api_key=api_key,
        model=args.model,
        cache_path=args.output_dir / "cache.jsonl",
        failures_path=args.output_dir / "failures.jsonl",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        reasoning_effort=args.reasoning_effort,
        workers=1,
        max_consecutive_failures=max(3, args.workers),
        ca_bundle=args.ca_bundle,
        allow_duplicate_facts=True,
    )

    by_name = {name: rows for name, rows in loaded}

    def score(index: int) -> dict[str, Any]:
        first = loaded[0][1][index]
        row_id = str(first.get("id"))
        prompt = prompt_value(first)
        for name in names[1:]:
            if prompt_value(by_name[name][index]) != prompt:
                raise ValueError(f"prompt mismatch for row {row_id}: {name}")
        order = blind_order(row_id, names, args.seed)
        completions = [completion_value(by_name[name][index]) for name in order]
        scores = judge.score_group(prompt, completions)
        context = extract_judge_context(prompt)
        indexed: list[tuple[int, dict[str, Any]]] = []
        for rollout_index, completion in enumerate(completions):
            payload = parse_json_object(completion)
            if payload is not None and payload_is_judge_eligible(
                payload, context["evidence"], allow_duplicate_facts=True
            ):
                indexed.append((rollout_index, payload))
        cache = None
        if indexed:
            cache = judge._cache.get(judge._cache_key(context, indexed))
        judgement_by_index = {
            int(row["rollout_index"]): row
            for row in ((cache or {}).get("judgement") or {}).get("rollouts", [])
        }
        models: dict[str, dict[str, Any]] = {}
        for rollout_index, name in enumerate(order):
            payload = parse_json_object(completions[rollout_index])
            eligible = payload is not None and payload_is_judge_eligible(
                payload, context["evidence"], allow_duplicate_facts=True
            )
            models[name] = {
                "blind_index": rollout_index,
                "eligible": eligible,
                "score": float(scores[rollout_index]),
                "protocol_adjusted_score": (
                    float(scores[rollout_index]) if eligible else -1.0
                ),
                "judgement": judgement_by_index.get(rollout_index),
            }
        return {
            "index": index,
            "id": row_id,
            "question": context["question"],
            "round": context["round"],
            "models": models,
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(score, index): index for index in range(row_count)}
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: item["index"])

    summary: dict[str, Any] = {"rows": row_count, "models": {}}
    for name in names:
        values = [float(row["models"][name]["score"]) for row in completed]
        eligible_values = [
            float(row["models"][name]["score"])
            for row in completed
            if row["models"][name]["eligible"]
        ]
        protocol_values = [
            float(row["models"][name]["protocol_adjusted_score"])
            for row in completed
        ]
        counts: Counter[str] = Counter()
        for row in completed:
            judgement = row["models"][name].get("judgement")
            if judgement:
                counts.update(judgement_counts({"rollouts": [judgement]}))
        summary["models"][name] = {
            "mean_score": sum(values) / len(values) if values else 0.0,
            "mean_score_on_eligible": (
                sum(eligible_values) / len(eligible_values) if eligible_values else 0.0
            ),
            "protocol_adjusted_mean_score": (
                sum(protocol_values) / len(protocol_values) if protocol_values else 0.0
            ),
            "eligible": sum(bool(row["models"][name]["eligible"]) for row in completed),
            "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "judge_counts": dict(sorted(counts.items())),
        }
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_scores = [float(row["models"][left]["score"]) for row in completed]
            right_scores = [float(row["models"][right]["score"]) for row in completed]
            eligible_pairs = [
                (a, b)
                for row, a, b in zip(completed, left_scores, right_scores, strict=True)
                if row["models"][left]["eligible"] and row["models"][right]["eligible"]
            ]
            protocol_pairs = [
                (
                    float(row["models"][left]["protocol_adjusted_score"]),
                    float(row["models"][right]["protocol_adjusted_score"]),
                )
                for row in completed
            ]
            pairwise[f"{left}_vs_{right}"] = {
                "both_semantically_eligible": len(eligible_pairs),
                "semantic_left_wins": sum(a > b for a, b in eligible_pairs),
                "semantic_ties": sum(a == b for a, b in eligible_pairs),
                "semantic_right_wins": sum(a < b for a, b in eligible_pairs),
                "semantic_mean_delta_left_minus_right": (
                    sum(a - b for a, b in eligible_pairs) / len(eligible_pairs)
                    if eligible_pairs
                    else 0.0
                ),
                "protocol_adjusted_left_wins": sum(a > b for a, b in protocol_pairs),
                "protocol_adjusted_ties": sum(a == b for a, b in protocol_pairs),
                "protocol_adjusted_right_wins": sum(a < b for a, b in protocol_pairs),
            }
    summary["pairwise"] = pairwise
    (args.output_dir / "scored.json").write_text(
        json.dumps(completed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
