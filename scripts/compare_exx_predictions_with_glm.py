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
    parse_strict_json_object,
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


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    samples: int = 10_000,
    seed: int = 20260831,
) -> dict[str, float | int | None]:
    """Return a deterministic paired bootstrap interval for the mean delta."""
    if not deltas:
        return {"samples": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    count = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    low_index = max(0, int(samples * 0.025))
    high_index = min(samples - 1, int(samples * 0.975))
    return {
        "samples": count,
        "mean": sum(deltas) / count,
        "ci95_low": means[low_index],
        "ci95_high": means[high_index],
    }


def build_summary(
    completed: list[dict[str, Any]],
    names: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(completed), "models": {}}
    for name in names:
        values = [float(row["models"][name]["score"]) for row in completed]
        eligible_values = [
            float(row["models"][name]["score"])
            for row in completed
            if row["models"][name]["judged"]
        ]
        protocol_values = [
            float(row["models"][name]["protocol_adjusted_score"])
            for row in completed
            if row["models"][name]["protocol_adjusted_score"] is not None
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
            "judged": sum(bool(row["models"][name]["judged"]) for row in completed),
            "judge_failed": sum(
                bool(row["models"][name]["eligible"])
                and not bool(row["models"][name]["judged"])
                for row in completed
            ),
            "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "judge_counts": dict(sorted(counts.items())),
        }
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            eligible_pairs = [
                (
                    float(row["models"][left]["score"]),
                    float(row["models"][right]["score"]),
                )
                for row in completed
                if row["models"][left]["judged"] and row["models"][right]["judged"]
            ]
            protocol_pairs = [
                (
                    float(row["models"][left]["protocol_adjusted_score"]),
                    float(row["models"][right]["protocol_adjusted_score"]),
                )
                for row in completed
                if row["models"][left]["protocol_adjusted_score"] is not None
                and row["models"][right]["protocol_adjusted_score"] is not None
            ]
            semantic_deltas = [a - b for a, b in eligible_pairs]
            protocol_deltas = [a - b for a, b in protocol_pairs]
            pairwise[f"{left}_vs_{right}"] = {
                "both_semantically_eligible": len(eligible_pairs),
                "semantic_left_wins": sum(a > b for a, b in eligible_pairs),
                "semantic_ties": sum(a == b for a, b in eligible_pairs),
                "semantic_right_wins": sum(a < b for a, b in eligible_pairs),
                "semantic_delta": paired_bootstrap_ci(
                    semantic_deltas,
                    samples=bootstrap_samples,
                    seed=seed + left_index,
                ),
                "protocol_adjusted_left_wins": sum(a > b for a, b in protocol_pairs),
                "protocol_adjusted_ties": sum(a == b for a, b in protocol_pairs),
                "protocol_adjusted_right_wins": sum(a < b for a, b in protocol_pairs),
                "protocol_adjusted_delta": paired_bootstrap_ci(
                    protocol_deltas,
                    samples=bootstrap_samples,
                    seed=seed + 100 + left_index,
                ),
            }
    summary["pairwise"] = pairwise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--long-context-threshold", type=int, default=5000)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument(
        "--endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    )
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=0)
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
        max_consecutive_failures=args.max_consecutive_failures,
        ca_bundle=args.ca_bundle,
        allow_duplicate_facts=True,
        strict_json=True,
    )

    by_name = {name: rows for name, rows in loaded}

    def score(index: int) -> dict[str, Any]:
        first = loaded[0][1][index]
        row_id = str(first.get("id"))
        prompt = prompt_value(first)
        prompt_token_count = int(first.get("prompt_token_count") or 0)
        for name in names[1:]:
            if prompt_value(by_name[name][index]) != prompt:
                raise ValueError(f"prompt mismatch for row {row_id}: {name}")
            other_prompt_tokens = int(by_name[name][index].get("prompt_token_count") or 0)
            if other_prompt_tokens != prompt_token_count:
                raise ValueError(f"prompt token count mismatch for row {row_id}: {name}")
        order = blind_order(row_id, names, args.seed)
        completions = [completion_value(by_name[name][index]) for name in order]
        scores = judge.score_group(prompt, completions)
        context = extract_judge_context(prompt)
        indexed: list[tuple[int, dict[str, Any]]] = []
        for rollout_index, completion in enumerate(completions):
            payload = parse_strict_json_object(completion)
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
            payload = parse_strict_json_object(completions[rollout_index])
            eligible = payload is not None and payload_is_judge_eligible(
                payload, context["evidence"], allow_duplicate_facts=True
            )
            models[name] = {
                "blind_index": rollout_index,
                "eligible": eligible,
                "judged": judgement_by_index.get(rollout_index) is not None,
                "score": float(scores[rollout_index]),
                "protocol_adjusted_score": (
                    float(scores[rollout_index])
                    if judgement_by_index.get(rollout_index) is not None
                    else (-1.0 if not eligible else None)
                ),
                "judgement": judgement_by_index.get(rollout_index),
            }
        return {
            "index": index,
            "id": row_id,
            "question": context["question"],
            "round": context["round"],
            "prompt_token_count": prompt_token_count,
            "models": models,
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(score, index): index for index in range(row_count)}
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: item["index"])

    summary = build_summary(
        completed,
        names,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    long_context = [
        row
        for row in completed
        if int(row.get("prompt_token_count") or 0) > args.long_context_threshold
    ]
    summary["subsets"] = {
        f"prompt_tokens_gt_{args.long_context_threshold}": build_summary(
            long_context,
            names,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1000,
        )
    }
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
