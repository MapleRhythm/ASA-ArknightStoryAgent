#!/usr/bin/env python3
"""Score frozen Exx prediction files with the evidence-only GLM judge."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from glm_exx_semantic_reward import GlmEvidenceJudge


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def prompt_value(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"prediction has no prompt: {row.get('id')}")
    return str(conversations[0].get("value") or "")


def completion_value(row: dict[str, Any]) -> str:
    return str(row.get("raw_output") or row.get("output") or row["conversations"][-1].get("value") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument(
        "--endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    )
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")
    rows = read_rows(args.predictions)
    if args.limit > 0:
        rows = rows[: args.limit]
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
    )

    def score(index: int, row: dict[str, Any]) -> dict[str, Any]:
        values = judge.score_group(prompt_value(row), [completion_value(row)])
        return {"index": index, "id": row.get("id"), "score": values[0]}

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(score, index, row): index for index, row in enumerate(rows)}
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: item["index"])
    scores = [float(item["score"]) for item in completed]
    cache_rows = [
        json.loads(line)
        for line in (args.output_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregate = Counter()
    for row in cache_rows:
        aggregate.update(row.get("counts") or {})
    summary = {
        "predictions": str(args.predictions.resolve()),
        "rows": len(completed),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "positive": sum(value > 0 for value in scores),
        "zero": sum(value == 0 for value in scores),
        "negative": sum(value < 0 for value in scores),
        "judge_counts": dict(sorted(aggregate.items())),
        "scores": completed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "scores"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
