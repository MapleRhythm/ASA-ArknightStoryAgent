#!/usr/bin/env python3
"""Build a deterministic, length-safe, task/action-stratified Exx smoke set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def action(row: dict[str, Any]) -> str:
    if row.get("task_type") != "grounded_action_generation":
        return "-"
    value = json.loads(row["conversations"][-1]["value"])
    return str(value.get("next_action") or "")


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_type") or ""), action(row)


def stable_order(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("id") or "").encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--total", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/zhb/ASA-ArknightStoryAgent/model/qwen3.5-4b",
        trust_remote_code=True,
    )
    eligible: list[dict[str, Any]] = []
    lengths: dict[str, int] = {}
    for row in rows:
        messages = [
            {"role": "system", "content": str(row.get("system") or "")},
            {"role": "user", "content": str(row["conversations"][0].get("value") or "")},
            {"role": "assistant", "content": str(row["conversations"][-1].get("value") or "")},
        ]
        token_count = len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, enable_thinking=False
            )
        )
        if token_count <= args.max_tokens:
            item = dict(row)
            item["_token_count"] = token_count
            eligible.append(item)
            lengths[str(row.get("id"))] = token_count

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[key(row)].append(row)
    for values in groups.values():
        values.sort(key=stable_order)

    # Preserve the full eligible-set mix while guaranteeing representation.
    desired = {
        ("user_question_hypothesis_generation", "-"): 18,
        ("follow_up_hypothesis_generation", "-"): 18,
        ("grounded_action_generation", "answer_directly"): 123,
        ("grounded_action_generation", "retrieve_more"): 36,
        ("grounded_action_generation", "abstain"): 31,
    }
    selected: list[dict[str, Any]] = []
    for group, quota in desired.items():
        selected.extend(groups.get(group, [])[:quota])
    if len(selected) < args.total:
        remaining = [
            row
            for row in eligible
            if str(row.get("id")) not in {str(item.get("id")) for item in selected}
        ]
        remaining.sort(key=stable_order)
        selected.extend(remaining[: args.total - len(selected)])
    selected = selected[: args.total]
    selected.sort(key=lambda row: stable_order(row))
    for row in selected:
        row.pop("_token_count", None)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "source": str(args.source),
        "max_tokens_inclusive": args.max_tokens,
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "selected_task_action_counts": {
            f"{task}:{act}": count
            for (task, act), count in sorted(Counter(key(row) for row in selected).items())
        },
        "selected_token_min": min(lengths[str(row["id"])] for row in selected),
        "selected_token_max": max(lengths[str(row["id"])] for row in selected),
        "selected_token_mean": sum(lengths[str(row["id"])] for row in selected) / len(selected),
        "selection": "stable sha256(id), fixed task/action quotas, no truncation",
        "seed": args.seed,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
