#!/usr/bin/env python3
"""Materialize reward-gated gap-follow-up SFT examples.

Only follow-ups with a positive objective coverage gain are used as SFT
targets.  Zero-gain generations remain in the raw JSONL for audit/KTO mining
but are not silently presented as good planner behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_eval(source_name: str, ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64) < ratio


def make_row(item: dict[str, Any], index: int) -> dict[str, Any]:
    query = str(item.get("query") or "").strip()
    qtype = str(item.get("query_type") or "unknown")
    known = str(item.get("known_covered") or "").strip()
    missing = str(item.get("missing_segment") or "").strip()
    follow_up = item.get("gap_followup")
    if not query or not isinstance(follow_up, dict) or not follow_up.get("question"):
        raise ValueError(f"invalid gap item at index {index}")
    source = str(item.get("source_name") or "gap_followup")
    digest = hashlib.sha256(f"{source}\0{query}".encode()).hexdigest()[:20]
    brief = known or "（当前轮没有覆盖到可直接确认的证据）"
    prompt = "\n".join(
        (
            "task: follow_up_hypothesis_generation",
            f"question: {query}",
            f"hypothesis: {compact({'query_type': qtype, 'entities': [], 'keywords': [], 'expected_answer_type': ''})}",
            "round: 2/3",
            "evidence_brief:",
            f"1. 已覆盖证据摘要：{brief[:1800]}",
            f"missing_slots: {missing[:800]}",
            "output_schema: follow_up_hypothesis_v2",
        )
    )
    meta = {
        "generation_mode": "gap_conditioned_glm53_reward_gated",
        "source_name": source,
        "query_type": qtype,
        "coverage_gain": item.get("reward", {}).get("coverage_gain"),
        "novelty": item.get("reward", {}).get("novelty"),
        "missing_segment": missing,
    }
    return {
        "id": f"gap-followup-{digest}",
        "task_type": "follow_up_hypothesis_generation",
        "bucket": "tool",
        "system": SYSTEM,
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": compact(follow_up)},
        ],
        "meta": compact(meta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-coverage-gain", type=float, default=1e-9)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_jsonl(args.input)
    selected = [
        item
        for item in raw
        if float((item.get("reward") or {}).get("coverage_gain") or 0.0) >= args.min_coverage_gain
    ]
    rows = [make_row(item, index) for index, item in enumerate(selected)]
    eval_ids = {
        str(item.get("source_name") or "gap_followup")
        for item in selected
        if stable_eval(str(item.get("source_name") or "gap_followup"), args.eval_ratio, args.seed)
    }
    if not eval_ids and len({str(item.get("source_name") or "") for item in selected}) > 1:
        eval_ids.add(str(selected[0].get("source_name") or "gap_followup"))
    train = [
        row for row in rows
        if str(json.loads(row["meta"]).get("source_name") or "") not in eval_ids
    ]
    val = [
        row for row in rows
        if str(json.loads(row["meta"]).get("source_name") or "") in eval_ids
    ]
    if not train and val:
        train, val = val, []
    for split, values in (("train", train), ("val", val), ("test", [])):
        (args.out_dir / f"{split}.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "gap_followup_sft_train": {
                    "file_name": "train.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
                    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
                },
                "gap_followup_sft_val": {
                    "file_name": "val.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
                    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "source": str(args.input),
        "raw_rows": len(raw),
        "selected_rows": len(rows),
        "selected_coverage_gain_distribution": dict(
            Counter(str((item.get("reward") or {}).get("coverage_gain")) for item in selected)
        ),
        "positive_rows": sum(
            float((item.get("reward") or {}).get("coverage_gain") or 0.0) > 0
            for item in raw
        ),
        "train_rows": len(train),
        "val_rows": len(val),
        "eval_sources": sorted(eval_ids),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
    }
    (args.out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
