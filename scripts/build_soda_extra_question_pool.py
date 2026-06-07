#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


DEFAULT_LISTWISE = Path("data/processed/soda_550_questions_listwise.jsonl")
DEFAULT_EVAL50 = Path("data/processed/eval50_recall_questions_for_soda.jsonl")
DEFAULT_NEGATIVE_CASES = Path("outputs/soda_negative_cases_20260531/negative_cases.jsonl")
DEFAULT_OUTPUT = Path("data/processed/soda_extra_hard_questions_v1.jsonl")

HIGH_SIGNAL_CATEGORIES = {
    "likely_unsupported_cause",
    "entity_mismatch",
    "likely_hallucinated_event",
    "likely_unsupported_motive",
    "entity_or_plan_confusion",
    "premature_answer",
    "over_abstain_or_retrieve",
    "final_abstain",
}


def stable_key(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def candidate_gold(record: dict[str, Any]) -> str:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        return ""
    positives: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("label") or "").lower() == "positive":
            text = str(candidate.get("text") or "").strip()
            if text:
                positives.append(text)
    return "\n".join(positives)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def add_question(
    output: list[dict[str, Any]],
    seen: set[str],
    *,
    question: str,
    source: str,
    prefix: str,
    source_record: dict[str, Any] | None = None,
    category: str = "",
) -> bool:
    question = normalize_question(question)
    if not question or question in seen:
        return False
    seen.add(question)
    record: dict[str, Any] = {
        "question": question,
        "question_key": f"{prefix}_{stable_key(source, question)}",
        "pool_source": source,
    }
    if category:
        record["hard_category"] = category
    if source_record:
        record["source_record"] = source_record
    output.append(record)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extra hard question pool for improved SODA data generation.")
    parser.add_argument("--listwise", type=Path, default=DEFAULT_LISTWISE)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    parser.add_argument("--negative-cases", type=Path, default=DEFAULT_NEGATIVE_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--include-eval50", action="store_true")
    parser.add_argument("--negative-target", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: Counter[str] = Counter()

    if not args.include_eval50:
        for record in read_jsonl(args.eval50):
            question = normalize_question(str(record.get("question") or record.get("query") or ""))
            if question:
                seen.add(question)

    negative_records = read_jsonl(args.negative_cases)
    negative_records.sort(
        key=lambda item: (
            0 if str(item.get("task_type") or "") in {"final_answer", "conclusion_generation"} else 1,
            0 if str(item.get("category") or "") in HIGH_SIGNAL_CATEGORIES else 1,
            str(item.get("question") or ""),
        )
    )
    added_negative = 0
    for record in negative_records:
        if added_negative >= max(0, args.negative_target):
            break
        category = str(record.get("category") or "")
        task_type = str(record.get("task_type") or "")
        if category not in HIGH_SIGNAL_CATEGORIES:
            continue
        if task_type not in {"final_answer", "conclusion_generation"}:
            continue
        question = str(record.get("question") or "")
        if add_question(
            output,
            seen,
            question=question,
            source="negative_case",
            prefix="extra_neg",
            source_record={
                "query_type": "hard_case",
                "answer_focus": str(record.get("reason") or ""),
                "gold": "",
            },
            category=category,
        ):
            added_negative += 1
            stats[f"negative:{category}"] += 1

    listwise_records = read_jsonl(args.listwise)
    rng.shuffle(listwise_records)
    # Prefer causality/reasoning/reveal first; these create more useful conclusion states.
    listwise_records.sort(
        key=lambda item: (
            0 if str(item.get("query_type") or "") in {"causality", "reasoning", "reveal", "mystery"} else 1,
            stable_key(str(item.get("query") or ""), str(args.seed)),
        )
    )
    for record in listwise_records:
        if len(output) >= args.limit:
            break
        question = str(record.get("query") or record.get("question") or "")
        source_record = {
            "query_type": record.get("query_type"),
            "answer_focus": record.get("answer_focus"),
            "gold": candidate_gold(record),
        }
        if add_question(
            output,
            seen,
            question=question,
            source="soda_550_listwise",
            prefix="extra_550",
            source_record=source_record,
            category=str(record.get("query_type") or ""),
        ):
            stats[f"listwise:{record.get('query_type') or 'unknown'}"] += 1

    if len(output) > args.limit:
        output = output[: args.limit]
    write_jsonl(args.output, output)
    summary = {
        "output": str(args.output),
        "questions": len(output),
        "limit": args.limit,
        "include_eval50": args.include_eval50,
        "stats": dict(stats),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
