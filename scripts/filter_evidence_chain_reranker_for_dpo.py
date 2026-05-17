#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chain_set(record: dict[str, Any], key: str) -> set[str]:
    value = record.get(key) or []
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def score(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return default


def build_prompt(record: dict[str, Any]) -> str:
    query = str(record.get("query") or "").strip()
    query_type = str(record.get("query_type") or "").strip()
    answer_focus = str(record.get("answer_focus") or "").strip()
    parts = [
        "判断哪条证据链更能充分、直接、完整回答用户问题。",
        f"用户问题：{query}",
    ]
    if query_type:
        parts.append(f"问题类别：{query_type}")
    if answer_focus:
        parts.append(f"答案焦点：{answer_focus}")
    parts.append("优先选择包含答案揭示点、因果/时序完整、背景噪声更少的证据链。")
    return "\n".join(parts)


def pairwise_drop_reason(
    record: dict[str, Any],
    *,
    max_negative_score: float,
    answer_evidence_score_threshold: float,
) -> str | None:
    positive = str(record.get("positive") or "").strip()
    negative = str(record.get("negative") or "").strip()
    if not positive or not negative:
        return "empty_text"
    if positive == negative:
        return "same_text"
    positive_chain = chain_set(record, "positive_chain")
    negative_chain = chain_set(record, "negative_chain")
    if positive_chain and negative_chain and positive_chain == negative_chain:
        return "same_evidence_set"
    positive_score = score(record, "positive_score", 1.0)
    negative_score = score(record, "negative_score", 0.0)
    if positive_score <= negative_score:
        return "non_positive_margin"
    if negative_score >= max_negative_score:
        return "negative_score_too_high"
    answer_evidence = set(str(item) for item in (record.get("answer_evidence") or []) if str(item))
    if answer_evidence and answer_evidence.issubset(negative_chain) and negative_score > answer_evidence_score_threshold:
        return "negative_contains_all_answer_evidence"
    return None


def candidate_drop_reason(
    candidate: dict[str, Any],
    gold: dict[str, Any],
    answer_evidence: set[str],
    *,
    max_negative_score: float,
    answer_evidence_score_threshold: float,
) -> str | None:
    text = str(candidate.get("text_with_metadata") or candidate.get("text") or "").strip()
    gold_text = str(gold.get("text_with_metadata") or gold.get("text") or "").strip()
    if not text:
        return "empty_text"
    if text == gold_text:
        return "same_text"
    chain = {str(item) for item in (candidate.get("chain") or []) if str(item)}
    gold_chain = {str(item) for item in (gold.get("chain") or []) if str(item)}
    if chain and gold_chain and chain == gold_chain:
        return "same_evidence_set"
    candidate_score = score(candidate, "score", 0.0)
    gold_score = score(gold, "score", 1.0)
    if gold_score <= candidate_score:
        return "non_positive_margin"
    if candidate_score >= max_negative_score:
        return "negative_score_too_high"
    if answer_evidence and answer_evidence.issubset(chain) and candidate_score > answer_evidence_score_threshold:
        return "negative_contains_all_answer_evidence"
    return None


def filter_listwise(
    records: list[dict[str, Any]],
    *,
    max_negative_score: float,
    answer_evidence_score_threshold: float,
    min_negatives: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for record in records:
        stats["input_listwise"] += 1
        candidates = record.get("candidates") or []
        positives = [
            candidate
            for candidate in candidates
            if candidate.get("label") == "positive" or candidate.get("type") == "gold"
        ]
        if len(positives) != 1:
            stats["drop_sample_gold_count"] += 1
            continue
        gold = positives[0]
        answer_evidence = {str(item) for item in (record.get("answer_evidence") or []) if str(item)}
        kept_candidates = [gold]
        for candidate in candidates:
            if candidate is gold:
                continue
            if candidate.get("label") != "negative" or candidate.get("type") == "gold":
                stats["drop_candidate_not_negative"] += 1
                continue
            reason = candidate_drop_reason(
                candidate,
                gold,
                answer_evidence,
                max_negative_score=max_negative_score,
                answer_evidence_score_threshold=answer_evidence_score_threshold,
            )
            if reason is not None:
                stats[f"drop_candidate_{reason}"] += 1
                continue
            kept_candidates.append(candidate)
            stats["kept_negative_candidate"] += 1
        if len(kept_candidates) - 1 < min_negatives:
            stats["drop_sample_too_few_negatives"] += 1
            continue
        next_record = dict(record)
        next_record["candidates"] = kept_candidates
        filtered.append(next_record)
        stats["kept_listwise"] += 1
    return filtered, stats


def filter_pairwise(
    records: list[dict[str, Any]],
    *,
    max_negative_score: float,
    answer_evidence_score_threshold: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        stats["input_pairwise"] += 1
        reason = pairwise_drop_reason(
            record,
            max_negative_score=max_negative_score,
            answer_evidence_score_threshold=answer_evidence_score_threshold,
        )
        if reason is not None:
            stats[f"drop_pairwise_{reason}"] += 1
            continue
        key = (
            str(record.get("query") or ""),
            str(record.get("positive") or ""),
            str(record.get("negative") or ""),
        )
        if key in seen:
            stats["drop_pairwise_duplicate"] += 1
            continue
        seen.add(key)
        filtered.append(record)
        stats["kept_pairwise"] += 1
    return filtered, stats


def convert_to_flag_records(listwise_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in listwise_records:
        positives: list[str] = []
        negatives: list[str] = []
        for candidate in record.get("candidates") or []:
            text = str(candidate.get("text_with_metadata") or candidate.get("text") or "").strip()
            if not text:
                continue
            if candidate.get("label") == "positive" or candidate.get("type") == "gold":
                positives.append(text)
            elif candidate.get("label") == "negative" and candidate.get("type") != "gold":
                negatives.append(text)
        if positives and negatives:
            records.append(
                {
                    "query": record.get("query", ""),
                    "answer": record.get("answer", ""),
                    "pos": positives,
                    "neg": negatives,
                }
            )
    return records


def convert_to_dpo_records(pairwise_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dpo_records: list[dict[str, Any]] = []
    for record in pairwise_records:
        chosen = str(record.get("positive") or "").strip()
        rejected = str(record.get("negative") or "").strip()
        if not chosen or not rejected or chosen == rejected:
            continue
        dpo_records.append(
            {
                "prompt": build_prompt(record),
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "query": record.get("query", ""),
                    "query_type": record.get("query_type", ""),
                    "source_name": record.get("source_name", ""),
                    "negative_type": record.get("negative_type", ""),
                    "positive_score": record.get("positive_score"),
                    "negative_score": record.get("negative_score"),
                    "positive_chain": record.get("positive_chain", []),
                    "negative_chain": record.get("negative_chain", []),
                    "positive_chain_role_tags": record.get("positive_chain_role_tags", []),
                    "negative_chain_role_tags": record.get("negative_chain_role_tags", []),
                    "positive_chain_structure": record.get("positive_chain_structure", {}),
                    "negative_chain_structure": record.get("negative_chain_structure", {}),
                    "answer": record.get("answer", ""),
                    "answer_evidence": record.get("answer_evidence", []),
                    "answer_focus": record.get("answer_focus", ""),
                },
            }
        )
    return dpo_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conservative DPO-training copy from evidence-chain reranker exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v2"),
        help="Source reranker export directory. This directory is read-only for this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v2_dpo_filtered"),
        help="Output directory for filtered DPO/listwise/pairwise files.",
    )
    parser.add_argument(
        "--max-negative-score",
        type=float,
        default=0.9,
        help="Drop negatives with score >= this value.",
    )
    parser.add_argument(
        "--answer-evidence-score-threshold",
        type=float,
        default=0.6,
        help="Drop negatives that contain all answer_evidence and score above this value.",
    )
    parser.add_argument(
        "--min-negatives",
        type=int,
        default=1,
        help="Minimum remaining negatives required to keep a listwise sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = read_jsonl(input_dir / "annotations.cleaned.jsonl")
    listwise = read_jsonl(input_dir / "reranker_listwise.jsonl")
    pairwise = read_jsonl(input_dir / "reranker_pairwise.jsonl")
    validation_issues = read_jsonl(input_dir / "validation_issues.jsonl")
    manifest = {}
    if (input_dir / "manifest.json").exists():
        manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))

    filtered_listwise, listwise_stats = filter_listwise(
        listwise,
        max_negative_score=args.max_negative_score,
        answer_evidence_score_threshold=args.answer_evidence_score_threshold,
        min_negatives=args.min_negatives,
    )
    filtered_pairwise, pairwise_stats = filter_pairwise(
        pairwise,
        max_negative_score=args.max_negative_score,
        answer_evidence_score_threshold=args.answer_evidence_score_threshold,
    )
    flag_records = convert_to_flag_records(filtered_listwise)
    dpo_records = convert_to_dpo_records(filtered_pairwise)

    write_jsonl(output_dir / "annotations.cleaned.jsonl", annotations)
    write_jsonl(output_dir / "reranker_listwise.jsonl", filtered_listwise)
    write_jsonl(output_dir / "reranker_pairwise.jsonl", filtered_pairwise)
    write_jsonl(output_dir / "flag_embedding_reranker.jsonl", flag_records)
    write_jsonl(output_dir / "reranker_dpo.jsonl", dpo_records)
    write_jsonl(output_dir / "validation_issues.source.jsonl", validation_issues)

    summary = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_usage": "Keep source_dir unchanged as MiniRAG relation source; use output_dir for DPO/reranker training.",
        "filter": {
            "drop_same_evidence_set_as_gold": True,
            "drop_positive_equal_negative_text": True,
            "drop_non_positive_margin": True,
            "drop_negative_score_gte": args.max_negative_score,
            "drop_negative_contains_all_answer_evidence_when_score_gt": args.answer_evidence_score_threshold,
            "min_negatives_per_listwise_sample": args.min_negatives,
        },
        "input_manifest": manifest,
        "input_counts": {
            "annotations": len(annotations),
            "listwise_records": len(listwise),
            "pairwise_records": len(pairwise),
            "validation_issues": len(validation_issues),
        },
        "output_counts": {
            "annotations": len(annotations),
            "listwise_records": len(filtered_listwise),
            "pairwise_records": len(filtered_pairwise),
            "flag_embedding_records": len(flag_records),
            "dpo_records": len(dpo_records),
        },
        "listwise_filter_stats": dict(listwise_stats),
        "pairwise_filter_stats": dict(pairwise_stats),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
