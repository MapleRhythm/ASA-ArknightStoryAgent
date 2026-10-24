#!/usr/bin/env python3
"""Export query-only teacher labels as weighted BGE pairwise examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_SCORE = {
    "direct_answer": 1.0,
    "partial_answer": 0.75,
    "supporting_context": 0.45,
    "hard_context": 0.1,
    "irrelevant": 0.0,
}
NEGATIVE_TYPE = {
    "hard_context": "query_only_hard_context",
    "irrelevant": "query_only_irrelevant",
    "supporting_context": "query_only_supporting_context",
    "partial_answer": "query_only_partial_answer",
}
HARD_NEGATIVE_TYPES = {"query_only_hard_context"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hard-negatives-per-positive", type=int, default=3)
    parser.add_argument("--max-irrelevant-per-positive", type=int, default=1)
    parser.add_argument("--max-soft-pairs-per-record", type=int, default=4)
    parser.add_argument(
        "--max-positive-candidates-per-record",
        type=int,
        default=5,
        help="Skip over-broad/ambiguous records whose D+P+S pool exceeds this guard.",
    )
    parser.add_argument(
        "--max-selected-positives-per-record",
        type=int,
        help=(
            "After the record-level guard passes, cap positives used to build pairs. "
            "Selection is deterministic and tier-aware so high-positive records do "
            "not dominate the gradient merely by producing more pairs."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def stable_rng(record: dict[str, Any], seed: int) -> random.Random:
    material = f"{seed}\n{record.get('source_name','')}\n{record.get('query','')}"
    value = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return random.Random(value)


def build_pair(
    record: dict[str, Any], positive: dict[str, Any], negative: dict[str, Any]
) -> dict[str, Any]:
    positive_label = str(positive["label"])
    negative_label = str(negative["label"])
    return {
        "query": str(record["query"]),
        "positive": str(positive["text"]),
        "negative": str(negative["text"]),
        "positive_score": LABEL_SCORE[positive_label],
        "negative_score": LABEL_SCORE[negative_label],
        "positive_label": positive_label,
        "negative_label": negative_label,
        "negative_type": NEGATIVE_TYPE[negative_label],
        "query_type": str(record.get("query_type") or "unknown"),
        "source_name": str(record.get("source_name") or ""),
        "story_family": str(record.get("story_family") or ""),
        "teacher_protocol": str(record.get("teacher_protocol") or ""),
        "teacher_prompt_version": str(record.get("teacher_prompt_version") or ""),
        "teacher_provider": str(record.get("teacher_provider") or ""),
        "teacher_model": str(record.get("teacher_model") or ""),
        # The existing trainer does not know every future negative_type.  This
        # explicit multiplier makes the intended weight portable and auditable.
        "pair_weight_multiplier": 1.5
        if NEGATIVE_TYPE[negative_label] in HARD_NEGATIVE_TYPES
        else 1.0,
    }


def choose_hard_negatives(
    hard_pool: list[dict[str, Any]], positive_rank: int, limit: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample across the teacher's hard-negative frontier, not just its tail."""

    if limit <= 0 or not hard_pool:
        return []
    ordered = sorted(hard_pool, key=lambda item: int(item.get("rank") or 10**9))
    frontier_size = max(limit * 4, min(12, len(ordered)))
    frontier = ordered[:frontier_size]
    rng.shuffle(frontier)
    frontier.sort(key=lambda item: abs(int(item.get("rank") or 10**9) - positive_rank))
    return frontier[:limit]


def choose_positive_candidates(
    pools: dict[str, list[dict[str, Any]]], limit: int | None
) -> dict[str, list[dict[str, Any]]]:
    """Choose a bounded D/P/S subset while retaining answerability tiers.

    For the six-positive setting used by the wider baseline, the initial
    allocation is D/P/S=3/2/1.  Empty tier slots are filled in D>P>S order.
    Within each tier the teacher's existing rank is preserved.
    """

    labels = ("direct_answer", "partial_answer", "supporting_context")
    selected = {label: [] for label in labels}
    total = sum(len(pools[label]) for label in labels)
    if limit is None or limit <= 0 or total <= limit:
        return {label: list(pools[label]) for label in labels}

    ordered = {
        label: sorted(pools[label], key=lambda item: int(item.get("rank") or 10**9))
        for label in labels
    }
    base_quota = {
        "direct_answer": max(1, (limit + 1) // 2),
        "partial_answer": max(1, limit // 3),
        "supporting_context": max(0, limit - ((limit + 1) // 2) - max(1, limit // 3)),
    }
    selected_ids: set[str] = set()
    for label in labels:
        for item in ordered[label][: base_quota[label]]:
            selected[label].append(item)
            selected_ids.add(str(item["candidate_id"]))

    remaining = limit - sum(len(items) for items in selected.values())
    if remaining > 0:
        for label in labels:
            for item in ordered[label]:
                candidate_id = str(item["candidate_id"])
                if candidate_id in selected_ids:
                    continue
                selected[label].append(item)
                selected_ids.add(candidate_id)
                remaining -= 1
                if remaining == 0:
                    break
            if remaining == 0:
                break
    return selected


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = pair_count = 0
    label_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    ambiguous_records = 0
    no_positive_records = 0
    over_broad_records = 0
    no_pair_records = 0
    usable_records = 0
    selected_positive_candidates = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("teacher_protocol") or "") != "query_answerability_v3":
                raise ValueError(f"unexpected protocol at line {line_number}")
            query_decision = record.get("teacher", {}).get("query_decision") or {}
            if str(query_decision.get("status") or "answerable") == "ambiguous":
                ambiguous_records += 1
                record_count += 1
                continue
            candidates = {
                str(item.get("candidate_id") or ""): item
                for item in record.get("candidates") or []
            }
            ranked = record.get("teacher", {}).get("ranking") or []
            if len(ranked) != len(candidates):
                raise ValueError(f"incomplete ranking at line {line_number}")
            pools: dict[str, list[dict[str, Any]]] = {label: [] for label in LABEL_SCORE}
            for judgement in ranked:
                candidate_id = str(judgement.get("candidate_id") or "")
                label = str(judgement.get("label") or "")
                if candidate_id not in candidates or label not in pools:
                    raise ValueError(f"invalid candidate/label at line {line_number}")
                item = {**judgement, "text": candidates[candidate_id]["text"]}
                pools[label].append(item)
                label_counts[label] += 1
            if not any(
                pools[label]
                for label in ("direct_answer", "partial_answer", "supporting_context")
            ):
                no_positive_records += 1
                record_count += 1
                continue
            positive_count = sum(
                len(pools[label])
                for label in ("direct_answer", "partial_answer", "supporting_context")
            )
            if positive_count > args.max_positive_candidates_per_record:
                over_broad_records += 1
                record_count += 1
                continue
            rng = stable_rng(record, args.seed)
            selected_positive_pools = choose_positive_candidates(
                pools, args.max_selected_positives_per_record
            )
            selected_positive_candidates += sum(
                len(selected_positive_pools[label])
                for label in ("direct_answer", "partial_answer", "supporting_context")
            )
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            # Principal objective: all answer-helpful levels must outrank the
            # same-query hard contexts and a small sample of easy negatives.
            for positive_label in ("direct_answer", "partial_answer", "supporting_context"):
                for positive in selected_positive_pools[positive_label]:
                    irrelevant = list(pools["irrelevant"])
                    rng.shuffle(irrelevant)
                    pairs.extend(
                        (positive, negative)
                        for negative in choose_hard_negatives(
                            pools["hard_context"],
                            int(positive.get("rank") or 1),
                            args.max_hard_negatives_per_positive,
                            rng,
                        )
                    )
                    pairs.extend(
                        (positive, negative)
                        for negative in irrelevant[: args.max_irrelevant_per_positive]
                    )
            # A few lower-weight ordinal pairs teach D>P>S without pretending
            # that within-tier teacher rank is a calibrated numerical target.
            soft_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for positive_label, negative_labels in (
                ("direct_answer", ("partial_answer", "supporting_context")),
                ("partial_answer", ("supporting_context",)),
            ):
                for positive in selected_positive_pools[positive_label]:
                    for negative_label in negative_labels:
                        negatives = list(selected_positive_pools[negative_label])
                        rng.shuffle(negatives)
                        if negatives:
                            soft_pairs.append((positive, negatives[0]))
            rng.shuffle(soft_pairs)
            pairs.extend(soft_pairs[: args.max_soft_pairs_per_record])

            seen: set[tuple[str, str]] = set()
            for positive, negative in pairs:
                key = (str(positive["candidate_id"]), str(negative["candidate_id"]))
                if key in seen:
                    continue
                seen.add(key)
                pair = build_pair(record, positive, negative)
                pair_counts[f"{positive['label']}>{negative['label']}"] += 1
                sink.write(json.dumps(pair, ensure_ascii=False) + "\n")
                pair_count += 1
            if not seen:
                no_pair_records += 1
            else:
                usable_records += 1
            record_count += 1
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": record_count,
        "pairs": pair_count,
        "usable_records": usable_records,
        "selected_positive_candidates": selected_positive_candidates,
        "candidate_labels": dict(sorted(label_counts.items())),
        "pair_types": dict(sorted(pair_counts.items())),
        "ambiguous_records_skipped": ambiguous_records,
        "no_positive_records_skipped": no_positive_records,
        "over_broad_records_skipped": over_broad_records,
        "usable_records_with_no_pairs": no_pair_records,
        "max_positive_candidates_per_record": args.max_positive_candidates_per_record,
        "max_selected_positives_per_record": args.max_selected_positives_per_record,
        "label_scores": LABEL_SCORE,
        "answer_fields_exported": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if no_pair_records:
        raise RuntimeError(
            f"{no_pair_records} usable records produced no pairs; see {output_path.with_suffix('.summary.json')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
