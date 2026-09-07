#!/usr/bin/env python3
"""Evaluate stage-level retrieval snapshots without changing inference behavior.

The evaluator expects one JSON object per question (JSON array or JSONL).  The
canonical row shape is:

{
  "id": "q1",
  "question": "...",
  "question_family": "family-a",
  "gold_evidence_ids": ["doc-a", "doc-b"],
  "stages": {
    "dense": ["doc-a", "doc-x"],
    "sparse": ["doc-y", "doc-b"],
    "fusion": ["doc-a", "doc-y", "doc-b"],
    "rerank": ["doc-a", "doc-b"],
    "prompt": ["doc-a"]
  },
  "rounds": [
    {"round": 1, "query": "...", "doc_ids": ["doc-a", "doc-x"]},
    {"round": 2, "query": "...", "doc_ids": ["doc-b", "doc-z"]}
  ]
}

Stage lists may also contain objects with ``id``, ``doc_id`` or
``evidence_id``.  This tool deliberately does not infer a missing stage from a
different stage: a historical result with only ``first_hit_rank`` is not a
five-stage snapshot and must be regenerated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGES = ("dense", "sparse", "fusion", "rerank", "prompt")
STAGE_ALIASES = {
    "dense": ("dense", "dense_hits", "dense_ids"),
    "sparse": ("sparse", "sparse_hits", "sparse_ids", "bm25", "bm25_hits"),
    "fusion": ("fusion", "fusion_hits", "fusion_ids", "candidate_pool"),
    "rerank": ("rerank", "rerank_hits", "rerank_ids", "reranked"),
    "prompt": (
        "prompt",
        "prompt_evidence",
        "prompt_evidence_ids",
        "final_prompt",
        "final_prompt_evidence",
    ),
}
ROUND_ALIASES = ("rounds", "retrieval_rounds", "retrieval_trace")
GOLD_ALIASES = (
    "gold_evidence_ids",
    "gold_ids",
    "answer_evidence_ids",
    "gold_evidence",
)
ID_FIELDS = ("id", "doc_id", "evidence_id", "document_id")


class SnapshotError(ValueError):
    """Raised when a row is not an auditable stage snapshot."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            for key in ("rows", "records", "results", "items"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
            else:
                values = [value]
        else:
            raise SnapshotError(f"expected JSON object/array: {path}")
    if not all(isinstance(row, dict) for row in values):
        raise SnapshotError(f"all rows must be JSON objects: {path}")
    return [row for row in values if isinstance(row, dict)]


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _id_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return str(item)
    if isinstance(item, dict):
        value = _first_present(item, ID_FIELDS)
        if value is not None:
            return str(value).strip()
    return ""


def ordered_ids(value: Any) -> list[str]:
    """Normalize a ranked ID list while preserving first occurrence order."""
    if isinstance(value, dict):
        for key in ("ids", "doc_ids", "evidence_ids", "items", "hits", "documents"):
            if key in value:
                value = value[key]
                break
        else:
            value = list(value)
    if not isinstance(value, (list, tuple)):
        raise SnapshotError("stage/round document IDs must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        item_id = _id_from_item(item)
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        result.append(item_id)
    return result


def _nested_stage_container(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("stages", "retrieval_snapshot", "snapshot"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return row


def extract_gold_ids(row: dict[str, Any]) -> list[str]:
    value = _first_present(row, GOLD_ALIASES)
    if value is None and isinstance(row.get("gold"), dict):
        value = _first_present(row["gold"], GOLD_ALIASES + ("evidence",))
    if value is None:
        raise SnapshotError("missing gold_evidence_ids")
    ids = ordered_ids(value)
    if not ids:
        raise SnapshotError("gold_evidence_ids must not be empty")
    return ids


def extract_stages(row: dict[str, Any]) -> dict[str, list[str]]:
    container = _nested_stage_container(row)
    stages: dict[str, list[str]] = {}
    missing: list[str] = []
    for stage in STAGES:
        value = _first_present(container, STAGE_ALIASES[stage])
        if value is None and container is not row:
            value = _first_present(row, STAGE_ALIASES[stage])
        if value is None:
            missing.append(stage)
            continue
        stages[stage] = ordered_ids(value)
    if missing:
        raise SnapshotError("missing stages: " + ",".join(missing))
    return stages


def _round_doc_ids(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = _first_present(
            value,
            ("doc_ids", "evidence_ids", "top_doc_ids", "documents", "evidence_summary"),
        )
    return ordered_ids(value)


def extract_rounds(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = _first_present(row, ROUND_ALIASES)
    if value is None:
        return []
    if not isinstance(value, list):
        raise SnapshotError("rounds/retrieval_trace must be a list")
    rounds: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise SnapshotError(f"round {index} must be an object")
        ids_value = _first_present(
            item,
            ("doc_ids", "evidence_ids", "top_doc_ids", "documents", "evidence_summary"),
        )
        if ids_value is None:
            raise SnapshotError(f"round {index} missing doc_ids")
        rounds.append(
            {
                "round": item.get("round", index),
                "query": str(
                    _first_present(item, ("query", "question", "retrieval_query")) or ""
                ).strip(),
                "doc_ids": _round_doc_ids(ids_value),
            }
        )
    return rounds


def normalize_question(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip("？?。！!，,；;：:\"'")


def row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("task_id") or index)


def family_key(row: dict[str, Any]) -> str:
    explicit = row.get("question_family") or row.get("family") or row.get("family_id")
    if explicit:
        return str(explicit)
    question = normalize_question(row.get("question"))
    return question or "<unknown-family>"


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)


def all_gold_at_k(gold: set[str], ranked: list[str], k: int) -> bool:
    return bool(gold) and gold.issubset(ranked[:k])


def any_gold_at_k(gold: set[str], ranked: list[str], k: int) -> bool:
    return bool(gold.intersection(ranked[:k]))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def stage_record(gold: set[str], ranked: list[str], top_ks: list[int]) -> dict[str, Any]:
    return {
        "count": 1,
        "candidate_count": len(ranked),
        "gold_count": len(gold),
        "recall": {f"@{k}": recall_at_k(gold, ranked, k) for k in top_ks},
        "any_hit": {f"@{k}": int(any_gold_at_k(gold, ranked, k)) for k in top_ks},
        "all_gold": {f"@{k}": int(all_gold_at_k(gold, ranked, k)) for k in top_ks},
    }


def retrieval_coverage_attribution(
    gold: set[str], stages: dict[str, list[str]]
) -> str:
    """Locate the first stage where a complete gold set is unavailable."""
    if not gold.issubset(stages["fusion"]):
        return "candidate_pool_missing"
    if not gold.issubset(stages["rerank"]):
        return "rerank_drop"
    if not gold.issubset(stages["prompt"]):
        return "prompt_drop"
    return "retrieval_coverage_complete"


def chain_length_bucket(gold_count: int) -> str:
    if gold_count <= 1:
        return "1"
    if gold_count == 2:
        return "2"
    return "3+"


def merge_stage_records(records: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "candidate_count_mean": 0.0,
            "gold_count_mean": 0.0,
            "recall": {f"@{k}": 0.0 for k in top_ks},
            "any_hit_rate": {f"@{k}": 0.0 for k in top_ks},
            "all_gold_rate": {f"@{k}": 0.0 for k in top_ks},
        }
    count = len(records)
    return {
        "count": count,
        "candidate_count_mean": sum(row["candidate_count"] for row in records) / count,
        "gold_count_mean": sum(row["gold_count"] for row in records) / count,
        "recall": {
            f"@{k}": sum(row["recall"][f"@{k}"] for row in records) / count
            for k in top_ks
        },
        "any_hit_rate": {
            f"@{k}": sum(row["any_hit"][f"@{k}"] for row in records) / count
            for k in top_ks
        },
        "all_gold_rate": {
            f"@{k}": sum(row["all_gold"][f"@{k}"] for row in records) / count
            for k in top_ks
        },
    }


def macro_average_stage_records(
    family_records: dict[str, list[dict[str, Any]]], top_ks: list[int]
) -> dict[str, Any]:
    """Average stage metrics with each question family receiving equal weight."""
    summaries = [
        merge_stage_records(records, top_ks)
        for _, records in sorted(family_records.items())
        if records
    ]
    if not summaries:
        return {
            "family_count": 0,
            "recall": {f"@{k}": 0.0 for k in top_ks},
            "any_hit_rate": {f"@{k}": 0.0 for k in top_ks},
            "all_gold_rate": {f"@{k}": 0.0 for k in top_ks},
        }
    count = len(summaries)
    return {
        "family_count": count,
        "recall": {
            f"@{k}": sum(item["recall"][f"@{k}"] for item in summaries) / count
            for k in top_ks
        },
        "any_hit_rate": {
            f"@{k}": sum(item["any_hit_rate"][f"@{k}"] for item in summaries) / count
            for k in top_ks
        },
        "all_gold_rate": {
            f"@{k}": sum(item["all_gold_rate"][f"@{k}"] for item in summaries) / count
            for k in top_ks
        },
    }


def round_metrics(rounds: list[dict[str, Any]], gold: set[str], top_k: int) -> dict[str, Any]:
    if not rounds:
        return {"round_count": 0, "transitions": []}
    transitions: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    seen_gold: set[str] = set()
    round_rows: list[dict[str, Any]] = []
    previous_ids: list[str] | None = None
    previous_query = ""
    for position, current in enumerate(rounds, start=1):
        current_ids = current["doc_ids"]
        current_set = set(current_ids[:top_k])
        current_gold = current_set & gold
        fresh_docs = current_set - seen_docs
        fresh_gold = current_gold - seen_gold
        row = {
            "round": current["round"],
            "query": current["query"],
            "doc_count": len(current_ids),
            "gold_hit_count": len(current_gold),
            "fresh_gold_hit_count": len(fresh_gold),
            "fresh_doc_count": len(fresh_docs),
            "fresh_doc_rate": len(fresh_docs) / len(current_set) if current_set else 0.0,
            "fresh_recall": len(fresh_gold) / len(gold) if gold else 0.0,
            "cumulative_recall": len(seen_gold | current_gold) / len(gold) if gold else 0.0,
        }
        if previous_ids is not None:
            normalized_query = normalize_question(current["query"])
            row["previous_top_k_jaccard"] = jaccard(previous_ids[:top_k], current_ids[:top_k])
            row["query_exact_duplicate"] = int(
                bool(normalized_query) and normalized_query == normalize_question(previous_query)
            )
        round_rows.append(row)
        seen_docs.update(current_set)
        seen_gold.update(current_gold)
        previous_ids = current_ids
        previous_query = current["query"]
    for current, previous in zip(round_rows[1:], round_rows[:-1]):
        transitions.append(
            {
                "from_round": previous["round"],
                "to_round": current["round"],
                "top_k_jaccard": current.get("previous_top_k_jaccard", 1.0),
                "fresh_doc_rate": current["fresh_doc_rate"],
                "fresh_recall": current["fresh_recall"],
                "query_exact_duplicate": current.get("query_exact_duplicate", 0),
            }
        )
    return {
        "round_count": len(rounds),
        "rounds": round_rows,
        "transitions": transitions,
        "mean_transition_top_k_jaccard": (
            sum(row["top_k_jaccard"] for row in transitions) / len(transitions)
            if transitions
            else None
        ),
        "mean_transition_fresh_doc_rate": (
            sum(row["fresh_doc_rate"] for row in transitions) / len(transitions)
            if transitions
            else None
        ),
        "mean_transition_fresh_recall": (
            sum(row["fresh_recall"] for row in transitions) / len(transitions)
            if transitions
            else None
        ),
        "query_exact_duplicate_rate": (
            sum(row["query_exact_duplicate"] for row in transitions) / len(transitions)
            if transitions
            else None
        ),
    }


def evaluate_rows(rows: list[dict[str, Any]], top_ks: list[int], round_top_k: int) -> dict[str, Any]:
    if not rows:
        raise SnapshotError("no rows")
    top_ks = sorted({int(k) for k in top_ks if int(k) > 0})
    if not top_ks:
        raise SnapshotError("top_ks must contain a positive integer")
    stage_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_stage_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    query_type_stage_records: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    chain_length_stage_records: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    attribution_counts: Counter[str] = Counter()
    row_outputs: list[dict[str, Any]] = []
    round_outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        key = row_key(row, index)
        try:
            gold_ids = extract_gold_ids(row)
            stages = extract_stages(row)
            family = family_key(row)
            query_type = str(row.get("query_type") or "unknown")
        except SnapshotError as exc:
            failures.append({"id": key, "error": str(exc)})
            continue
        gold = set(gold_ids)
        per_stage = {
            stage: stage_record(gold, stages[stage], top_ks) for stage in STAGES
        }
        for stage in STAGES:
            stage_records[stage].append(per_stage[stage])
            family_stage_records[family][stage].append(per_stage[stage])
            query_type_stage_records[query_type][stage].append(per_stage[stage])
            chain_length_stage_records[chain_length_bucket(len(gold))][stage].append(
                per_stage[stage]
            )
        attribution = retrieval_coverage_attribution(gold, stages)
        attribution_counts[attribution] += 1
        row_outputs.append(
            {
                "id": key,
                "question": str(row.get("question") or ""),
                "question_family": family,
                "query_type": query_type,
                "gold_evidence_ids": gold_ids,
                "gold_chain_length": len(gold_ids),
                "coverage_attribution": attribution,
                "stages": {
                    stage: {
                        "candidate_count": len(stages[stage]),
                        "recall": per_stage[stage]["recall"],
                        "any_hit": per_stage[stage]["any_hit"],
                        "all_gold": per_stage[stage]["all_gold"],
                    }
                    for stage in STAGES
                },
            }
        )
        rounds = extract_rounds(row)
        if rounds:
            round_outputs.append(
                {
                    "id": key,
                    "question_family": family,
                    **round_metrics(rounds, gold, round_top_k),
                }
            )
    if not row_outputs:
        raise SnapshotError("no valid stage snapshots; failures=" + str(failures[:3]))
    summary = {
        "rows_total": len(rows),
        "rows_evaluable": len(row_outputs),
        "rows_failed": len(failures),
        "failure_examples": failures[:20],
        "top_ks": top_ks,
        "round_top_k": round_top_k,
        "stages": {
            stage: merge_stage_records(stage_records[stage], top_ks) for stage in STAGES
        },
        "family_macro": {
            family: {
                stage: merge_stage_records(family_stage_records[family][stage], top_ks)
                for stage in STAGES
            }
            for family in sorted(family_stage_records)
        },
        "family_macro_mean": {
            stage: macro_average_stage_records(
                {
                    family: family_stage_records[family][stage]
                    for family in family_stage_records
                },
                top_ks,
            )
            for stage in STAGES
        },
        "by_query_type": {
            query_type: {
                stage: merge_stage_records(
                    query_type_stage_records[query_type][stage], top_ks
                )
                for stage in STAGES
            }
            for query_type in sorted(query_type_stage_records)
        },
        "by_gold_chain_length": {
            bucket: {
                stage: merge_stage_records(
                    chain_length_stage_records[bucket][stage], top_ks
                )
                for stage in STAGES
            }
            for bucket in ("1", "2", "3+")
            if bucket in chain_length_stage_records
        },
        "coverage_attribution": dict(sorted(attribution_counts.items())),
        "question_family_count": len(family_stage_records),
        "rounds": {
            "rows_with_rounds": len(round_outputs),
            "mean_transition_top_k_jaccard": _mean_or_none(
                row.get("mean_transition_top_k_jaccard") for row in round_outputs
            ),
            "mean_transition_fresh_doc_rate": _mean_or_none(
                row.get("mean_transition_fresh_doc_rate") for row in round_outputs
            ),
            "mean_transition_fresh_recall": _mean_or_none(
                row.get("mean_transition_fresh_recall") for row in round_outputs
            ),
            "query_exact_duplicate_rate": _mean_or_none(
                row.get("query_exact_duplicate_rate") for row in round_outputs
            ),
        },
        "rows": row_outputs,
        "round_rows": round_outputs,
    }
    return summary


def _mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", action="append", type=int, dest="top_ks")
    parser.add_argument("--round-top-k", type=int, default=5)
    args = parser.parse_args()
    top_ks = args.top_ks or [1, 3, 5, 10, 20]
    result = evaluate_rows(read_rows(args.snapshots), top_ks, max(1, args.round_top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "rows_total",
                    "rows_evaluable",
                    "rows_failed",
                    "question_family_count",
                    "stages",
                    "rounds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
