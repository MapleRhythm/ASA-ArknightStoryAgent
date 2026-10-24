#!/usr/bin/env python3
"""Mine legacy coverage groups or export the production pre-rerank pool.

Each labelled answer evidence unit is matched independently against candidates
returned by dense + sparse + MiniRAG + fusion + the current reranker.  Candidates
covering any required unit are positives; high-ranked zero-coverage candidates
are hard negatives.  Positive relevance weights emphasize evidence units that
have few matching candidates, so a multi-hop query cannot be optimized by
repeatedly retrieving only its easiest fact.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_retrieval_recall import (  # noqa: E402
    ArknightsHybridRetriever,
    QueryConfig,
    extract_gold_units,
    load_listwise,
    normalize_text,
    text_hit_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listwise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/arknights_story"))
    parser.add_argument(
        "--embedding-model", type=Path, default=Path("model/embeddings/bge-small-zh-v1.5")
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=Path("model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch"),
    )
    parser.add_argument(
        "--minirag-index", type=Path, default=Path("indexes/arknights_story_minirag_v3/graph.json")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--candidate-source",
        choices=("pre_rerank", "reranked"),
        default="pre_rerank",
        help="Use the full fusion pool for recall training, or the current reranked list for diagnostics.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=120)
    parser.add_argument("--max-group-size", type=int, default=24)
    parser.add_argument("--max-positives-per-unit", type=int, default=2)
    parser.add_argument("--max-chain-chars", type=int, default=2600)
    parser.add_argument(
        "--output-mode",
        choices=("coverage", "raw"),
        default="coverage",
        help=(
            "coverage selects a compact legacy gold-labelled group; raw preserves "
            "legacy labels. Gold-free teacher inputs must be made from an existing "
            "fixed full-candidate export with sanitize_reranker_teacher_inputs.py."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jaccard-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=float, default=0.32)
    parser.add_argument("--min-overlap-grams", type=int, default=60)
    parser.add_argument("--min-candidate-grams", type=int, default=80)
    parser.add_argument("--sample", type=int)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def candidate_representations(item: dict[str, Any], *, rank: int, max_chain_chars: int) -> list[dict[str, Any]]:
    document = item.get("document") or {}
    values = [
        (
            "document",
            str(document.get("search_text") or document.get("clean_text") or "").strip(),
        ),
        ("chain", str(item.get("evidence_chain_text") or "").strip()),
    ]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, text in values:
        if kind == "chain" and len(text) > max_chain_chars:
            continue
        compact = normalize_text(text)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        output.append(
            {
                "text": text,
                "kind": kind,
                "retrieval_rank": rank,
                "doc_id": str(document.get("id") or ""),
                "story_id": str(document.get("story_id") or ""),
                "stage_code": str(document.get("stage_code") or ""),
                "rerank_score": item.get("rerank_score"),
                "fusion_score": item.get("fusion_score"),
            }
        )
    return output


def coverage_for_text(
    text: str,
    gold_units: list[dict[str, str]],
    *,
    args: argparse.Namespace,
) -> list[int]:
    covered: list[int] = []
    for unit_index, unit in enumerate(gold_units):
        if text_hit_score(
            text,
            unit["text"],
            jaccard_threshold=args.jaccard_threshold,
            overlap_threshold=args.overlap_threshold,
            min_overlap_grams=args.min_overlap_grams,
            min_candidate_grams=args.min_candidate_grams,
        ) is not None:
            covered.append(unit_index)
    return covered


def select_group_candidates(
    candidates: list[dict[str, Any]],
    gold_units: list[dict[str, str]],
    *,
    max_group_size: int,
    max_positives_per_unit: int,
) -> list[dict[str, Any]]:
    positives = [item for item in candidates if item["covered_unit_indices"]]
    negatives = [item for item in candidates if not item["covered_unit_indices"]]
    positives.sort(
        key=lambda item: (
            item["retrieval_rank"],
            -len(item["covered_unit_indices"]),
            item["kind"] != "document",
            len(item["text"]),
        )
    )
    negatives.sort(
        key=lambda item: (
            item["retrieval_rank"],
            item["kind"] != "document",
            len(item["text"]),
        )
    )

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for unit_index in range(len(gold_units)):
        added = 0
        for item in positives:
            if unit_index not in item["covered_unit_indices"]:
                continue
            key = normalize_text(item["text"])
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
            added += 1
            if added >= max_positives_per_unit:
                break

    # Keep strong multi-unit positives even if every unit already has a document
    # example. They teach the same model to score evidence chains at inference.
    for item in sorted(
        positives,
        key=lambda value: (-len(value["covered_unit_indices"]), value["retrieval_rank"]),
    ):
        key = normalize_text(item["text"])
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= max(4, max_group_size // 2):
            break

    for item in negatives:
        key = normalize_text(item["text"])
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= max_group_size:
            break

    positive_count = sum(bool(item["covered_unit_indices"]) for item in selected)
    negative_count = len(selected) - positive_count
    if positive_count == 0 or negative_count == 0:
        return []

    unit_frequency = Counter(
        unit_index
        for item in selected
        for unit_index in item["covered_unit_indices"]
    )
    for item in selected:
        item["covered_evidence_ids"] = [
            gold_units[index]["evidence_id"] for index in item["covered_unit_indices"]
        ]
        item["relevance"] = round(
            sum(1.0 / unit_frequency[index] for index in item["covered_unit_indices"]),
            6,
        )
        item.pop("covered_unit_indices", None)
    return selected


def build_query_config(args: argparse.Namespace) -> QueryConfig:
    return QueryConfig(
        dense_top_k=120,
        sparse_top_k=120,
        minirag_top_k=120,
        fusion_top_k=80,
        rerank_top_k=args.candidate_top_k,
        minirag_weight=0.35,
        minirag_mode_weights={
            "fact": 0.5,
            "relation": 1.0,
            "causality": 1.0,
            "reveal": 1.0,
            "reasoning": 0.75,
            "mystery": 0.75,
            "answerability": 0.75,
        },
        minirag_fusion_mode="score",
        reranker_candidate_top_k=120,
        enable_neighbor_expansion=True,
        neighbor_max_seed_docs=16,
        neighbor_story_window=2,
        neighbor_activity_story_sort_window=1,
        rerank_batch_size=args.batch_size,
    )


def main() -> int:
    args = parse_args()
    records = load_listwise(resolve(args.listwise))
    if args.sample:
        records = records[: args.sample]
    index_dir = resolve(args.index_dir)
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=resolve(args.embedding_model),
        reranker_model_path=(
            resolve(args.reranker_model) if args.candidate_source == "reranked" else None
        ),
        documents_path=index_dir / "documents.jsonl",
        faiss_index_path=index_dir / "faiss.index",
        bm25_tokens_path=index_dir / "bm25_tokens.pkl",
        minirag_index_path=resolve(args.minirag_index),
        device=args.device,
    )
    query_config = build_query_config(args)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    groups: list[dict[str, Any]] = []

    for record_index, record in enumerate(records, start=1):
        query = str(record.get("query") or "").strip()
        gold_units = extract_gold_units(record)
        if not query or not gold_units:
            stats["skipped_missing_query_or_gold"] += 1
            continue
        hits = (
            retriever.search(query, config=query_config)
            if args.candidate_source == "reranked"
            else retriever.search_pre_rerank(query, config=query_config)[: args.candidate_top_k]
        )
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, item in enumerate(hits, start=1):
            for candidate in candidate_representations(
                item, rank=rank, max_chain_chars=args.max_chain_chars
            ):
                key = normalize_text(candidate["text"])
                if key in seen:
                    continue
                seen.add(key)
                candidate["covered_unit_indices"] = coverage_for_text(
                    candidate["text"], gold_units, args=args
                )
                candidates.append(candidate)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate["candidate_id"] = f"C{candidate_index:03d}"

        coverable_units = {
            unit_index
            for candidate in candidates
            for unit_index in candidate["covered_unit_indices"]
        }
        stats["queries"] += 1
        stats["gold_units"] += len(gold_units)
        stats["coverable_gold_units"] += len(coverable_units)
        stats["fully_coverable_queries"] += int(len(coverable_units) == len(gold_units))
        if args.output_mode == "raw":
            group_candidates = []
            for candidate in candidates:
                copied = dict(candidate)
                covered_indices = list(copied.pop("covered_unit_indices"))
                copied["covered_evidence_ids"] = [
                    gold_units[index]["evidence_id"] for index in covered_indices
                ]
                copied["relevance"] = float(bool(covered_indices))
                group_candidates.append(copied)
        else:
            group_candidates = select_group_candidates(
                candidates,
                gold_units,
                max_group_size=args.max_group_size,
                max_positives_per_unit=args.max_positives_per_unit,
            )
        if not group_candidates:
            stats["skipped_no_trainable_group"] += 1
            continue
        represented_units = {
            evidence_id
            for candidate in group_candidates
            for evidence_id in candidate.get("covered_evidence_ids", [])
        }
        stats["groups"] += 1
        stats["selected_candidates"] += len(group_candidates)
        stats["represented_gold_units"] += len(represented_units)
        group = {
            "query": query,
            "query_type": str(record.get("query_type") or "unknown"),
            "source_name": str(record.get("source_name") or ""),
            "story_family": str(record.get("story_family") or ""),
            "answer": str(record.get("answer") or ""),
            "candidates": group_candidates,
        }
        group.update(
            {
                "gold_evidence_ids": [unit["evidence_id"] for unit in gold_units],
                "gold_evidence": gold_units,
                "coverable_evidence_ids": [
                    gold_units[index]["evidence_id"]
                    for index in sorted(coverable_units)
                ],
            }
        )
        groups.append(group)
        if record_index % 25 == 0:
            print(
                f"[{record_index}/{len(records)}] groups={len(groups)} "
                f"unit_coverage={stats['coverable_gold_units'] / max(stats['gold_units'], 1):.3f}",
                file=sys.stderr,
                flush=True,
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    summary = {
        "records": len(groups),
        "stats": dict(stats),
        "candidate_top_k": args.candidate_top_k,
        "candidate_source": args.candidate_source,
        "output_mode": args.output_mode,
        "max_group_size": args.max_group_size,
        "max_positives_per_unit": args.max_positives_per_unit,
        "gold_unit_candidate_coverage": round(
            stats["coverable_gold_units"] / max(stats["gold_units"], 1), 6
        ),
        "complete_query_candidate_coverage": round(
            stats["fully_coverable_queries"] / max(stats["queries"], 1), 6
        ),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
