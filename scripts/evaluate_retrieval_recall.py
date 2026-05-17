#!/usr/bin/env python3
"""End-to-end retrieval recall evaluation.

Uses ``reranker_listwise.jsonl`` as gold set: each record's ``positive``
candidate text is treated as the answer-bearing evidence. For each query,
we run the full hybrid retriever (dense + sparse + RRF + reranker) and
check whether any top-k document overlaps strongly enough with the gold
text (character-trigram Jaccard >= threshold) to be considered a hit.

Outputs overall and per-query_type recall@K and MRR.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def _should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_env == "train":
        return True
    executable = Path(sys.executable).as_posix().lower()
    return "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if _should_use_train_overrides():
    if TRAIN_PYTHON_OVERLAY_DIR.exists():
        sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
    if TRAIN_OVERRIDE_DIR.exists():
        sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import (  # noqa: E402  - path injection above
    EMBEDDING_MODEL_DIR,
    INDEX_ROOT,
    QueryConfig,
    RERANKER_MODEL_DIR,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402


CHAIN_TAG_RE = re.compile(r"\[E\d+\]\s*")
WHITESPACE_RE = re.compile(r"\s+")


def parse_top_ks(value: str) -> list[int]:
    return sorted({int(part) for part in value.split(",") if part.strip()})


def parse_mode_weights(value: str) -> dict[str, float]:
    if not value.strip():
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid mode weight {item!r}; expected query_type=multiplier"
            )
        key, raw_weight = item.split("=", 1)
        query_type = key.strip()
        if not query_type:
            raise argparse.ArgumentTypeError(f"Invalid empty query type in {item!r}")
        try:
            weights[query_type] = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid weight for {query_type!r}: {raw_weight!r}"
            ) from exc
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path(
            "data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"
        ),
        help="Path to reranker_listwise.jsonl gold file.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--top-ks", type=parse_top_ks, default="1,5,10,20")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional cap on number of records (useful for smoke test).",
    )
    parser.add_argument("--dense-top-k", type=int, default=120)
    parser.add_argument("--sparse-top-k", type=int, default=120)
    parser.add_argument("--minirag-top-k", type=int, default=120)
    parser.add_argument("--minirag-weight", type=float, default=1.2)
    parser.add_argument(
        "--minirag-fusion-mode",
        choices=("score", "append"),
        default="score",
        help="score: MiniRAG participates in RRF; append: MiniRAG only appends supplemental candidates.",
    )
    parser.add_argument(
        "--reranker-candidate-top-k",
        type=int,
        default=120,
        help="Candidate count after supplemental append before reranking.",
    )
    parser.add_argument(
        "--enable-neighbor-expansion",
        action="store_true",
        help="Append same-story/stage neighboring chunks after fusion and before reranking.",
    )
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=24)
    parser.add_argument("--neighbor-story-window", type=int, default=2)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=1)
    parser.add_argument(
        "--minirag-mode-weights",
        type=parse_mode_weights,
        default={},
        help=(
            "Optional per query-type MiniRAG multipliers, e.g. "
            "relation=1.0,fact=0.6,causality=0.4,reveal=0.4,reasoning=0"
        ),
    )
    parser.add_argument("--fusion-top-k", type=int, default=80)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.25,
        help="Character-trigram Jaccard threshold for a hit.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_ROOT,
        help="Override index directory (use indexes/.baseline_arknights_story for baseline).",
    )
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=RERANKER_MODEL_DIR)
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Evaluate pre-rerank fused candidates as a candidate-pool oracle diagnostic.",
    )
    parser.add_argument(
        "--oracle-sources",
        action="store_true",
        help=(
            "Also evaluate source candidate-pool oracle recall for dense, sparse, "
            "MiniRAG, fused pre-rerank, and source union pools."
        ),
    )
    parser.add_argument("--tag", type=str, default="", help="Free-form tag stored in output JSON.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = CHAIN_TAG_RE.sub("", text)
    cleaned = WHITESPACE_RE.sub("", cleaned)
    return cleaned


def char_trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def trigram_jaccard(left: str, right: str) -> float:
    left_grams = char_trigrams(normalize_text(left))
    right_grams = char_trigrams(normalize_text(right))
    if not left_grams or not right_grams:
        return 0.0
    intersection = left_grams & right_grams
    union = left_grams | right_grams
    return len(intersection) / len(union)


def load_listwise(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_gold_text(record: dict[str, Any]) -> str | None:
    candidates = record.get("candidates") or []
    for candidate in candidates:
        if candidate.get("label") == "positive" or candidate.get("type") == "gold":
            text = candidate.get("text") or ""
            if isinstance(text, str) and text.strip():
                return text
    return None


def evaluate(
    records: list[dict[str, Any]],
    retriever: ArknightsHybridRetriever,
    *,
    query_config: QueryConfig,
    top_ks: list[int],
    jaccard_threshold: float,
    skip_rerank: bool = False,
) -> dict[str, Any]:
    max_k = max(top_ks)
    overall_hits: dict[int, int] = {k: 0 for k in top_ks}
    overall_reciprocal_rank_sum = 0.0
    overall_first_hit_rank_sum = 0
    overall_count = 0
    overall_missed = 0
    per_type: dict[str, dict[str, Any]] = {}
    misses: list[dict[str, Any]] = []

    started = time.time()
    for index, record in enumerate(records):
        gold_text = extract_gold_text(record)
        if not gold_text:
            continue
        query = str(record.get("query") or "").strip()
        if not query:
            continue
        query_type = str(record.get("query_type") or "unknown")
        per_type_stats = per_type.setdefault(
            query_type,
            {
                "count": 0,
                "hits": {k: 0 for k in top_ks},
                "rr_sum": 0.0,
                "missed": 0,
            },
        )

        results = (
            retriever.search_pre_rerank(query, config=query_config)
            if skip_rerank
            else retriever.search(query, config=query_config)
        )
        first_hit_rank: int | None = None
        for rank, item in enumerate(results[:max_k], start=1):
            doc_text = str(item["document"].get("clean_text") or "")
            if trigram_jaccard(gold_text, doc_text) >= jaccard_threshold:
                first_hit_rank = rank
                break

        overall_count += 1
        per_type_stats["count"] += 1
        if first_hit_rank is None:
            overall_missed += 1
            per_type_stats["missed"] += 1
            if len(misses) < 20:
                misses.append(
                    {
                        "query": query,
                        "query_type": query_type,
                        "gold_excerpt": normalize_text(gold_text)[:200],
                    }
                )
            continue
        reciprocal_rank = 1.0 / first_hit_rank
        overall_reciprocal_rank_sum += reciprocal_rank
        overall_first_hit_rank_sum += first_hit_rank
        per_type_stats["rr_sum"] += reciprocal_rank
        for k in top_ks:
            if first_hit_rank <= k:
                overall_hits[k] += 1
                per_type_stats["hits"][k] += 1

        if (index + 1) % 50 == 0:
            elapsed = time.time() - started
            sys.stderr.write(
                f"[{index + 1}/{len(records)}] elapsed={elapsed:.1f}s "
                f"running_mrr={overall_reciprocal_rank_sum / max(overall_count, 1):.3f}\n"
            )

    def finalize_bucket(stats: dict[str, Any]) -> dict[str, Any]:
        count = stats["count"]
        if count == 0:
            return {"count": 0}
        return {
            "count": count,
            "missed": stats["missed"],
            "mrr": round(stats["rr_sum"] / count, 4),
            "recall": {
                f"@{k}": round(stats["hits"][k] / count, 4) for k in top_ks
            },
        }

    overall_summary = {
        "count": overall_count,
        "missed": overall_missed,
        "mrr": round(overall_reciprocal_rank_sum / overall_count, 4) if overall_count else 0.0,
        "mean_first_hit_rank": (
            round(overall_first_hit_rank_sum / max(overall_count - overall_missed, 1), 3)
            if overall_count
            else 0.0
        ),
        "recall": {
            f"@{k}": round(overall_hits[k] / overall_count, 4) if overall_count else 0.0
            for k in top_ks
        },
    }

    return {
        "overall": overall_summary,
        "by_query_type": {
            qt: finalize_bucket(stats) for qt, stats in sorted(per_type.items())
        },
        "sample_misses": misses,
        "wall_seconds": round(time.time() - started, 2),
    }


def _first_hit_rank(
    hits: list[dict[str, Any]],
    gold_text: str,
    *,
    jaccard_threshold: float,
    max_k: int,
) -> int | None:
    for rank, item in enumerate(hits[:max_k], start=1):
        doc_text = str(item["document"].get("clean_text") or "")
        if trigram_jaccard(gold_text, doc_text) >= jaccard_threshold:
            return rank
    return None


def evaluate_source_oracle(
    records: list[dict[str, Any]],
    retriever: ArknightsHybridRetriever,
    *,
    query_config: QueryConfig,
    top_ks: list[int],
    jaccard_threshold: float,
) -> dict[str, Any]:
    """Compare whether gold evidence enters each pre-rerank source pool.

    This is a diagnostic upper bound, not the production ranking. If a source has
    high oracle recall but final recall is low, the bottleneck is fusion/reranking.
    If source oracle recall is low, the bottleneck is candidate generation.
    """
    max_k = max(top_ks)
    source_names = ("dense", "sparse", "minirag", "fusion", "source_union")
    buckets: dict[str, dict[str, Any]] = {
        name: {
            "count": 0,
            "missed": 0,
            "rr_sum": 0.0,
            "first_hit_rank_sum": 0,
            "hits": {k: 0 for k in top_ks},
        }
        for name in source_names
    }
    source_misses: dict[str, list[dict[str, Any]]] = {name: [] for name in source_names}
    started = time.time()

    for index, record in enumerate(records):
        gold_text = extract_gold_text(record)
        if not gold_text:
            continue
        query = str(record.get("query") or "").strip()
        if not query:
            continue
        query_type = str(record.get("query_type") or "unknown")

        dense_hits = retriever.dense_search(query, top_k=query_config.dense_top_k)
        sparse_hits = retriever.sparse_search(query, top_k=query_config.sparse_top_k)
        minirag_hits = retriever.minirag_search(query, top_k=query_config.minirag_top_k)
        minirag_weight = retriever.effective_minirag_weight(query, config=query_config)
        if query_config.minirag_fusion_mode == "append":
            primary_hits = retriever.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=[],
                top_k=query_config.fusion_top_k,
                rrf_k=query_config.rrf_k,
                dense_weight=query_config.dense_weight,
                sparse_weight=query_config.sparse_weight,
                minirag_weight=0.0,
            )
            fusion_hits = retriever.append_supplemental_hits(
                primary_hits,
                minirag_hits if minirag_weight > 0 else [],
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
                source_name="minirag",
            )
        else:
            fusion_hits = retriever.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=minirag_hits if minirag_weight > 0 else [],
                top_k=query_config.fusion_top_k,
                rrf_k=query_config.rrf_k,
                dense_weight=query_config.dense_weight,
                sparse_weight=query_config.sparse_weight,
                minirag_weight=minirag_weight,
            )
        if query_config.enable_neighbor_expansion:
            fusion_hits = retriever.expand_hits_with_neighbors(
                fusion_hits,
                max_seed_docs=query_config.neighbor_max_seed_docs,
                story_window=query_config.neighbor_story_window,
                activity_story_sort_window=query_config.neighbor_activity_story_sort_window,
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
            )
        hit_sets = {
            "dense": dense_hits,
            "sparse": sparse_hits,
            "minirag": minirag_hits,
            "fusion": fusion_hits,
        }
        source_ranks: dict[str, int | None] = {}

        for source_name, hits in hit_sets.items():
            stats = buckets[source_name]
            stats["count"] += 1
            first_hit_rank = _first_hit_rank(
                hits,
                gold_text,
                jaccard_threshold=jaccard_threshold,
                max_k=max_k,
            )
            if source_name in {"dense", "sparse", "minirag"}:
                source_ranks[source_name] = first_hit_rank
            if first_hit_rank is None:
                stats["missed"] += 1
                if len(source_misses[source_name]) < 10:
                    source_misses[source_name].append(
                        {
                            "query": query,
                            "query_type": query_type,
                            "gold_excerpt": normalize_text(gold_text)[:200],
                        }
                    )
                continue
            stats["rr_sum"] += 1.0 / first_hit_rank
            stats["first_hit_rank_sum"] += first_hit_rank
            for k in top_ks:
                if first_hit_rank <= k:
                    stats["hits"][k] += 1

        union_rank_candidates = [
            rank for rank in source_ranks.values() if rank is not None
        ]
        union_first_hit_rank = min(union_rank_candidates) if union_rank_candidates else None
        union_stats = buckets["source_union"]
        union_stats["count"] += 1
        if union_first_hit_rank is None:
            union_stats["missed"] += 1
            if len(source_misses["source_union"]) < 10:
                source_misses["source_union"].append(
                    {
                        "query": query,
                        "query_type": query_type,
                        "gold_excerpt": normalize_text(gold_text)[:200],
                    }
                )
        else:
            union_stats["rr_sum"] += 1.0 / union_first_hit_rank
            union_stats["first_hit_rank_sum"] += union_first_hit_rank
            for k in top_ks:
                if union_first_hit_rank <= k:
                    union_stats["hits"][k] += 1

        if (index + 1) % 50 == 0:
            elapsed = time.time() - started
            fusion_stats = buckets["fusion"]
            fusion_mrr = fusion_stats["rr_sum"] / max(fusion_stats["count"], 1)
            sys.stderr.write(
                f"[oracle {index + 1}/{len(records)}] elapsed={elapsed:.1f}s "
                f"fusion_mrr={fusion_mrr:.3f}\n"
            )

    def finalize(stats: dict[str, Any]) -> dict[str, Any]:
        count = stats["count"]
        hit_count = count - stats["missed"]
        return {
            "count": count,
            "missed": stats["missed"],
            "mrr": round(stats["rr_sum"] / count, 4) if count else 0.0,
            "mean_first_hit_rank": (
                round(stats["first_hit_rank_sum"] / max(hit_count, 1), 3)
                if count
                else 0.0
            ),
            "recall": {
                f"@{k}": round(stats["hits"][k] / count, 4) if count else 0.0
                for k in top_ks
            },
        }

    return {
        "sources": {name: finalize(stats) for name, stats in buckets.items()},
        "sample_misses": source_misses,
        "wall_seconds": round(time.time() - started, 2),
    }


def main() -> int:
    args = parse_args()

    listwise_path = (
        args.listwise if args.listwise.is_absolute() else PROJECT_ROOT / args.listwise
    )
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    index_dir = args.index_dir if args.index_dir.is_absolute() else PROJECT_ROOT / args.index_dir
    documents_path = index_dir / "documents.jsonl"
    faiss_index_path = index_dir / "faiss.index"
    bm25_tokens_path = index_dir / "bm25_tokens.pkl"

    records = load_listwise(listwise_path)
    if args.sample is not None:
        records = records[: args.sample]

    sys.stderr.write(
        f"Loading retriever from {index_dir} on device={args.device} "
        f"({len(records)} records)\n"
    )
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=args.reranker_model,
        documents_path=documents_path,
        faiss_index_path=faiss_index_path,
        bm25_tokens_path=bm25_tokens_path,
        minirag_index_path=args.minirag_index,
        device=args.device,
    )

    max_k = max(args.top_ks)
    query_config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        minirag_top_k=args.minirag_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=max_k,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights=args.minirag_mode_weights,
        minirag_fusion_mode=args.minirag_fusion_mode,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        enable_neighbor_expansion=args.enable_neighbor_expansion,
        neighbor_max_seed_docs=args.neighbor_max_seed_docs,
        neighbor_story_window=args.neighbor_story_window,
        neighbor_activity_story_sort_window=args.neighbor_activity_story_sort_window,
        rerank_batch_size=args.rerank_batch_size,
    )

    result = evaluate(
        records,
        retriever,
        query_config=query_config,
        top_ks=args.top_ks,
        jaccard_threshold=args.jaccard_threshold,
        skip_rerank=args.skip_rerank,
    )
    source_oracle = None
    if args.oracle_sources:
        source_oracle = evaluate_source_oracle(
            records,
            retriever,
            query_config=query_config,
            top_ks=args.top_ks,
            jaccard_threshold=args.jaccard_threshold,
        )

    summary = {
        "tag": args.tag,
        "listwise_path": str(listwise_path),
        "index_dir": str(index_dir),
        "device": args.device,
        "reranker_model": str(args.reranker_model),
        "embedding_model": str(args.embedding_model),
        "minirag_index": str(args.minirag_index) if args.minirag_index else "",
        "skip_rerank": bool(args.skip_rerank),
        "oracle_sources": bool(args.oracle_sources),
        "query_config": {
            "dense_top_k": query_config.dense_top_k,
            "sparse_top_k": query_config.sparse_top_k,
            "minirag_top_k": query_config.minirag_top_k,
            "fusion_top_k": query_config.fusion_top_k,
            "rerank_top_k": query_config.rerank_top_k,
            "minirag_weight": query_config.minirag_weight,
            "minirag_mode_weights": query_config.minirag_mode_weights,
            "minirag_fusion_mode": query_config.minirag_fusion_mode,
            "reranker_candidate_top_k": query_config.reranker_candidate_top_k,
            "enable_neighbor_expansion": query_config.enable_neighbor_expansion,
            "neighbor_max_seed_docs": query_config.neighbor_max_seed_docs,
            "neighbor_story_window": query_config.neighbor_story_window,
            "neighbor_activity_story_sort_window": query_config.neighbor_activity_story_sort_window,
            "rerank_batch_size": query_config.rerank_batch_size,
        },
        "jaccard_threshold": args.jaccard_threshold,
        "top_ks": args.top_ks,
        **result,
    }
    if source_oracle is not None:
        summary["source_oracle"] = source_oracle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("by_query_type:")
    print(json.dumps(summary["by_query_type"], ensure_ascii=False, indent=2))
    if source_oracle is not None:
        print("source_oracle:")
        print(json.dumps(source_oracle["sources"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
