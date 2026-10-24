#!/usr/bin/env python3
"""Cache source retrieval once, then sweep RRF weights, depths, pools, and quotas.

This diagnostic deliberately stops before reranking.  It uses the same segment
hit test and source-quota semantics as ``evaluate_retrieval_recall.py`` while
avoiding repeated embedding/BM25 inference for every fusion configuration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from evaluate_retrieval_recall import (
    candidate_hit_segment,
    extract_gold_text,
    load_listwise,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listwise", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--segment-coverage-threshold", type=float, default=0.9)
    return parser.parse_args()


def quota_pairs(pool: int) -> list[tuple[int, int]]:
    if pool == 120:
        return [
            (0, 0),
            (20, 100),
            (20, 80),
            (40, 80),
            (40, 60),
            (60, 60),
            (60, 40),
            (80, 40),
            (80, 20),
            (100, 20),
        ]
    if pool == 160:
        return [
            (0, 0),
            (20, 140),
            (20, 120),
            (40, 120),
            (40, 100),
            (60, 100),
            (60, 80),
            (80, 80),
            (80, 60),
            (100, 60),
            (100, 40),
            (120, 40),
            (140, 20),
        ]
    raise ValueError(f"Unsupported pool: {pool}")


def fuse(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    *,
    pool: int,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
    dense_cutoff: int,
    sparse_cutoff: int,
    documents: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for rank, (doc_id, score) in enumerate(dense):
        item = fused.setdefault(
            doc_id,
            {
                "doc_index": doc_id,
                "document": documents[doc_id],
                "dense_score": None,
                "sparse_score": None,
                "minirag_score": None,
                "fusion_score": 0.0,
            },
        )
        item["dense_score"] = score
        item["fusion_score"] += dense_weight / (rrf_k + rank + 1)
    for rank, (doc_id, score) in enumerate(sparse):
        item = fused.setdefault(
            doc_id,
            {
                "doc_index": doc_id,
                "document": documents[doc_id],
                "dense_score": None,
                "sparse_score": None,
                "minirag_score": None,
                "fusion_score": 0.0,
            },
        )
        item["sparse_score"] = score
        item["fusion_score"] += sparse_weight / (rrf_k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)
    return ArknightsHybridRetriever.apply_source_quotas(
        ranked,
        dense_hits=[{"doc_index": doc_id} for doc_id, _ in dense],
        sparse_hits=[{"doc_index": doc_id} for doc_id, _ in sparse],
        top_k=pool,
        dense_min_quota=dense_cutoff,
        sparse_min_quota=sparse_cutoff,
    )


def summarize(first_ranks: list[int | None], top_ks: tuple[int, ...]) -> dict[str, Any]:
    count = len(first_ranks)
    found = [rank for rank in first_ranks if rank is not None]
    return {
        "count": count,
        "missed": count - len(found),
        "mrr": round(sum(1.0 / rank for rank in found) / max(count, 1), 6),
        "mean_first_hit_rank": round(sum(found) / max(len(found), 1), 4),
        "recall": {
            f"@{k}": round(sum(rank is not None and rank <= k for rank in first_ranks) / max(count, 1), 6)
            for k in top_ks
        },
    }


def main() -> int:
    args = parse_args()
    records = load_listwise(args.listwise)
    index_dir = args.index_dir
    load_started = time.time()
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=None,
        documents_path=index_dir / "documents.jsonl",
        faiss_index_path=index_dir / "faiss.index",
        bm25_tokens_path=index_dir / "bm25_tokens.pkl",
        sparse_index_path=index_dir / "sparse_index.pkl",
        index_metadata_path=index_dir / "index_meta.json",
        minirag_index_path=None,
        device=args.device,
    )
    load_seconds = time.time() - load_started

    cached: list[dict[str, Any]] = []
    retrieval_started = time.time()
    for row_index, record in enumerate(records, start=1):
        query = str(record.get("query") or "").strip()
        gold = extract_gold_text(record)
        if not query or not gold:
            continue
        dense_hits = retriever.dense_search(query, top_k=args.source_top_k)
        sparse_hits = retriever.sparse_search(query, top_k=args.source_top_k)
        relevant: set[int] = set()
        for hit in dense_hits + sparse_hits:
            doc_id = int(hit["doc_index"])
            if doc_id in relevant:
                continue
            if candidate_hit_segment(
                hit,
                gold,
                segment_coverage_threshold=args.segment_coverage_threshold,
            ) is not None:
                relevant.add(doc_id)
        cached.append(
            {
                "query": query,
                "query_type": str(record.get("query_type") or "unknown"),
                "dense": [(int(hit["doc_index"]), float(hit["score"])) for hit in dense_hits],
                "sparse": [(int(hit["doc_index"]), float(hit["score"])) for hit in sparse_hits],
                "relevant": relevant,
                "gold": gold,
            }
        )
        if row_index % 50 == 0:
            print(f"[cache {row_index}/{len(records)}] elapsed={time.time() - retrieval_started:.1f}s", flush=True)
    retrieval_seconds = time.time() - retrieval_started

    top_ks = (1, 5, 10, 20, 50, 120, 160)
    weights = (0.35, 0.5, 0.75, 1.0, 1.25, 1.5)
    configurations: list[dict[str, Any]] = []
    documents = {index: document for index, document in enumerate(retriever.documents)}
    for source_depth, pool in ((120, 120), (160, 120), (160, 160), (200, 120), (200, 160)):
        for dense_weight in weights:
            for dense_cutoff, sparse_cutoff in quota_pairs(pool):
                first_ranks: list[int | None] = []
                for row in cached:
                    ranking = fuse(
                        row["dense"][:source_depth],
                        row["sparse"][:source_depth],
                        pool=pool,
                        rrf_k=args.rrf_k,
                        dense_weight=dense_weight,
                        sparse_weight=1.0,
                        dense_cutoff=dense_cutoff,
                        sparse_cutoff=sparse_cutoff,
                        documents=documents,
                    )
                    first_ranks.append(
                        next(
                            (
                                rank
                                for rank, item in enumerate(ranking, start=1)
                                if candidate_hit_segment(
                                    item,
                                        row["gold"],
                                    segment_coverage_threshold=args.segment_coverage_threshold,
                                ) is not None
                            ),
                            None,
                        )
                    )
                configurations.append(
                    {
                        "source_depth": source_depth,
                        "pool": pool,
                        "dense_weight": dense_weight,
                        "sparse_weight": 1.0,
                        "dense_cutoff": dense_cutoff,
                        "sparse_cutoff": sparse_cutoff,
                        "metrics": summarize(first_ranks, top_ks),
                    }
                )

    configurations.sort(
        key=lambda item: (
            item["metrics"]["recall"]["@120"],
            item["metrics"]["recall"]["@50"],
            item["metrics"]["mrr"],
        ),
        reverse=True,
    )
    payload = {
        "listwise": str(args.listwise),
        "index_dir": str(args.index_dir),
        "embedding_model": str(args.embedding_model),
        "source_top_k": args.source_top_k,
        "rrf_k": args.rrf_k,
        "segment_coverage_threshold": args.segment_coverage_threshold,
        "records": len(cached),
        "load_seconds": round(load_seconds, 3),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configurations[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
