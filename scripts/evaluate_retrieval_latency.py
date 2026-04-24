#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import EMBEDDING_MODEL_DIR, QueryConfig, RERANKER_MODEL_DIR  # noqa: E402
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever, load_jsonl  # noqa: E402


DOCUMENTS_PATH = PROJECT_ROOT / "indexes" / "arknights_story" / "documents.jsonl"


def build_retrieval_seed_query(document: dict, *, max_chars: int) -> str:
    parts: list[str] = []
    for key in ("activity_name", "story_name", "stage_code", "avg_tag"):
        value = document.get(key)
        if value:
            parts.append(str(value))

    speakers: list[str] = []
    for segment in document.get("segments") or []:
        speaker = segment.get("speaker") if isinstance(segment, dict) else None
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) >= 4:
            break
    if speakers:
        parts.append(" ".join(speakers))

    clean_text = str(document.get("clean_text") or "")
    if clean_text:
        parts.append(clean_text[:max_chars])

    return "\n".join(part for part in parts if part).strip()


def elapsed_since(started: float) -> float:
    return time.perf_counter() - started


def timed_retrieval(
    retriever: ArknightsHybridRetriever,
    query: str,
    config: QueryConfig,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    timings: dict[str, float | int] = {}

    started = time.perf_counter()
    dense_hits = retriever.dense_search(query, top_k=config.dense_top_k)
    timings["dense_seconds"] = elapsed_since(started)
    timings["dense_hits"] = len(dense_hits)

    started = time.perf_counter()
    sparse_hits = retriever.sparse_search(query, top_k=config.sparse_top_k)
    timings["sparse_seconds"] = elapsed_since(started)
    timings["sparse_hits"] = len(sparse_hits)

    started = time.perf_counter()
    fused_hits = retriever.reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        top_k=config.fusion_top_k,
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
    )
    timings["fusion_seconds"] = elapsed_since(started)
    timings["fusion_hits"] = len(fused_hits)

    if not retriever.reranker:
        results = fused_hits[: config.rerank_top_k]
        timings["rerank_seconds"] = 0.0
        timings["rerank_hits"] = len(results)
        return results, timings

    started = time.perf_counter()
    rerank_scores = retriever.reranker.score(
        query=query,
        documents=[item["document"]["search_text"] for item in fused_hits],
        batch_size=config.rerank_batch_size,
    )
    timings["rerank_seconds"] = elapsed_since(started)
    timings["rerank_hits"] = len(rerank_scores)

    for item, rerank_score in zip(fused_hits, rerank_scores):
        item["rerank_score"] = rerank_score
    results = sorted(
        fused_hits,
        key=lambda item: item.get("rerank_score", float("-inf")),
        reverse=True,
    )[: config.rerank_top_k]
    return results, timings


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline latency.")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--seed-query-max-chars", type=int, default=260)
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--sparse-top-k", type=int, default=40)
    parser.add_argument("--fusion-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but the current PyTorch build has no CUDA support. "
            "Install a CUDA-enabled torch in .conda or run with --device cpu."
        )

    rng = random.Random(args.seed)
    config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_batch_size=args.rerank_batch_size,
    )

    load_started = time.perf_counter()
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=None if args.no_reranker else RERANKER_MODEL_DIR,
        device=args.device,
    )
    load_seconds = elapsed_since(load_started)
    print(
        json.dumps(
            {
                "load": {
                    "seconds": load_seconds,
                    "included_in_run_timings": False,
                    "device": args.device,
                    "torch_version": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "reranker_enabled": not args.no_reranker,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    documents = load_jsonl(DOCUMENTS_PATH)
    runs: list[dict[str, Any]] = []

    for run_index in range(args.runs):
        seed_doc = rng.choice(documents)
        query_started = time.perf_counter()
        query = build_retrieval_seed_query(
            seed_doc,
            max_chars=args.seed_query_max_chars,
        )
        query_seconds = elapsed_since(query_started)

        total_started = time.perf_counter()
        results, timings = timed_retrieval(retriever, query, config)
        total_seconds = elapsed_since(total_started)

        run_record = {
            "run": run_index + 1,
            "seed_doc_id": seed_doc.get("id"),
            "query_chars": len(query),
            "query_build_seconds": query_seconds,
            "total_retrieval_seconds": total_seconds,
            "top_doc_ids": [item["document"]["id"] for item in results],
            **timings,
        }
        runs.append(run_record)
        print(json.dumps(run_record, ensure_ascii=False), flush=True)

    timing_keys = [
        "query_build_seconds",
        "dense_seconds",
        "sparse_seconds",
        "fusion_seconds",
        "rerank_seconds",
        "total_retrieval_seconds",
    ]
    summary = {
        "runs": args.runs,
        "device": args.device,
        "reranker_enabled": not args.no_reranker,
        "load_seconds": load_seconds,
        "load_included_in_run_timings": False,
        "timings": {
            key: summarize([float(run[key]) for run in runs])
            for key in timing_keys
        },
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
