#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"


def should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_env == "train":
        return True
    executable = Path(sys.executable).as_posix().lower()
    return "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if should_use_train_overrides() and TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from goldenglow.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)
from goldenglow.retrieval.hybrid import (  # noqa: E402
    ArknightsHybridRetriever,
    load_jsonl,
    tokenize_for_bm25,
)
from goldenglow.retrieval.reranker import CrossEncoderReranker  # noqa: E402


def now() -> float:
    return time.perf_counter()


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values) * 1000, 3),
        "p50_ms": round(statistics.median(ordered) * 1000, 3),
        "p95_ms": round(percentile(ordered, 0.95) * 1000, 3),
        "min_ms": round(ordered[0] * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
    }


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval pipeline stage latencies.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=[],
        help="Single query to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Text file with one query per line.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--sparse-top-k", type=int, default=40)
    parser.add_argument("--fusion-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=RERANKER_MODEL_DIR)
    parser.add_argument("--documents-path", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--faiss-index-path", type=Path, default=FAISS_INDEX_PATH)
    parser.add_argument("--bm25-tokens-path", type=Path, default=BM25_TOKENS_PATH)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--show-top-docs", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [query.strip() for query in args.queries if query.strip()]
    if args.query_file:
        queries.extend(
            line.strip()
            for line in args.query_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not queries:
        raise SystemExit("Please provide at least one query via --query or --query-file.")
    return queries


def load_retriever_with_timings(args: argparse.Namespace) -> tuple[ArknightsHybridRetriever, dict[str, float]]:
    timings: dict[str, float] = {}

    start = now()
    documents = load_jsonl(args.documents_path)
    timings["load_documents_s"] = now() - start

    start = now()
    index = faiss.read_index(str(args.faiss_index_path))
    timings["load_faiss_s"] = now() - start

    start = now()
    with args.bm25_tokens_path.open("rb") as handle:
        tokenized_corpus = pickle.load(handle)
    bm25 = BM25Okapi(tokenized_corpus)
    timings["load_bm25_s"] = now() - start

    sync_device(args.device)
    start = now()
    embedding_model = SentenceTransformer(str(args.embedding_model), device=args.device)
    sync_device(args.device)
    timings["load_embedding_model_s"] = now() - start

    reranker = None
    if not args.disable_reranker and args.reranker_model.exists():
        sync_device(args.device)
        start = now()
        reranker = CrossEncoderReranker(args.reranker_model, device=args.device)
        sync_device(args.device)
        timings["load_reranker_model_s"] = now() - start
    else:
        timings["load_reranker_model_s"] = 0.0

    retriever = ArknightsHybridRetriever(
        documents=documents,
        index=index,
        bm25=bm25,
        embedding_model=embedding_model,
        reranker=reranker,
    )
    timings["load_total_s"] = sum(timings.values())
    return retriever, timings


def benchmark_query(
    retriever: ArknightsHybridRetriever,
    query: str,
    *,
    config: QueryConfig,
    device: str,
    show_top_docs: int,
) -> dict[str, Any]:
    stage_times: dict[str, float] = {}

    sync_device(device)
    total_started = now()

    sync_device(device)
    started = now()
    vector = retriever.embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sync_device(device)
    stage_times["dense_encode_s"] = now() - started

    started = now()
    scores, indices = retriever.index.search(vector.astype(np.float32), config.dense_top_k)
    dense_hits: list[dict[str, Any]] = []
    for score, doc_index in zip(scores[0].tolist(), indices[0].tolist()):
        if doc_index < 0:
            continue
        dense_hits.append(
            {
                "doc_index": doc_index,
                "score": float(score),
                "document": retriever.documents[doc_index],
            }
        )
    stage_times["faiss_search_s"] = now() - started
    stage_times["dense_total_s"] = stage_times["dense_encode_s"] + stage_times["faiss_search_s"]

    started = now()
    tokens = tokenize_for_bm25(query)
    bm25_scores = retriever.bm25.get_scores(tokens)
    if config.sparse_top_k >= len(bm25_scores):
        top_indices = np.argsort(bm25_scores)[::-1]
    else:
        top_indices = np.argpartition(bm25_scores, -config.sparse_top_k)[-config.sparse_top_k:]
        top_indices = top_indices[np.argsort(bm25_scores[top_indices])[::-1]]
    sparse_hits: list[dict[str, Any]] = []
    for doc_index in top_indices.tolist():
        score = float(bm25_scores[doc_index])
        if score <= 0:
            continue
        sparse_hits.append(
            {
                "doc_index": int(doc_index),
                "score": score,
                "document": retriever.documents[int(doc_index)],
            }
        )
    stage_times["sparse_total_s"] = now() - started

    started = now()
    fused_hits = retriever.reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        top_k=config.fusion_top_k,
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
    )
    stage_times["fusion_s"] = now() - started

    reranked_hits = fused_hits
    if retriever.reranker:
        sync_device(device)
        started = now()
        rerank_scores = retriever.reranker.score(
            query=query,
            documents=[item["document"]["search_text"] for item in fused_hits],
            batch_size=config.rerank_batch_size,
        )
        sync_device(device)
        for item, rerank_score in zip(fused_hits, rerank_scores):
            item["rerank_score"] = rerank_score
        reranked_hits = sorted(
            fused_hits,
            key=lambda item: item.get("rerank_score", float("-inf")),
            reverse=True,
        )[: config.rerank_top_k]
        stage_times["rerank_s"] = now() - started
    else:
        reranked_hits = fused_hits[: config.rerank_top_k]
        stage_times["rerank_s"] = 0.0

    sync_device(device)
    stage_times["end_to_end_s"] = now() - total_started

    result: dict[str, Any] = {
        "query": query,
        "stage_times_s": stage_times,
        "result_count": len(reranked_hits),
    }
    if show_top_docs > 0:
        result["top_docs"] = [
            {
                "id": item["document"].get("id"),
                "activity_name": item["document"].get("activity_name"),
                "story_name": item["document"].get("story_name"),
                "stage_code": item["document"].get("stage_code"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
            }
            for item in reranked_hits[:show_top_docs]
        ]
    return result


def main() -> None:
    args = parse_args()
    queries = load_queries(args)
    config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=args.rerank_top_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rerank_batch_size=args.rerank_batch_size,
    )

    retriever, load_timings = load_retriever_with_timings(args)

    for _ in range(args.warmup):
        for query in queries:
            benchmark_query(
                retriever,
                query,
                config=config,
                device=args.device,
                show_top_docs=0,
            )

    runs: list[dict[str, Any]] = []
    stage_samples: dict[str, list[float]] = {}
    for _ in range(args.repeat):
        for query in queries:
            run = benchmark_query(
                retriever,
                query,
                config=config,
                device=args.device,
                show_top_docs=args.show_top_docs,
            )
            runs.append(run)
            for stage_name, value in run["stage_times_s"].items():
                stage_samples.setdefault(stage_name, []).append(value)

    payload = {
        "environment": {
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "query_count": len(queries),
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "config": {
            "dense_top_k": config.dense_top_k,
            "sparse_top_k": config.sparse_top_k,
            "fusion_top_k": config.fusion_top_k,
            "rerank_top_k": config.rerank_top_k,
            "rrf_k": config.rrf_k,
            "dense_weight": config.dense_weight,
            "sparse_weight": config.sparse_weight,
            "rerank_batch_size": config.rerank_batch_size,
            "reranker_enabled": retriever.reranker is not None,
        },
        "load_timings_ms": {
            key: round(value * 1000, 3) for key, value in load_timings.items()
        },
        "stage_latency_summary": {
            stage_name: summarize(values) for stage_name, values in sorted(stage_samples.items())
        },
        "runs": runs,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
