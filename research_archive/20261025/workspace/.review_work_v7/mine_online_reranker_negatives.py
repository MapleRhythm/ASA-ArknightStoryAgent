#!/usr/bin/env python3
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
    candidate_hit_source,
    extract_gold_text,
    load_listwise,
)


def pair_from_candidate(
    record: dict[str, Any], item: dict[str, Any], *, rank: int, candidate_kind: str
) -> dict[str, Any]:
    document = item.get("document") or {}
    chain_text = str(item.get("evidence_chain_text") or "").strip()
    doc_text = str(document.get("search_text") or document.get("clean_text") or "").strip()
    negative = chain_text if candidate_kind == "chain" else doc_text
    return {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "source_name": str(record.get("source_name") or ""),
        "positive": extract_gold_text(record) or "",
        "negative": negative,
        "positive_score": 1.0,
        "negative_score": max(0.45, 0.83 - 0.03 * (rank - 1)),
        "negative_type": f"online_high_rank_{candidate_kind}",
        "answer": str(record.get("answer") or ""),
        "answer_evidence": list(record.get("answer_evidence") or []),
        "answer_focus": str(record.get("answer_focus") or ""),
        "positive_chain": list(record.get("answer_evidence") or []),
        "negative_chain": [],
        "online_rank": rank,
        "online_doc_id": str(document.get("id") or ""),
        "online_story_id": str(document.get("story_id") or ""),
        "online_rerank_score": item.get("rerank_score"),
        "online_evidence_chain_model_score": item.get("evidence_chain_model_score"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine real high-ranked ASA retriever mistakes.")
    parser.add_argument("--listwise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/arknights_story"))
    parser.add_argument("--embedding-model", type=Path, default=Path("model/embeddings/bge-small-zh-v1.5"))
    parser.add_argument("--reranker-model", type=Path, default=Path("model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch"))
    parser.add_argument("--minirag-index", type=Path, default=Path("indexes/arknights_story_minirag_v3/graph.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rank", type=int, default=12)
    parser.add_argument("--max-negatives-per-query", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    listwise = load_listwise(resolve(args.listwise))
    index_dir = resolve(args.index_dir)
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=resolve(args.embedding_model),
        reranker_model_path=resolve(args.reranker_model),
        documents_path=index_dir / "documents.jsonl",
        faiss_index_path=index_dir / "faiss.index",
        bm25_tokens_path=index_dir / "bm25_tokens.pkl",
        minirag_index_path=resolve(args.minirag_index),
        device=args.device,
    )
    config = QueryConfig(
        dense_top_k=120,
        sparse_top_k=120,
        minirag_top_k=120,
        fusion_top_k=80,
        rerank_top_k=max(args.max_rank, 32),
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
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, record in enumerate(listwise, start=1):
        query = str(record.get("query") or "").strip()
        gold = extract_gold_text(record)
        if not query or not gold:
            continue
        hits = retriever.search(query, config=config)
        added = 0
        seen: set[str] = set()
        for rank, item in enumerate(hits[: args.max_rank], start=1):
            if candidate_hit_source(
                item,
                gold,
                jaccard_threshold=0.25,
                overlap_threshold=0.32,
                min_overlap_grams=60,
                min_candidate_grams=80,
            ) is not None:
                continue
            for kind, text in (
                ("chain", str(item.get("evidence_chain_text") or "").strip()),
                ("doc", str((item.get("document") or {}).get("search_text") or "").strip()),
            ):
                compact = "".join(text.split())
                if not compact or compact in seen:
                    continue
                seen.add(compact)
                output.append(pair_from_candidate(record, item, rank=rank, candidate_kind=kind))
                stats[f"added_{kind}"] += 1
                added += 1
                break
            if added >= args.max_negatives_per_query:
                break
        stats["queries"] += 1
        stats["queries_with_negatives"] += int(added > 0)
        if index % 50 == 0:
            print(f"[{index}/{len(listwise)}] mined={len(output)}", file=sys.stderr)
    args.output = resolve(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"records": len(output), "stats": dict(stats)}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
