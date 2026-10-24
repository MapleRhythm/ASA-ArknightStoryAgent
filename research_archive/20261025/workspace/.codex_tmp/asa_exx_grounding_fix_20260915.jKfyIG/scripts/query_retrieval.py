#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.config import EMBEDDING_MODEL_DIR, QueryConfig, RERANKER_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the hybrid story retriever.")
    parser.add_argument("query", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--sparse-top-k", type=int, default=40)
    parser.add_argument("--fusion-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=RERANKER_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=args.reranker_model,
        device=args.device,
    )
    results = retriever.search(
        args.query,
        config=QueryConfig(
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            fusion_top_k=args.fusion_top_k,
            rerank_top_k=args.rerank_top_k,
        ),
    )
    simplified = []
    for item in results:
        document = item["document"]
        simplified.append(
            {
                "id": document["id"],
                "activity_name": document.get("activity_name"),
                "story_name": document.get("story_name"),
                "stage_code": document.get("stage_code"),
                "avg_tag": document.get("avg_tag"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "evidence_chain_model_score": item.get("evidence_chain_model_score"),
                "evidence_chain_roles": item.get("evidence_chain_roles"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "text": document["clean_text"],
                "source_path": document["source_path"],
            }
        )
    print(json.dumps(simplified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
