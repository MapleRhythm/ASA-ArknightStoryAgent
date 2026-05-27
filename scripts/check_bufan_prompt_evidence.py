#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldenglow.config import EMBEDDING_MODEL_DIR, MINIRAG_GRAPH_PATH, RERANKER_MODEL_DIR, QueryConfig
from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    build_hypothesis,
    build_retrieval_query,
    render_evidence_blocks,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prompt evidence anchors for the hard '不反' action-target case.")
    parser.add_argument("--runtime-config", type=Path, default=PROJECT_ROOT / "api-mode" / "runtime_api.json")
    parser.add_argument("--question", type=str, default="真龙为什么要启动不反？")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--with-reranker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    retrieval_cfg = runtime_config.get("retrieval", {})
    inference_cfg = runtime_config.get("inference", {})

    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}
    query_config = QueryConfig(
        dense_top_k=int(retrieval_cfg.get("dense_top_k", 120)),
        sparse_top_k=int(retrieval_cfg.get("sparse_top_k", 120)),
        minirag_top_k=int(retrieval_cfg.get("minirag_top_k", 120)),
        fusion_top_k=int(retrieval_cfg.get("fusion_top_k", 80)),
        rerank_top_k=int(retrieval_cfg.get("rerank_top_k", 32)),
        minirag_weight=float(retrieval_cfg.get("minirag_weight", 0.35)),
        minirag_mode_weights={str(key): float(value) for key, value in minirag_mode_weights.items()},
        minirag_fusion_mode=str(retrieval_cfg.get("minirag_fusion_mode", "score")),
        reranker_candidate_top_k=int(retrieval_cfg.get("reranker_candidate_top_k", 120)),
        enable_neighbor_expansion=bool(retrieval_cfg.get("enable_neighbor_expansion", False)),
        neighbor_max_seed_docs=int(retrieval_cfg.get("neighbor_max_seed_docs", 24)),
        neighbor_story_window=int(retrieval_cfg.get("neighbor_story_window", 2)),
        neighbor_activity_story_sort_window=int(retrieval_cfg.get("neighbor_activity_story_sort_window", 1)),
        rerank_batch_size=int(retrieval_cfg.get("rerank_batch_size", 4)),
    )
    minirag_index_path = None
    if retrieval_cfg.get("enable_minirag", False):
        configured_minirag = Path(str(retrieval_cfg.get("minirag_index_path") or MINIRAG_GRAPH_PATH))
        minirag_index_path = configured_minirag if configured_minirag.is_absolute() else PROJECT_ROOT / configured_minirag
    reranker_model = RERANKER_MODEL_DIR if args.with_reranker else None

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=reranker_model,
        minirag_index_path=minirag_index_path,
        device=args.device,
    )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=type("DummyGenerator", (), {"max_tokens": 1})(),
        query_config=query_config,
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(inference_cfg.get("prompt_evidence_max_chars_per_doc", 520)),
        prompt_conclusion_evidence_max_total_chars=int(
            inference_cfg.get("prompt_conclusion_evidence_max_total_chars", 5000)
        ),
        enable_mmr=bool(inference_cfg.get("enable_mmr", True)),
        mmr_lambda=float(inference_cfg.get("mmr_lambda", 0.72)),
        enable_pyramid_order=bool(inference_cfg.get("enable_pyramid_order", True)),
        enable_crag_refinement=False,
    )
    hypothesis = build_hypothesis(args.question, "")
    dense_hits, sparse_hits, minirag_hits = pipeline._search_queries(
        [args.question, build_retrieval_query(hypothesis)]
    )
    evidence = pipeline._finalize_hits(args.question, hypothesis, dense_hits, sparse_hits, minirag_hits)
    prompt_evidence = pipeline.prepare_prompt_evidence(args.question, hypothesis, evidence)
    rendered = render_evidence_blocks(
        prompt_evidence,
        max_chars_per_doc=int(inference_cfg.get("prompt_evidence_max_chars_per_doc", 520)),
        max_total_chars=int(inference_cfg.get("prompt_conclusion_evidence_max_total_chars", 5000)),
    )

    checks = {
        "has_bufan": "不反" in rendered,
        "has_suiling_crisis": "岁陵" in rendered and "危机" in rendered,
        "has_life_cost": "性命" in rendered and "代价" in rendered,
    }
    top = []
    for item in prompt_evidence[:5]:
        doc = item.get("document") or {}
        text = str(item.get("evidence_chain_text") or doc.get("clean_text") or "")
        top.append(
            {
                "id": doc.get("id"),
                "stage_code": doc.get("stage_code"),
                "anchors": [
                    anchor
                    for anchor in ("岁陵", "危机", "性命", "代价", "不反", "太尉", "莫佚")
                    if anchor in text
                ],
            }
        )
    result = {
        "question": args.question,
        "checks": checks,
        "top_prompt_evidence": top,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
