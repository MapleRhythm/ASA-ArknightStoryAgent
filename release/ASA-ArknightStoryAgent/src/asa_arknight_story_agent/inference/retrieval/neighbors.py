from __future__ import annotations

from typing import Any


def expand_fused_hits_with_neighbors(*, pipeline: Any, fused_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collect_neighbors = getattr(pipeline.retriever, "_collect_story_and_stage_neighbors", None)
    if not fused_hits or collect_neighbors is None:
        return fused_hits

    expanded_by_doc_index: dict[int, dict[str, Any]] = {
        int(item["doc_index"]): item
        for item in fused_hits
    }
    neighbor_doc_indices = collect_neighbors(
        fused_hits,
        max_seed_docs=min(pipeline.query_config.neighbor_max_seed_docs, len(fused_hits)),
        story_window=pipeline.query_config.neighbor_story_window,
        activity_story_sort_window=pipeline.query_config.neighbor_activity_story_sort_window,
        same_story_sweep=pipeline.query_config.enable_same_story_sweep,
        same_story_max_seed_docs=pipeline.query_config.same_story_sweep_max_seed_docs,
        same_story_max_docs_per_story=pipeline.query_config.same_story_sweep_max_docs_per_story,
    )
    max_candidates = max(
        pipeline.query_config.reranker_candidate_top_k,
        pipeline.query_config.fusion_top_k,
        pipeline.query_config.rerank_top_k,
    )
    if pipeline.query_config.enable_same_story_sweep:
        max_candidates += max(0, pipeline.query_config.same_story_sweep_extra_candidates)
    for doc_index in neighbor_doc_indices:
        if doc_index in expanded_by_doc_index:
            continue
        expanded_by_doc_index[doc_index] = {
            "doc_index": doc_index,
            "document": pipeline.retriever.documents[doc_index],
            "dense_score": None,
            "sparse_score": None,
            "fusion_score": 0.0,
            "supplemental_source": "neighbor",
        }
        if len(expanded_by_doc_index) >= max_candidates:
            break
    return list(expanded_by_doc_index.values())
