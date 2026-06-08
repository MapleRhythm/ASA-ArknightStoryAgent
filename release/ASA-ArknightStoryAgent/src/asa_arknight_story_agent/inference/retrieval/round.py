from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval.finalization import (
    build_final_rerank_query,
    fuse_retrieval_hits,
    rerank_and_rescue_hits,
)
from asa_arknight_story_agent.inference.retrieval.merge import merge_ranked_hits
from asa_arknight_story_agent.inference.retrieval.scope import filter_hits_by_chapter_scope
from asa_arknight_story_agent.inference.retrieval.neighbors import expand_fused_hits_with_neighbors
from asa_arknight_story_agent.inference.retrieval.search import search_queries, search_scoped_chapter_queries
from asa_arknight_story_agent.inference.common.text_utils import expand_queries_with_main_chapter_terms


def finalize_hits(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    minirag_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved_question, rerank_query, safe_related_terms = build_final_rerank_query(question, hypothesis)
    fused_hits = fuse_retrieval_hits(
        pipeline.retriever,
        rerank_query=rerank_query,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=minirag_hits,
        query_config=pipeline.query_config,
    )
    if pipeline.query_config.enable_neighbor_expansion:
        fused_hits = expand_fused_hits_with_neighbors(pipeline=pipeline, fused_hits=fused_hits)

    return rerank_and_rescue_hits(
        pipeline.retriever,
        question=question,
        resolved_question=resolved_question,
        rerank_query=rerank_query,
        hypothesis=hypothesis,
        fused_hits=fused_hits,
        safe_related_terms=safe_related_terms,
        query_config=pipeline.query_config,
    )


def retrieve_round(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
    minirag_chapter_scope: str | None = None,
    candidate_chapter_scope: str | None = None,
    sparse_storyline_scope: str | None = None,
    enable_minirag: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_queries = expand_queries_with_main_chapter_terms(queries)
    dense_hits, sparse_hits, minirag_hits = search_queries(
        pipeline=pipeline,
        queries=expanded_queries,
        minirag_chapter_scope=minirag_chapter_scope,
        sparse_storyline_scope=sparse_storyline_scope,
        enable_minirag=enable_minirag,
    )
    if candidate_chapter_scope:
        local_dense_hits, local_sparse_hits = search_scoped_chapter_queries(
            pipeline=pipeline,
            queries=expanded_queries,
            chapter_scope=candidate_chapter_scope,
        )
        dense_hits = merge_ranked_hits(
            filter_hits_by_chapter_scope(dense_hits, candidate_chapter_scope),
            local_dense_hits,
        )
        sparse_hits = merge_ranked_hits(
            filter_hits_by_chapter_scope(sparse_hits, candidate_chapter_scope),
            local_sparse_hits,
        )
    evidence = finalize_hits(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=minirag_hits,
    )
    return dense_hits, sparse_hits, evidence
