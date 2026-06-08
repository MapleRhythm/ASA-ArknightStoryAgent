from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asa_arknight_story_agent.inference.retrieval.minirag_query_planning import build_minirag_expansion_queries
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval.merge import merge_ranked_hits
from asa_arknight_story_agent.inference.retrieval.scope import filter_hits_by_chapter_scope
from asa_arknight_story_agent.inference.common.text_utils import expand_queries_with_main_chapter_terms


@dataclass
class MiniRAGSecondPassHits:
    second_pass_queries: list[str]
    graph_hits: list[dict[str, Any]]
    second_dense_hits: list[dict[str, Any]]
    second_sparse_hits: list[dict[str, Any]]
    second_minirag_hits: list[dict[str, Any]]
    local_dense_hits: list[dict[str, Any]]
    local_sparse_hits: list[dict[str, Any]]
    scoped_dense_hits: list[dict[str, Any]]
    scoped_sparse_hits: list[dict[str, Any]]
    combined_minirag_hits: list[dict[str, Any]]
    scoped_candidate_count: int
    use_scoped_candidates: bool
    global_dense_hits: list[dict[str, Any]]
    global_sparse_hits: list[dict[str, Any]]
    combined_dense_hits: list[dict[str, Any]]
    combined_sparse_hits: list[dict[str, Any]]


def run_minirag_second_pass(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    expanded_queries: list[str],
    graph_hits: list[dict[str, Any]],
    chapter_scope: str,
    graph_scope: str | None,
    graph_scope_enabled: bool,
    second_pass_scope_enabled: bool,
    sparse_storyline_scope: str | None,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    scope_info: dict[str, Any],
) -> MiniRAGSecondPassHits:
    second_pass_queries = build_minirag_expansion_queries(
        question,
        hypothesis,
        graph_hits,
        chapter_scope_label=str(scope_info.get("label") or chapter_scope) if graph_scope_enabled else "global",
        top_k=max(1, int(pipeline.query_config.minirag_expansion_query_top_k)),
    )
    expanded_second_pass_queries = expand_queries_with_main_chapter_terms(second_pass_queries)
    second_dense_hits, second_sparse_hits, second_minirag_hits = pipeline._search_queries(
        expanded_second_pass_queries,
        minirag_chapter_scope=graph_scope,
        sparse_storyline_scope=sparse_storyline_scope,
        enable_minirag=True,
    )
    local_dense_hits, local_sparse_hits = pipeline._search_scoped_chapter_queries(
        [*expanded_queries, *expanded_second_pass_queries],
        chapter_scope=chapter_scope,
    )
    scoped_dense_hits = merge_ranked_hits(
        filter_hits_by_chapter_scope(dense_hits, chapter_scope),
        filter_hits_by_chapter_scope(second_dense_hits, chapter_scope),
        local_dense_hits,
    )
    scoped_sparse_hits = merge_ranked_hits(
        filter_hits_by_chapter_scope(sparse_hits, chapter_scope),
        filter_hits_by_chapter_scope(second_sparse_hits, chapter_scope),
        local_sparse_hits,
    )
    combined_minirag_hits = merge_ranked_hits(graph_hits, second_minirag_hits)
    scoped_candidate_count = len(scoped_dense_hits) + len(scoped_sparse_hits) + len(combined_minirag_hits)
    use_scoped_candidates = (
        second_pass_scope_enabled and scoped_candidate_count >= max(8, pipeline.query_config.rerank_top_k)
    )
    global_dense_hits = merge_ranked_hits(dense_hits, second_dense_hits)
    global_sparse_hits = merge_ranked_hits(sparse_hits, second_sparse_hits)
    if use_scoped_candidates:
        combined_dense_hits = merge_ranked_hits(scoped_dense_hits, global_dense_hits)
        combined_sparse_hits = merge_ranked_hits(scoped_sparse_hits, global_sparse_hits)
    else:
        combined_dense_hits = global_dense_hits
        combined_sparse_hits = global_sparse_hits
    return MiniRAGSecondPassHits(
        second_pass_queries=second_pass_queries,
        graph_hits=graph_hits,
        second_dense_hits=second_dense_hits,
        second_sparse_hits=second_sparse_hits,
        second_minirag_hits=second_minirag_hits,
        local_dense_hits=local_dense_hits,
        local_sparse_hits=local_sparse_hits,
        scoped_dense_hits=scoped_dense_hits,
        scoped_sparse_hits=scoped_sparse_hits,
        combined_minirag_hits=combined_minirag_hits,
        scoped_candidate_count=scoped_candidate_count,
        use_scoped_candidates=use_scoped_candidates,
        global_dense_hits=global_dense_hits,
        global_sparse_hits=global_sparse_hits,
        combined_dense_hits=combined_dense_hits,
        combined_sparse_hits=combined_sparse_hits,
    )
