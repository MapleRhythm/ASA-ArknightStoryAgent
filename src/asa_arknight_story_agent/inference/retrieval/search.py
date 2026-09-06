from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from asa_arknight_story_agent.inference.retrieval.merge import merge_ranked_hits


def _search_one_query(
    *,
    pipeline: Any,
    query: str,
    minirag_chapter_scope: str | None,
    sparse_storyline_scope: str | None,
    enable_minirag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all retrieval lanes for one query.

    Keeping the per-query operation in one function makes the parallel path
    deterministic: results are collected in input order and merged with the
    same RRF implementation as the legacy sequential path.
    """
    dense_hits = pipeline.retriever.dense_search(
        query,
        top_k=pipeline.query_config.dense_top_k,
    )
    sparse_hits = pipeline.retriever.sparse_search(
        query,
        top_k=pipeline.query_config.sparse_top_k,
        storyline_scope=sparse_storyline_scope,
    )
    minirag_hits: list[dict[str, Any]] = []
    minirag_search = getattr(pipeline.retriever, "minirag_search", None)
    if enable_minirag and minirag_search is not None:
        minirag_hits = minirag_search(
            query,
            top_k=pipeline.query_config.minirag_top_k,
            chapter_scope=minirag_chapter_scope,
        ) or []
    return dense_hits or [], sparse_hits or [], minirag_hits


def _bounded_workers(pipeline: Any, query_count: int) -> int:
    configured = int(getattr(pipeline.query_config, "retrieval_workers", 1) or 1)
    return max(1, min(configured, query_count))


def search_queries(
    *,
    pipeline: Any,
    queries: list[str],
    minirag_chapter_scope: str | None = None,
    sparse_storyline_scope: str | None = None,
    enable_minirag: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not queries:
        return [], [], []
    workers = _bounded_workers(pipeline, len(queries))
    if workers == 1:
        per_query = [
            _search_one_query(
                pipeline=pipeline,
                query=query,
                minirag_chapter_scope=minirag_chapter_scope,
                sparse_storyline_scope=sparse_storyline_scope,
                enable_minirag=enable_minirag,
            )
            for query in queries
        ]
    else:
        # executor.map preserves input order, so enabling workers cannot
        # change tie-breaking or the merged ranking relative to the legacy
        # path. Exceptions still propagate to the caller rather than silently
        # dropping a retrieval lane.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asa-retrieval") as executor:
            per_query = list(
                executor.map(
                    lambda query: _search_one_query(
                        pipeline=pipeline,
                        query=query,
                        minirag_chapter_scope=minirag_chapter_scope,
                        sparse_storyline_scope=sparse_storyline_scope,
                        enable_minirag=enable_minirag,
                    ),
                    queries,
                )
            )
    dense_ranked_lists = [result[0] for result in per_query]
    sparse_ranked_lists = [result[1] for result in per_query]
    minirag_ranked_lists = [result[2] for result in per_query if result[2]]
    return (
        merge_ranked_hits(*dense_ranked_lists),
        merge_ranked_hits(*sparse_ranked_lists),
        merge_ranked_hits(*minirag_ranked_lists),
    )


def search_minirag_queries(
    *,
    pipeline: Any,
    queries: list[str],
    minirag_chapter_scope: str | None = None,
) -> list[dict[str, Any]]:
    minirag_search = getattr(pipeline.retriever, "minirag_search", None)
    if minirag_search is None:
        return []
    if not queries:
        return []
    workers = _bounded_workers(pipeline, len(queries))
    if workers == 1:
        per_query = [
            minirag_search(
                query,
                top_k=pipeline.query_config.minirag_top_k,
                chapter_scope=minirag_chapter_scope,
            )
            for query in queries
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asa-minirag") as executor:
            per_query = list(
                executor.map(
                    lambda query: minirag_search(
                        query,
                        top_k=pipeline.query_config.minirag_top_k,
                        chapter_scope=minirag_chapter_scope,
                    ),
                    queries,
                )
            )
    ranked_lists = [hits for hits in per_query if hits]
    return merge_ranked_hits(*ranked_lists)


def search_scoped_chapter_queries(
    *,
    pipeline: Any,
    queries: list[str],
    chapter_scope: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pipeline.query_config.enable_scoped_chapter_search or not chapter_scope or not queries:
        return [], []
    dense_search = getattr(pipeline.retriever, "dense_search_chapter", None)
    sparse_search = getattr(pipeline.retriever, "sparse_search_chapter", None)
    def search_scoped_one(query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []
        if dense_search is not None and pipeline.query_config.scoped_chapter_dense_top_k > 0:
            dense_hits = dense_search(
                query,
                top_k=pipeline.query_config.scoped_chapter_dense_top_k,
                chapter_scope=chapter_scope,
            ) or []
        if sparse_search is not None and pipeline.query_config.scoped_chapter_sparse_top_k > 0:
            sparse_hits = sparse_search(
                query,
                top_k=pipeline.query_config.scoped_chapter_sparse_top_k,
                chapter_scope=chapter_scope,
            ) or []
        return dense_hits, sparse_hits

    workers = _bounded_workers(pipeline, len(queries))
    if workers == 1:
        per_query = [search_scoped_one(query) for query in queries]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asa-scoped") as executor:
            per_query = list(executor.map(search_scoped_one, queries))
    dense_ranked_lists = [result[0] for result in per_query if result[0]]
    sparse_ranked_lists = [result[1] for result in per_query if result[1]]
    return merge_ranked_hits(*dense_ranked_lists), merge_ranked_hits(*sparse_ranked_lists)
