from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.retrieval.merge import merge_ranked_hits


def search_queries(
    *,
    pipeline: Any,
    queries: list[str],
    minirag_chapter_scope: str | None = None,
    sparse_storyline_scope: str | None = None,
    enable_minirag: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    del minirag_chapter_scope, enable_minirag
    dense_ranked_lists: list[list[dict[str, Any]]] = []
    sparse_ranked_lists: list[list[dict[str, Any]]] = []
    for query in queries:
        dense_ranked_lists.append(pipeline.retriever.dense_search(query, top_k=pipeline.query_config.dense_top_k))
        sparse_ranked_lists.append(
            pipeline.retriever.sparse_search(
                query,
                top_k=pipeline.query_config.sparse_top_k,
                storyline_scope=sparse_storyline_scope,
            )
        )
    return (
        merge_ranked_hits(*dense_ranked_lists),
        merge_ranked_hits(*sparse_ranked_lists),
        [],
    )


def search_minirag_queries(
    *,
    pipeline: Any,
    queries: list[str],
    minirag_chapter_scope: str | None = None,
) -> list[dict[str, Any]]:
    del pipeline, queries, minirag_chapter_scope
    return []


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
    dense_ranked_lists: list[list[dict[str, Any]]] = []
    sparse_ranked_lists: list[list[dict[str, Any]]] = []
    for query in queries:
        if dense_search is not None and pipeline.query_config.scoped_chapter_dense_top_k > 0:
            dense_hits = dense_search(
                query,
                top_k=pipeline.query_config.scoped_chapter_dense_top_k,
                chapter_scope=chapter_scope,
            )
            if dense_hits:
                dense_ranked_lists.append(dense_hits)
        if sparse_search is not None and pipeline.query_config.scoped_chapter_sparse_top_k > 0:
            sparse_hits = sparse_search(
                query,
                top_k=pipeline.query_config.scoped_chapter_sparse_top_k,
                chapter_scope=chapter_scope,
            )
            if sparse_hits:
                sparse_ranked_lists.append(sparse_hits)
    return merge_ranked_hits(*dense_ranked_lists), merge_ranked_hits(*sparse_ranked_lists)
