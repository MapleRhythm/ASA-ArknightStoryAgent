from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.retrieval.minirag_expansion import MiniRAGExpansionMixin
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval.neighbors import expand_fused_hits_with_neighbors
from asa_arknight_story_agent.inference.retrieval.round import finalize_hits, retrieve_round
from asa_arknight_story_agent.inference.retrieval.search import (
    search_minirag_queries,
    search_queries,
    search_scoped_chapter_queries,
)


class RetrievalPipelineMixin(MiniRAGExpansionMixin):
    def _search_queries(
        self,
        queries: list[str],
        *,
        minirag_chapter_scope: str | None = None,
        sparse_storyline_scope: str | None = None,
        enable_minirag: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return search_queries(
            pipeline=self,
            queries=queries,
            minirag_chapter_scope=minirag_chapter_scope,
            sparse_storyline_scope=sparse_storyline_scope,
            enable_minirag=enable_minirag,
        )

    def _search_minirag_queries(
        self,
        queries: list[str],
        *,
        minirag_chapter_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        return search_minirag_queries(
            pipeline=self,
            queries=queries,
            minirag_chapter_scope=minirag_chapter_scope,
        )

    def _search_scoped_chapter_queries(
        self,
        queries: list[str],
        *,
        chapter_scope: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return search_scoped_chapter_queries(
            pipeline=self,
            queries=queries,
            chapter_scope=chapter_scope,
        )

    def _finalize_hits(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        minirag_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return finalize_hits(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=minirag_hits,
        )

    def _expand_fused_hits_with_neighbors(self, fused_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return expand_fused_hits_with_neighbors(pipeline=self, fused_hits=fused_hits)

    def _retrieve_round(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        queries: list[str],
        *,
        minirag_chapter_scope: str | None = None,
        candidate_chapter_scope: str | None = None,
        sparse_storyline_scope: str | None = None,
        enable_minirag: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return retrieve_round(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            queries=queries,
            minirag_chapter_scope=minirag_chapter_scope,
            candidate_chapter_scope=candidate_chapter_scope,
            sparse_storyline_scope=sparse_storyline_scope,
            enable_minirag=enable_minirag,
        )
