from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.retrieval.minirag_scope_expansion import (
    build_empty_graph_expansion_record,
    build_minirag_scope_context,
    build_second_pass_expansion_record,
    minirag_scope_values,
    run_minirag_second_pass,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import expand_queries_with_main_chapter_terms


class MiniRAGExpansionMixin:
    def _retrieve_first_round_with_scoped_minirag_expansion(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        expanded_queries = expand_queries_with_main_chapter_terms(queries)
        dense_hits, sparse_hits, _ = self._search_queries(expanded_queries, enable_minirag=False)
        first_pass_evidence = self._finalize_hits(question, hypothesis, dense_hits, sparse_hits, [])
        scope_context = build_minirag_scope_context(
            pipeline=self,
            first_pass_evidence=first_pass_evidence,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
        )
        if not scope_context.scope_info:
            return dense_hits, sparse_hits, first_pass_evidence, None

        chapter_scope, scope_ratio, graph_scope_enabled, second_pass_scope_enabled, graph_scope = minirag_scope_values(
            self,
            scope_context.scope_info,
        )
        graph_hits = self._search_minirag_queries(
            expanded_queries,
            minirag_chapter_scope=graph_scope,
        )
        if not graph_hits:
            return dense_hits, sparse_hits, first_pass_evidence, build_empty_graph_expansion_record(
                pipeline=self,
                scope_context=scope_context,
                chapter_scope=chapter_scope,
                scope_ratio=scope_ratio,
                graph_scope_enabled=graph_scope_enabled,
                second_pass_scope_enabled=second_pass_scope_enabled,
            )

        second_pass = run_minirag_second_pass(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            expanded_queries=expanded_queries,
            graph_hits=graph_hits,
            chapter_scope=chapter_scope,
            graph_scope=graph_scope,
            graph_scope_enabled=graph_scope_enabled,
            second_pass_scope_enabled=second_pass_scope_enabled,
            sparse_storyline_scope=scope_context.sparse_storyline_scope,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            scope_info=scope_context.scope_info,
        )
        evidence = self._finalize_hits(
            question,
            hypothesis,
            second_pass.combined_dense_hits,
            second_pass.combined_sparse_hits,
            second_pass.combined_minirag_hits,
        )
        expansion_record = build_second_pass_expansion_record(
            pipeline=self,
            scope_context=scope_context,
            chapter_scope=chapter_scope,
            scope_ratio=scope_ratio,
            graph_scope_enabled=graph_scope_enabled,
            second_pass_scope_enabled=second_pass_scope_enabled,
            second_pass=second_pass,
            evidence=evidence,
        )
        return second_pass.combined_dense_hits, second_pass.combined_sparse_hits, evidence, expansion_record
