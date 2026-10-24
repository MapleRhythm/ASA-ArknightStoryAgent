from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.evidence.rendering import summarize_evidence_for_trace
from asa_arknight_story_agent.inference.retrieval.minirag_scope_context import MiniRAGScopeContext
from asa_arknight_story_agent.inference.retrieval.minirag_second_pass import MiniRAGSecondPassHits


def build_empty_graph_expansion_record(
    *,
    pipeline: Any,
    scope_context: MiniRAGScopeContext,
    chapter_scope: str,
    scope_ratio: float,
    graph_scope_enabled: bool,
    second_pass_scope_enabled: bool,
) -> dict[str, Any]:
    scope_info = scope_context.scope_info or {}
    storyline_scope_info = scope_context.storyline_scope_info
    return {
        "chapter_scope": chapter_scope,
        "chapter_scope_label": scope_info.get("label") or chapter_scope,
        "scope_candidates": scope_info.get("candidates") or [],
        "scope_dominance_ratio": scope_ratio,
        "graph_scope_enabled": graph_scope_enabled,
        "second_pass_scope_enabled": second_pass_scope_enabled,
        "graph_scope_min_ratio": pipeline.query_config.minirag_graph_scope_min_ratio,
        "second_pass_scope_min_ratio": pipeline.query_config.minirag_second_pass_scope_min_ratio,
        "storyline_scope": scope_context.storyline_scope,
        "storyline_scope_label": storyline_scope_info.get("label") if storyline_scope_info else "",
        "storyline_scope_candidates": storyline_scope_info.get("candidates") if storyline_scope_info else [],
        "storyline_scope_dominance_ratio": scope_context.storyline_scope_ratio,
        "storyline_sparse_scope_enabled": scope_context.storyline_sparse_scope_enabled,
        "storyline_sparse_scope_min_ratio": pipeline.query_config.storyline_sparse_scope_min_ratio,
        "graph_hit_count": 0,
        "second_pass_queries": [],
    }


def build_second_pass_expansion_record(
    *,
    pipeline: Any,
    scope_context: MiniRAGScopeContext,
    chapter_scope: str,
    scope_ratio: float,
    graph_scope_enabled: bool,
    second_pass_scope_enabled: bool,
    second_pass: MiniRAGSecondPassHits,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    record = build_empty_graph_expansion_record(
        pipeline=pipeline,
        scope_context=scope_context,
        chapter_scope=chapter_scope,
        scope_ratio=scope_ratio,
        graph_scope_enabled=graph_scope_enabled,
        second_pass_scope_enabled=second_pass_scope_enabled,
    )
    record.update(
        {
            "graph_hit_count": len(second_pass.graph_hits),
            "graph_evidence_summary": summarize_evidence_for_trace(second_pass.graph_hits),
            "second_pass_queries": second_pass.second_pass_queries,
            "scoped_dense_hit_count": len(second_pass.scoped_dense_hits),
            "scoped_sparse_hit_count": len(second_pass.scoped_sparse_hits),
            "scoped_local_dense_hit_count": len(second_pass.local_dense_hits),
            "scoped_local_sparse_hit_count": len(second_pass.local_sparse_hits),
            "scoped_candidate_count": second_pass.scoped_candidate_count,
            "use_scoped_candidates": second_pass.use_scoped_candidates,
            "dual_lane_global_fallback_enabled": second_pass.use_scoped_candidates,
            "global_dense_hit_count": len(second_pass.global_dense_hits),
            "global_sparse_hit_count": len(second_pass.global_sparse_hits),
            "second_pass_evidence_summary": summarize_evidence_for_trace(evidence),
        }
    )
    return record
