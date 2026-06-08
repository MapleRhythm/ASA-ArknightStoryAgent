from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asa_arknight_story_agent.inference.retrieval.scope import (
    infer_dominant_minirag_chapter_scope,
    infer_dominant_storyline_scope,
)


@dataclass
class MiniRAGScopeContext:
    scope_info: dict[str, Any] | None
    storyline_scope_info: dict[str, Any] | None
    storyline_scope: str
    storyline_scope_ratio: float
    storyline_sparse_scope_enabled: bool

    @property
    def sparse_storyline_scope(self) -> str | None:
        return self.storyline_scope if self.storyline_sparse_scope_enabled else None


def build_minirag_scope_context(
    *,
    pipeline: Any,
    first_pass_evidence: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
) -> MiniRAGScopeContext:
    scope_info = infer_dominant_minirag_chapter_scope(
        first_pass_evidence,
        dense_hits,
        sparse_hits,
        max_items=max(1, int(pipeline.query_config.minirag_scope_seed_top_k)),
    )
    storyline_scope_info = infer_dominant_storyline_scope(
        first_pass_evidence,
        dense_hits,
        sparse_hits,
        max_items=max(1, int(pipeline.query_config.storyline_scope_seed_top_k)),
    )
    storyline_scope = ""
    storyline_scope_ratio = 0.0
    storyline_sparse_scope_enabled = False
    if storyline_scope_info is not None:
        storyline_scope = str(storyline_scope_info["scope"])
        storyline_scope_ratio = float(storyline_scope_info.get("dominance_ratio") or 0.0)
        storyline_sparse_scope_enabled = (
            pipeline.query_config.enable_storyline_sparse_scope
            and storyline_scope_ratio >= float(pipeline.query_config.storyline_sparse_scope_min_ratio)
        )
    return MiniRAGScopeContext(
        scope_info=scope_info,
        storyline_scope_info=storyline_scope_info,
        storyline_scope=storyline_scope,
        storyline_scope_ratio=storyline_scope_ratio,
        storyline_sparse_scope_enabled=storyline_sparse_scope_enabled,
    )


def minirag_scope_values(pipeline: Any, scope_info: dict[str, Any]) -> tuple[str, float, bool, bool, str | None]:
    chapter_scope = str(scope_info["scope"])
    scope_ratio = float(scope_info.get("dominance_ratio") or 0.0)
    graph_scope_enabled = scope_ratio >= float(pipeline.query_config.minirag_graph_scope_min_ratio)
    second_pass_scope_enabled = scope_ratio >= float(pipeline.query_config.minirag_second_pass_scope_min_ratio)
    graph_scope = chapter_scope if graph_scope_enabled else None
    return chapter_scope, scope_ratio, graph_scope_enabled, second_pass_scope_enabled, graph_scope
