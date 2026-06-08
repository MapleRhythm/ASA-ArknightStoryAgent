from __future__ import annotations

from typing import Any, Callable

from asa_arknight_story_agent.inference.pipeline.constants import (
    MINIRAG_CHAPTER_EXPANSION_TASK_TYPE,
    WEB_CONTEXT_TASK_TYPE,
)
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval.merge import merge_evidence_keep_order
from asa_arknight_story_agent.inference.web_context.evidence import build_web_context_evidence


def retrieve_round_evidence(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    pending_queries: list[str],
    round_index: int,
    state: PipelineRunState,
    progress_callback: Callable[[str], None] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    minirag_expansion_record: dict[str, Any] | None = None
    if (
        round_index == 1
        and pipeline.query_config.minirag_chapter_isolation
        and pipeline.query_config.minirag_auto_second_retrieval
    ):
        _, _, state.evidence, minirag_expansion_record = (
            pipeline._retrieve_first_round_with_scoped_minirag_expansion(
                question,
                hypothesis,
                pending_queries,
            )
        )
        if minirag_expansion_record is not None and progress_callback:
            progress_callback(MINIRAG_CHAPTER_EXPANSION_TASK_TYPE)
        if minirag_expansion_record is not None:
            state.retained_chapter_scope = str(minirag_expansion_record.get("chapter_scope") or "").strip() or None
            state.retained_storyline_scope = (
                str(minirag_expansion_record.get("storyline_scope") or "").strip() or None
            )
            state.scope_retention_enabled = bool(
                minirag_expansion_record.get("use_scoped_candidates")
                and state.retained_chapter_scope
            )
    else:
        _, _, state.evidence = pipeline._retrieve_round(
            question,
            hypothesis,
            pending_queries,
            minirag_chapter_scope=None,
            candidate_chapter_scope=None,
            sparse_storyline_scope=None,
        )

    if state.scope_retention_enabled and state.retained_scope_evidence and round_index > 1:
        state.evidence = merge_evidence_keep_order(
            state.retained_scope_evidence,
            state.evidence,
            limit=max(pipeline.query_config.reranker_candidate_top_k, pipeline.prompt_evidence_top_k * 2),
        )

    web_context_record: dict[str, Any] | None = None
    if round_index == 1 and pipeline.web_context_config.enabled:
        if progress_callback:
            progress_callback(WEB_CONTEXT_TASK_TYPE)
        state.web_context_evidence, web_context_record = build_web_context_evidence(
            question,
            state.evidence,
            pipeline.web_context_config,
            retriever=pipeline.retriever,
            hypothesis=hypothesis,
        )
    if state.web_context_evidence:
        state.evidence = [*state.web_context_evidence, *state.evidence]
    state.evidence = pipeline.merge_raw_definition_evidence(question, hypothesis, state.evidence)
    if round_index == 1 and state.scope_retention_enabled:
        state.retained_scope_evidence = list(state.evidence)
    return minirag_expansion_record, web_context_record
