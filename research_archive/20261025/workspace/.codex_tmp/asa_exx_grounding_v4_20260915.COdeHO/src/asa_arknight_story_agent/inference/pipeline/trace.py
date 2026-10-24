from __future__ import annotations

from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.evidence.rendering import summarize_evidence_for_trace
from asa_arknight_story_agent.inference.pipeline.constants import CONCLUSION_TASK_TYPE
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def build_step_record(
    *,
    round_index: int,
    pending_queries: list[str],
    hypothesis_task_type: str,
    hypothesis: HypothesisDocument,
    state: PipelineRunState,
    minirag_expansion_record: dict[str, Any] | None,
    web_context_record: dict[str, Any] | None,
) -> dict[str, Any]:
    step_record: dict[str, Any] = {
        "round": round_index,
        "queries": list(pending_queries),
        "planner_action": "retrieval_completed",
        "hypothesis_task_type": hypothesis_task_type,
        "hypothesis": asdict(hypothesis),
        "evidence_summary": summarize_evidence_for_trace(state.evidence),
        "retained_chapter_scope": state.retained_chapter_scope or "",
        "retained_storyline_scope": state.retained_storyline_scope or "",
        "scope_retention_enabled": state.scope_retention_enabled,
    }
    if web_context_record is not None:
        step_record["web_context"] = web_context_record
    if minirag_expansion_record is not None:
        step_record["minirag_chapter_expansion"] = minirag_expansion_record
    return step_record


def append_conclusion_to_step(
    step_record: dict[str, Any],
    conclusion: ConclusionResult,
) -> None:
    step_record["conclusion_task_type"] = CONCLUSION_TASK_TYPE
    step_record["conclusion"] = asdict(conclusion)
    step_record["planner_action"] = conclusion.next_action
    step_record["missing_slots"] = conclusion.missing_slots
    step_record["clarification_question"] = conclusion.clarification_question
