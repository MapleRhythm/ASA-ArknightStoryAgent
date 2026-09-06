from __future__ import annotations

from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.evidence.rendering import summarize_evidence_for_trace
from asa_arknight_story_agent.inference.pipeline.constants import CONCLUSION_TASK_TYPE
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def conclusion_to_tool_call(conclusion: ConclusionResult) -> dict[str, Any]:
    """Map the legacy conclusion envelope to an internal control-plane call.

    This is intentionally not emitted as user-visible chain-of-thought and
    does not add a model round. It lets the controller, logs and future native
    tool-call adapters share one finite action contract.
    """
    action = conclusion.next_action
    if action == "answer_directly":
        return {
            "protocol": "asa_round_tool_v1",
            "name": "answer",
            "arguments": {
                "supported_facts": conclusion.supported_facts,
                "inferred_facts": conclusion.inferred_facts,
            },
        }
    if action == "retrieve_more":
        follow_up = asdict(conclusion.follow_up_hypothesis) if conclusion.follow_up_hypothesis else None
        return {
            "protocol": "asa_round_tool_v1",
            "name": "search_more",
            "arguments": {
                "missing_requirements": list(conclusion.missing_slots),
                "follow_up_hypothesis": follow_up,
            },
        }
    if action == "clarify_user":
        return {
            "protocol": "asa_round_tool_v1",
            "name": "clarify",
            "arguments": {"question": conclusion.clarification_question},
        }
    return {
        "protocol": "asa_round_tool_v1",
        "name": "abstain",
        "arguments": {"reason": conclusion.answer or "insufficient_evidence"},
    }


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
    step_record["tool_call"] = conclusion_to_tool_call(conclusion)
