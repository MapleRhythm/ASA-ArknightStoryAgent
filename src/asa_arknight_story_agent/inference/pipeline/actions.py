from __future__ import annotations

from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult


def apply_terminal_conclusion(
    *,
    state: PipelineRunState,
    conclusion: ConclusionResult,
    round_index: int,
    max_retrieval_rounds: int,
) -> bool:
    if conclusion.next_action == "answer_directly":
        state.final_answer = conclusion.answer
        return True
    if conclusion.next_action == "clarify_user":
        state.final_answer = conclusion.clarification_question
        return True
    if conclusion.next_action == "abstain":
        state.final_answer = conclusion.answer
        return True
    if round_index >= max_retrieval_rounds:
        state.final_answer = conclusion.answer or "现有检索证据不足以确认。"
        return True
    return False
