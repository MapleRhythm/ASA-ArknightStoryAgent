from __future__ import annotations

from typing import Any, Callable

from asa_arknight_story_agent.inference.pipeline.constants import (
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    INITIAL_HYPOTHESIS_TASK_TYPE,
)
from asa_arknight_story_agent.inference.pipeline.actions import apply_terminal_conclusion
from asa_arknight_story_agent.inference.pipeline.query_flow import build_initial_queries, prepare_next_round
from asa_arknight_story_agent.inference.pipeline.retrieval_round import retrieve_round_evidence
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.trace import append_conclusion_to_step, build_step_record
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, InferenceResult
from asa_arknight_story_agent.inference.pipeline.result_rendering import build_inference_result


class PipelineOrchestrationMixin:
    def _build_initial_queries(self, question: str, hypothesis: HypothesisDocument) -> list[str]:
        return build_initial_queries(question, hypothesis)

    def _retrieve_round_evidence(
        self,
        *,
        question: str,
        hypothesis: HypothesisDocument,
        pending_queries: list[str],
        round_index: int,
        state: PipelineRunState,
        progress_callback: Callable[[str], None] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return retrieve_round_evidence(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            pending_queries=pending_queries,
            round_index=round_index,
            state=state,
            progress_callback=progress_callback,
        )

    def _build_step_record(
        self,
        *,
        round_index: int,
        pending_queries: list[str],
        hypothesis_task_type: str,
        hypothesis: HypothesisDocument,
        state: PipelineRunState,
        minirag_expansion_record: dict[str, Any] | None,
        web_context_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return build_step_record(
            round_index=round_index,
            pending_queries=pending_queries,
            hypothesis_task_type=hypothesis_task_type,
            hypothesis=hypothesis,
            state=state,
            minirag_expansion_record=minirag_expansion_record,
            web_context_record=web_context_record,
        )

    def run(
        self,
        question: str,
        dialogue_context: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> InferenceResult:
        if progress_callback:
            progress_callback(INITIAL_HYPOTHESIS_TASK_TYPE)
        current_hypothesis = self.build_hypothesis(question, dialogue_context)
        state = PipelineRunState()
        pending_queries = self._build_initial_queries(question, current_hypothesis)
        current_hypothesis_task_type = INITIAL_HYPOTHESIS_TASK_TYPE

        for round_index in range(1, self.max_retrieval_rounds + 1):
            if progress_callback:
                progress_callback("retrieval")
            minirag_expansion_record, web_context_record = self._retrieve_round_evidence(
                question=question,
                hypothesis=current_hypothesis,
                pending_queries=pending_queries,
                round_index=round_index,
                state=state,
                progress_callback=progress_callback,
            )
            step_record = self._build_step_record(
                round_index=round_index,
                pending_queries=pending_queries,
                hypothesis_task_type=current_hypothesis_task_type,
                hypothesis=current_hypothesis,
                state=state,
                minirag_expansion_record=minirag_expansion_record,
                web_context_record=web_context_record,
            )
            state.retrieval_trace.append(step_record)

            if progress_callback:
                progress_callback(CONCLUSION_TASK_TYPE)
            conclusion = self.generate_conclusion(
                question,
                current_hypothesis,
                state.evidence,
                state.retrieval_trace,
                round_index,
            )
            append_conclusion_to_step(step_record, conclusion)

            if apply_terminal_conclusion(
                state=state,
                conclusion=conclusion,
                round_index=round_index,
                max_retrieval_rounds=self.max_retrieval_rounds,
            ):
                break

            current_hypothesis, pending_queries = prepare_next_round(
                pipeline=self,
                question=question,
                current_hypothesis=current_hypothesis,
                conclusion=conclusion,
                state=state,
                step_record=step_record,
                round_index=round_index,
                progress_callback=progress_callback,
            )
            current_hypothesis_task_type = FOLLOW_UP_HYPOTHESIS_TASK_TYPE

        return build_inference_result(
            pipeline=self,
            question=question,
            current_hypothesis=current_hypothesis,
            state=state,
        )
