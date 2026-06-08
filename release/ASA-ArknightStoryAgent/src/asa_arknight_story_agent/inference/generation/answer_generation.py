from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.generation.answer_conclusion_generation import generate_conclusion_from_model
from asa_arknight_story_agent.inference.generation.direct_answer_generation import generate_direct_answer_from_model
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


class AnswerGenerationMixin:
    def generate_conclusion(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        current_round: int,
    ) -> ConclusionResult:
        return generate_conclusion_from_model(
            pipeline=self,
            question=question,
            current_hypothesis=current_hypothesis,
            evidence=evidence,
            retrieval_trace=retrieval_trace,
            current_round=current_round,
        )

    def generate_direct_answer(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> ConclusionResult:
        return generate_direct_answer_from_model(
            pipeline=self,
            question=question,
            current_hypothesis=current_hypothesis,
            evidence=evidence,
        )
