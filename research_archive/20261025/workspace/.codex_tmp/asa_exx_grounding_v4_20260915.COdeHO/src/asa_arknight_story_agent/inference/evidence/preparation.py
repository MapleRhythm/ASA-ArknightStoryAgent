from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.evidence.crag_refinement import refine_evidence_strips
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_preparation import (
    merge_raw_definition_evidence,
    prepare_prompt_evidence,
)


class EvidencePreparationMixin:
    def merge_raw_definition_evidence(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return merge_raw_definition_evidence(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
        )

    def prepare_prompt_evidence(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return prepare_prompt_evidence(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
        )

    def refine_evidence_strips(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return refine_evidence_strips(
            pipeline=self,
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
        )
