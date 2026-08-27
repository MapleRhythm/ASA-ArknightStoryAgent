from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.grounding.fallback import has_answerable_evidence
from asa_arknight_story_agent.inference.grounding.modes import (
    grounding_disabled,
    normalize_grounding_mode,
)
from asa_arknight_story_agent.inference.grounding.retrieval_fallbacks import (
    answer_from_retrieval_fallbacks,
    corrected_special_answer,
)
from asa_arknight_story_agent.inference.grounding.token_flow import validate_answer_token_grounding
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.grounding.quote_checks import validate_quote_grounding
from asa_arknight_story_agent.inference.grounding.evidence_id_checks import validate_evidence_id_answer


def validate_conclusion_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
    mode: str = "weak",
    evidence_prompt_text: str | None = None,
) -> ConclusionResult:
    if grounding_disabled(mode):
        return conclusion
    grounding_mode = normalize_grounding_mode(mode)
    # The legacy deterministic fallbacks do not carry structured evidence
    # references.  Letting them bypass evidence-id validation would recreate a
    # hallucination path, so they remain available only for legacy modes.
    if grounding_mode != "evidence_id":
        fallback_result = answer_from_retrieval_fallbacks(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            max_round_reached=max_round_reached,
        )
        if fallback_result is not None:
            return fallback_result

    if conclusion.next_action != "answer_directly" or not conclusion.answer:
        return conclusion

    if grounding_mode != "evidence_id":
        special_answer = corrected_special_answer(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
        )
        if special_answer is not None:
            return special_answer

    if grounding_mode == "evidence_id":
        evidence_id_result = validate_evidence_id_answer(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            max_round_reached=max_round_reached,
            evidence_prompt_text=evidence_prompt_text,
        )
        if evidence_id_result is not None:
            return evidence_id_result
        # A None result means every structured claim passed evidence-id
        # validation.  Do not run the legacy whole-answer lexical validator:
        # it is a second, incompatible grounding policy and can turn a valid
        # compact E-ID answer into retrieve_more/abstain.
        return conclusion

    if grounding_mode in {"quote", "grounded", "strict"}:
        quote_result = validate_quote_grounding(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
            conclusion=conclusion,
            max_round_reached=max_round_reached,
            evidence_prompt_text=evidence_prompt_text,
        )
        if quote_result is not None:
            return quote_result

    token_result = validate_answer_token_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
        conclusion=conclusion,
        grounding_mode=grounding_mode,
        max_round_reached=max_round_reached,
    )
    return token_result or conclusion
