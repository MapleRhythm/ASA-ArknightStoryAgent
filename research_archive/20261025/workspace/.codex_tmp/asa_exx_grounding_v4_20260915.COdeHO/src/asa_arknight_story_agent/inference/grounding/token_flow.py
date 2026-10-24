from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.grounding.anchor_checks import validate_required_anchor
from asa_arknight_story_agent.inference.grounding.causal import has_direct_causal_grounding
from asa_arknight_story_agent.inference.grounding.identity import is_identity_question
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.grounding.quote_match_utils import (
    GROUNDING_LONG_TOKEN_MIN_LEN,
    grounding_evidence_pool,
    grounding_extract_answer_tokens,
)
from asa_arknight_story_agent.inference.grounding.token_checks import validate_token_grounding


def validate_answer_token_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    grounding_mode: str,
    max_round_reached: bool,
) -> ConclusionResult | None:
    answer_tokens = grounding_extract_answer_tokens(conclusion.answer, question)
    long_tokens = [token for token in answer_tokens if len(token) >= GROUNDING_LONG_TOKEN_MIN_LEN]
    if not long_tokens:
        return None

    evidence_pool = grounding_evidence_pool(evidence)
    if not evidence_pool:
        return None

    anchor_result = validate_required_anchor(
        question=question,
        hypothesis=hypothesis,
        evidence_pool=evidence_pool,
        max_round_reached=max_round_reached,
    )
    if anchor_result is not None:
        return anchor_result

    if grounding_mode == "weak" and not is_identity_question(question, hypothesis):
        return None

    if has_direct_causal_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence_pool=evidence_pool,
    ):
        return None

    return validate_token_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
        conclusion=conclusion,
        long_tokens=long_tokens,
        evidence_pool=evidence_pool,
        max_round_reached=max_round_reached,
    )
