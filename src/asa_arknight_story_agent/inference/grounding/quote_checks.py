from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import answer_from_grounded_facts
from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def validate_quote_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
    evidence_prompt_text: str | None,
) -> ConclusionResult | None:
    quote_issues, quote_warnings = validate_grounded_quotes(
        conclusion=conclusion,
        evidence=evidence,
        question=question,
        evidence_prompt_text=evidence_prompt_text,
    )
    if quote_issues:
        missing_slots = dedupe_keep_order(
            [
                *(conclusion.missing_slots or []),
                "answer_directly 缺少可校验 quote 支撑",
                *quote_issues[:4],
            ]
        )
        if max_round_reached:
            return ConclusionResult(
                next_action="abstain",
                # Rejected claims and unvalidated retrieval snippets must not
                # become a pseudo-answer merely because the budget ran out.
                answer="现有检索证据不足以给出可校验的回答。",
                missing_slots=missing_slots,
                clarification_question="",
                follow_up_hypothesis=None,
                grounding_warnings=quote_issues,
                generation_diagnostics={
                    **conclusion.generation_diagnostics,
                    "grounding_status": "rejected_to_abstain",
                    "grounding_issues": quote_issues,
                },
            )
        return ConclusionResult(
            next_action="retrieve_more",
            answer="",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=build_heuristic_follow_up_hypothesis(question, hypothesis, missing_slots),
            grounding_warnings=quote_issues,
            generation_diagnostics={
                **conclusion.generation_diagnostics,
                "grounding_status": "rejected_to_retrieve_more",
                "grounding_issues": quote_issues,
            },
        )

    if not quote_warnings:
        return None

    repaired_answer = answer_from_grounded_facts(conclusion)
    if repaired_answer and repaired_answer != conclusion.answer:
        return ConclusionResult(
            next_action=conclusion.next_action,
            answer=repaired_answer,
            missing_slots=conclusion.missing_slots,
            clarification_question=conclusion.clarification_question,
            follow_up_hypothesis=conclusion.follow_up_hypothesis,
            supported_facts=conclusion.supported_facts,
            inferred_facts=conclusion.inferred_facts,
            grounding_warnings=quote_warnings,
        )
    conclusion.grounding_warnings.extend(quote_warnings)
    return None
