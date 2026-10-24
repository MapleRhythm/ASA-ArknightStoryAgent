from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import answer_from_grounded_facts
from asa_arknight_story_agent.inference.grounding.quote_match_utils import (
    normalize_for_evidence_match,
    quote_occurs_in_evidence,
)
from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map,
    evidence_id_text_map_from_prompt,
)


UNIQUE_REBIND_MIN_QUOTE_CHARS = 12
UNIQUE_REBIND_MAX_QUOTE_CHARS = 80


def rebind_unique_visible_quotes(
    *,
    conclusion: ConclusionResult,
    evidence: list[dict[str, Any]],
    evidence_prompt_text: str | None,
) -> tuple[ConclusionResult, list[str]]:
    """Repair only unambiguous E-ID slips for copied visible assertions.

    The quote text is never changed. Questions, very short spans, over-budget
    spans, and quotes occurring in zero or multiple visible E blocks remain
    untouched and therefore fail through the existing validator.
    """
    eid_map = (
        evidence_id_text_map(evidence)
        if evidence_prompt_text is None
        else evidence_id_text_map_from_prompt(evidence_prompt_text)
    )
    normalized_evidence = {
        evidence_id: normalize_for_evidence_match(text)
        for evidence_id, text in eid_map.items()
    }
    supported_facts = deepcopy(conclusion.supported_facts)
    inferred_facts = deepcopy(conclusion.inferred_facts)
    repairs: list[str] = []
    for group_name, facts in (
        ("supported_fact", supported_facts),
        ("inferred_fact", inferred_facts),
    ):
        for fact_index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict):
                continue
            refs = fact.get("evidence_refs")
            if not isinstance(refs, list):
                continue
            for ref_index, ref in enumerate(refs, start=1):
                if not isinstance(ref, dict):
                    continue
                quote = str(ref.get("quote") or "").strip()
                if (
                    len(quote) < UNIQUE_REBIND_MIN_QUOTE_CHARS
                    or len(quote) > UNIQUE_REBIND_MAX_QUOTE_CHARS
                    or quote.rstrip().endswith(("?", "？"))
                ):
                    continue
                normalized_quote = normalize_for_evidence_match(quote)
                if not normalized_quote:
                    continue
                current_id = str(ref.get("evidence_id") or "").strip()
                # Never repair an invented or missing ID. A rebind is only a
                # correction between two labels the model actually saw.
                if current_id not in normalized_evidence:
                    continue
                if (
                    quote_occurs_in_evidence(quote, normalized_evidence[current_id])
                ):
                    continue
                matches = [
                    evidence_id
                    for evidence_id, text in normalized_evidence.items()
                    if quote_occurs_in_evidence(quote, text)
                ]
                if len(matches) != 1:
                    continue
                resolved_id = matches[0]
                ref["evidence_id"] = resolved_id
                repairs.append(
                    f"{group_name}_{fact_index}_ref_{ref_index}_"
                    f"unique_evidence_id_rebound:{current_id or '<missing>'}->{resolved_id}"
                )
    if not repairs:
        return conclusion, []
    return (
        replace(
            conclusion,
            supported_facts=supported_facts,
            inferred_facts=inferred_facts,
            generation_diagnostics={
                **conclusion.generation_diagnostics,
                "grounding_repairs": repairs,
            },
        ),
        repairs,
    )


def validate_quote_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
    evidence_prompt_text: str | None,
) -> ConclusionResult | None:
    conclusion, grounding_repairs = rebind_unique_visible_quotes(
        conclusion=conclusion,
        evidence=evidence,
        evidence_prompt_text=evidence_prompt_text,
    )
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
                grounding_warnings=dedupe_keep_order([*quote_issues, *grounding_repairs]),
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
            grounding_warnings=dedupe_keep_order([*quote_issues, *grounding_repairs]),
            generation_diagnostics={
                **conclusion.generation_diagnostics,
                "grounding_status": "rejected_to_retrieve_more",
                "grounding_issues": quote_issues,
            },
        )

    if not quote_warnings and not conclusion.inferred_facts:
        if grounding_repairs:
            return replace(
                conclusion,
                grounding_warnings=dedupe_keep_order(
                    [*conclusion.grounding_warnings, *grounding_repairs]
                ),
            )
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
            grounding_warnings=dedupe_keep_order([*quote_warnings, *grounding_repairs]),
            generation_diagnostics=conclusion.generation_diagnostics,
        )
    combined_warnings = dedupe_keep_order(
        [*conclusion.grounding_warnings, *quote_warnings, *grounding_repairs]
    )
    if combined_warnings != conclusion.grounding_warnings:
        return replace(conclusion, grounding_warnings=combined_warnings)
    return None
