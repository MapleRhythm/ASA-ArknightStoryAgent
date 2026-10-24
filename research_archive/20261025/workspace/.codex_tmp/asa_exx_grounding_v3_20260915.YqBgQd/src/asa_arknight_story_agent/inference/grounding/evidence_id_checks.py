from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map, evidence_id_text_map_from_prompt,
)
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import (
    validate_evidence_id_grounding,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.payload.utils import answer_from_structured_facts
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import (
    build_heuristic_follow_up_hypothesis,
)


def validate_evidence_id_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
    evidence_prompt_text: str | None = None,
) -> ConclusionResult | None:
    issues, warnings = validate_evidence_id_grounding(
        conclusion=conclusion,
        evidence=evidence,
        question=question,
        evidence_prompt_text=evidence_prompt_text,
    )
    if not issues:
        # Never trust a separately generated final_answer in evidence-id mode.
        # Rebuild user-visible text only from the claims that just passed
        # citation validation, closing the "valid refs + invented summary" gap.
        grounded_answer = answer_from_structured_facts(conclusion.supported_facts, [])
        if grounded_answer and grounded_answer != conclusion.answer:
            return ConclusionResult(
                next_action=conclusion.next_action,
                answer=grounded_answer,
                missing_slots=conclusion.missing_slots,
                clarification_question=conclusion.clarification_question,
                follow_up_hypothesis=conclusion.follow_up_hypothesis,
                supported_facts=conclusion.supported_facts,
                inferred_facts=[],
                grounding_warnings=warnings,
                generation_diagnostics=dict(conclusion.generation_diagnostics),
            )
        conclusion.grounding_warnings.extend(warnings)
        return None
    missing_slots = dedupe_keep_order(
        [*(conclusion.missing_slots or []), "answer_directly 缺少可校验 evidence_id 支撑", *issues[:4]]
    )
    # Keep the candidate and only individually validated facts for local repair
    # and explicit partial results. These are not a separate semantic verdict.
    mapping = (
        evidence_id_text_map(evidence) if evidence_prompt_text is None
        else evidence_id_text_map_from_prompt(evidence_prompt_text)
    )
    partial = []
    for fact in conclusion.supported_facts:
        if not isinstance(fact, dict):
            continue
        single = replace(conclusion, supported_facts=[fact], inferred_facts=[])
        fact_issues, _ = validate_evidence_id_grounding(
            conclusion=single, evidence=evidence, question=question,
            evidence_prompt_text=evidence_prompt_text,
        )
        if not fact_issues:
            partial.append({
                "fact": deepcopy(fact),
                "evidence_texts": {
                    ref["evidence_id"]: mapping[ref["evidence_id"]]
                    for ref in fact["evidence_refs"]
                },
            })
    diagnostics = {
        **conclusion.generation_diagnostics,
        "grounding_candidate": deepcopy(conclusion.supported_facts),
        "grounding_partial_support": partial,
        "grounding_issues": issues,
        "grounding_failure_kind": (
            "lexical_support" if all("_terms_outside_cited_evidence:" in issue for issue in issues)
            else "structural_or_mixed"
        ),
    }
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            # The failed claims were not grounded.  Do not turn raw retrieved
            # snippets into a pseudo-answer here: that would bypass the very
            # claim -> evidence-id validation this mode is meant to enforce.
            answer="现有检索证据不足以给出可校验的回答。",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=None,
            grounding_warnings=[*issues, *warnings],
            generation_diagnostics={
                **diagnostics,
                "grounding_status": "rejected_to_abstain",
                "grounding_issues": issues,
            },
        )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=missing_slots,
        clarification_question="",
        follow_up_hypothesis=build_heuristic_follow_up_hypothesis(question, hypothesis, missing_slots),
        grounding_warnings=[*issues, *warnings],
        generation_diagnostics={
            **diagnostics,
            "grounding_status": "rejected_to_retrieve_more",
            "grounding_issues": issues,
        },
    )
