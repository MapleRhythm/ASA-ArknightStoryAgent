from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map,
    evidence_id_text_map_from_prompt,
)
from asa_arknight_story_agent.inference.grounding.quote_match_utils import (
    GROUNDING_LONG_TOKEN_MIN_LEN,
    normalize_for_evidence_match,
)
from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import (
    claim_has_unsupported_quote_required_terms,
)
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult


def validate_evidence_id_grounding(
    *,
    conclusion: ConclusionResult,
    evidence: list[dict[str, Any]],
    question: str,
    evidence_prompt_text: str | None = None,
) -> tuple[list[str], list[str]]:
    """Validate compact claim -> evidence-id references without model quotes.

    This deliberately performs conservative lexical support checks. It is a
    safe transition format and not a replacement for a trained entailment
    verifier: semantically unsupported claims can still share all key tokens
    with a cited passage.
    """
    issues: list[str] = []
    warnings: list[str] = []
    # Prefer exactly what the model saw.  Falling back to the evidence objects
    # keeps the function usable in offline validation and legacy callers.
    evidence_map = evidence_id_text_map_from_prompt(evidence_prompt_text) or evidence_id_text_map(evidence)
    if not conclusion.supported_facts:
        return ["missing_supported_facts"], warnings
    if len(conclusion.supported_facts) > 8:
        issues.append(f"too_many_supported_facts:{len(conclusion.supported_facts)}>8")
    if conclusion.inferred_facts:
        issues.append("inferred_facts_not_allowed_in_evidence_id_mode")

    for fact_index, fact in enumerate(conclusion.supported_facts, start=1):
        if not isinstance(fact, dict):
            issues.append(f"supported_fact_{fact_index}_not_object")
            continue
        claim = str(fact.get("fact") or "").strip()
        refs = fact.get("evidence_refs")
        if not claim:
            issues.append(f"supported_fact_{fact_index}_missing_fact")
            continue
        if len(claim) > 180:
            issues.append(f"supported_fact_{fact_index}_fact_over_180")
        if not isinstance(refs, list) or not refs:
            issues.append(f"supported_fact_{fact_index}_missing_evidence_refs")
            continue
        if len(refs) > 2:
            issues.append(f"supported_fact_{fact_index}_too_many_refs:{len(refs)}>2")
        cited_texts: list[str] = []
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_not_object")
                continue
            evidence_id = str(ref.get("evidence_id") or "").strip()
            if not evidence_id:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_missing_evidence_id")
            elif evidence_id not in evidence_map:
                issues.append(
                    f"supported_fact_{fact_index}_ref_{ref_index}_evidence_id_not_found:{evidence_id}"
                )
            else:
                cited_texts.append(evidence_map[evidence_id])
            if str(ref.get("quote") or "").strip():
                warnings.append(f"supported_fact_{fact_index}_ref_{ref_index}_quote_ignored")
        cited_pool = normalize_for_evidence_match("\n".join(cited_texts))
        sensitive_missing = claim_has_unsupported_quote_required_terms(claim, cited_pool)
        if sensitive_missing:
            issues.append(
                f"supported_fact_{fact_index}_sensitive_terms_outside_cited_evidence:"
                + ",".join(sensitive_missing[:8])
            )
        question_tokens = set(extract_content_tokens(question))
        claim_tokens = [token for token in extract_content_tokens(claim) if token not in question_tokens]
        missing_tokens: list[str] = []
        for token in claim_tokens:
            normalized = normalize_for_evidence_match(token)
            if len(normalized) < GROUNDING_LONG_TOKEN_MIN_LEN + 1 or normalized in cited_pool:
                continue
            # The Chinese tokenizer intentionally keeps short predicate phrases
            # together (e.g. "同意了娜塔莉娅"). Accept a conservative fuzzy
            # match only when most of a sufficiently long token is present.
            matched_chars = sum(1 for char in set(normalized) if char in cited_pool)
            if len(normalized) >= 6 and matched_chars / len(set(normalized)) >= 0.75:
                continue
            missing_tokens.append(token)
        if missing_tokens:
            issues.append(
                f"supported_fact_{fact_index}_terms_outside_cited_evidence:"
                + ",".join(dedupe_keep_order(missing_tokens)[:8])
            )
    return issues, warnings
