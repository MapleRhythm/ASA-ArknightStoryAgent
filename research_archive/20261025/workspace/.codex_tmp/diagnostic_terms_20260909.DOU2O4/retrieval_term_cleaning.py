"""Keep internal control/grounding diagnostics out of story search terms."""
from __future__ import annotations

from dataclasses import replace
import re

from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.missing_slot_queries import POLLUTED_RETRIEVAL_SLOT_PATTERNS


CONTROL_TERMS = frozenset({
    "answer_directly", "retrieve_more", "abstain", "clarify_user",
    "supported_facts", "inferred_facts", "missing_slots",
    "follow_up_hypothesis", "final_answer", "evidence_refs", "evidence_id",
    "missing_supported_facts", "missing_inferred_facts", "missing_quotes",
})
DIAGNOSTIC_PREFIX_RE = re.compile(
    r"^(?:(?:supported|inferred)_fact_\d+_|"
    r"answer_quote_total_over_|too_many_(?:supported|inferred)_facts|"
    r"quote_(?:not_found|over_|total_over_))",
    re.IGNORECASE,
)
# These words can be legitimate story/query terms on their own. Remove only
# alongside the recognizable "缺少可校验 quote 支撑" fragment, unless user-requested.
DIAGNOSTIC_FRAGMENTS = frozenset({"quote", "支撑"})


def is_internal_diagnostic_term(term: str) -> bool:
    lowered = term.casefold()
    return (
        lowered in CONTROL_TERMS
        or "缺少可校验" in lowered
        or bool(DIAGNOSTIC_PREFIX_RE.match(term))
        or any(pattern.casefold() in lowered for pattern in POLLUTED_RETRIEVAL_SLOT_PATTERNS)
    )


def clean_hypothesis_terms(
    entities: list[str], keywords: list[str], *, question: str,
) -> tuple[list[str], list[str]]:
    """Preserve explicit user terms and ordinary names, not all-ASCII filtering."""
    normalized_question = str(question or "").casefold()

    def user_requested(term: str) -> bool:
        return bool(term) and term.casefold() in normalized_question

    combined = [str(term or "").strip() for term in [*entities, *keywords]]
    diagnostic_context = any(
        term and "缺少可校验" in term and not user_requested(term)
        for term in combined
    )

    def clean(terms: list[str]) -> list[str]:
        kept: list[str] = []
        for raw in terms:
            term = str(raw or "").strip()
            if not term:
                continue
            if user_requested(term):
                kept.append(term)
                continue
            if is_internal_diagnostic_term(term):
                continue
            if diagnostic_context and term.casefold() in DIAGNOSTIC_FRAGMENTS:
                continue
            kept.append(term)
        return dedupe_keep_order(kept)

    return clean(entities), clean(keywords)


def sanitize_retrieval_hypothesis(
    hypothesis: HypothesisDocument, *, question: str | None = None,
) -> HypothesisDocument:
    entities, keywords = clean_hypothesis_terms(
        hypothesis.entities, hypothesis.keywords,
        question=hypothesis.question if question is None else question,
    )
    if entities == hypothesis.entities and keywords == hypothesis.keywords:
        return hypothesis
    return replace(hypothesis, entities=entities, keywords=keywords)
