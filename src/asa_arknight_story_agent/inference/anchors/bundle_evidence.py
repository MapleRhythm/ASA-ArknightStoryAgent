from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_score, evidence_text
from asa_arknight_story_agent.inference.common.lexicon import COMMON_NON_ENTITY_WORDS
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def anchor_bundle_score(text: str, core_terms: list[str], bundle_terms: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not bundle_terms:
        return 0
    compact_core_terms = dedupe_keep_order(
        re.sub(r"\s+", "", term or "")
        for term in core_terms
        if term and term not in COMMON_NON_ENTITY_WORDS
    )
    compact_bundle_terms = dedupe_keep_order(
        re.sub(r"\s+", "", term or "")
        for term in bundle_terms
        if term and term not in COMMON_NON_ENTITY_WORDS
    )
    core_hits = sum(1 for term in compact_core_terms if term and term in compact_text)
    if compact_core_terms and core_hits <= 0:
        return 0
    bundle_hits = sum(1 for term in compact_bundle_terms if term and term in compact_text)
    return core_hits * 3 + bundle_hits


def best_anchor_bundle_evidence(
    evidence: list[dict[str, Any]],
    *,
    core_terms: list[str],
    bundle_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not core_terms or not bundle_terms:
        return []
    candidates = [
        item
        for item in evidence
        if anchor_bundle_score(evidence_text(item), core_terms, bundle_terms) >= 5
    ]
    return sorted(
        candidates,
        key=lambda item: (
            anchor_bundle_score(evidence_text(item), core_terms, bundle_terms),
            evidence_score(item),
        ),
        reverse=True,
    )[:limit]
