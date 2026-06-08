from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import (
    evidence_text,
    is_moegirl_evidence,
    is_web_context_item,
    prompt_evidence_score,
)
from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    DEFINITION_EVIDENCE_MARKERS,
    DEFINITION_QUESTION_MARKERS,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta


def is_definition_or_identity_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = "\n".join([question or "", hypothesis.question or "", hypothesis.expected_answer_type or ""])
    return any(marker in text for marker in DEFINITION_QUESTION_MARKERS) or hypothesis.expected_answer_type in {
        "definition",
        "string",
    }


def definition_anchor_score(text: str, anchors: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not anchors:
        return 0
    compact_anchors = [
        re.sub(r"\s+", "", anchor or "")
        for anchor in anchors
        if anchor and anchor not in COMMON_NON_ENTITY_WORDS
    ]
    anchor_hits = [anchor for anchor in compact_anchors if anchor and anchor in compact_text]
    if not anchor_hits:
        return 0
    marker_hits = sum(1 for marker in DEFINITION_EVIDENCE_MARKERS if marker in compact_text)
    local_definition_hits = 0
    for anchor in anchor_hits[:4]:
        for marker in DEFINITION_EVIDENCE_MARKERS:
            if re.search(re.escape(anchor) + r".{0,32}" + re.escape(marker), compact_text):
                local_definition_hits += 1
                break
            if re.search(re.escape(marker) + r".{0,32}" + re.escape(anchor), compact_text):
                local_definition_hits += 1
                break
    return len(anchor_hits) * 4 + min(marker_hits, 4) + local_definition_hits * 3


def definition_source_bonus(item: dict[str, Any]) -> int:
    if is_moegirl_evidence(item):
        return -10
    if is_web_context_item(item):
        return -8
    source_path = str((item.get("document") or {}).get("source_path") or "")
    if "/data/ArknightsGameData/" in source_path or "data/ArknightsGameData/" in source_path:
        return 3
    return 0


def definition_candidate_score(item: dict[str, Any], anchors: list[str]) -> int:
    return definition_anchor_score(evidence_text(item), anchors) + definition_source_bonus(item)


def best_definition_evidence(
    evidence: list[dict[str, Any]],
    *,
    anchors: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not anchors:
        return []
    candidates = [
        item
        for item in evidence
        if definition_anchor_score(evidence_text(item), anchors) >= 7
    ]
    return sorted(
        candidates,
        key=lambda item: (
            definition_candidate_score(item, anchors),
            prompt_evidence_score(item),
        ),
        reverse=True,
    )[:limit]
