from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.strips import split_evidence_strips
from asa_arknight_story_agent.inference.evidence.texts import document_chain_text
from asa_arknight_story_agent.inference.event_reference.detection import is_event_reference_question
from asa_arknight_story_agent.inference.event_reference.terms import event_reference_anchor_terms
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.common.text_utils import (
    dedupe_keep_order,
    strip_internal_evidence_meta,
    truncate_text,
)


EVENT_REFERENCE_MARKERS = (
    "当时",
    "后来",
    "再后来",
    "因为",
    "因此",
    "导致",
    "结果",
    "遭遇",
    "发生",
    "病逝",
    "联姻",
    "再婚",
    "流言",
    "恶名",
    "仕途",
    "生计",
    "真相",
    "实情",
    "缘由",
    "做错",
    "拒绝",
    "权力",
    "陪葬",
    "牵连",
    "为难",
)


def event_reference_query_terms(
    question: str,
    hypothesis: HypothesisDocument,
    anchors: list[str],
) -> list[str]:
    return dedupe_keep_order(
        anchors
        + hypothesis.entities
        + hypothesis.keywords
        + extract_content_tokens(question)
    )[:16]


def event_reference_candidate_text(item: dict[str, Any]) -> str:
    doc = item.get("document") or item
    text = strip_internal_evidence_meta(
        str(doc.get("clean_text") or doc.get("search_text") or "")
    ).strip()
    if text:
        return text
    return document_chain_text(item)


def score_event_reference_window(
    *,
    compact_window: str,
    anchors: list[str],
    query_terms: list[str],
) -> int:
    score = 10
    score += sum(2 for anchor in anchors if anchor and anchor in compact_window)
    score += sum(1 for term in query_terms if term and re.sub(r"\s+", "", term) in compact_window)
    score += sum(1 for marker in EVENT_REFERENCE_MARKERS if marker in compact_window)
    return score


def select_event_reference_strips(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_strips: int = 3,
) -> tuple[str, list[str]]:
    if not is_event_reference_question(question, hypothesis):
        return "", []

    anchors = event_reference_anchor_terms(question, hypothesis)
    if not anchors:
        return "", []
    display_anchor = anchors[0]
    query_terms = event_reference_query_terms(question, hypothesis, anchors)

    candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        text = event_reference_candidate_text(item)
        if not text:
            continue
        strips = split_evidence_strips(text, max_strips=80)
        if not strips:
            continue
        for index, strip in enumerate(strips):
            compact_strip = re.sub(r"\s+", "", strip)
            if not any(anchor and anchor in compact_strip for anchor in anchors):
                continue
            start = max(0, index - 3)
            end = min(len(strips), index + 3)
            window = "；".join(strips[start:end])
            compact_window = re.sub(r"\s+", "", window)
            candidates.append(
                (
                    score_event_reference_window(
                        compact_window=compact_window,
                        anchors=anchors,
                        query_terms=query_terms,
                    ),
                    truncate_text(window, 520),
                )
            )

    selected: list[str] = []
    for _, strip in sorted(candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return display_anchor, selected
