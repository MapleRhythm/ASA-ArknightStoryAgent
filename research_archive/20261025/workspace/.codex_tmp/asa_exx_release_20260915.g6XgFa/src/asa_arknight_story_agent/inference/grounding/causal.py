from __future__ import annotations

import re

from asa_arknight_story_agent.inference.anchors.terms import extract_question_anchor_terms
from asa_arknight_story_agent.inference.grounding.identity import is_identity_question
from asa_arknight_story_agent.inference.common.lexicon import (
    ACTION_ANSWER_MARKERS,
    ACTION_WORDS,
    DOMAIN_ANCHOR_TERMS,
)
from asa_arknight_story_agent.inference.common.patterns import ACTION_TARGET_RE
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


GROUNDING_CAUSAL_MARKERS = ACTION_ANSWER_MARKERS


def has_direct_causal_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence_pool: str,
) -> bool:
    if is_identity_question(question, hypothesis):
        return False
    if not any(token in question + hypothesis.expected_answer_type for token in ("为什么", "为何", "原因", "动机", "目的")):
        return False

    compact_evidence = re.sub(r"\s+", "", strip_internal_evidence_meta(evidence_pool))
    if not compact_evidence:
        return False

    anchors = extract_question_anchor_terms(question, hypothesis)
    high_value_anchors = [
        anchor
        for anchor in anchors
        if anchor in DOMAIN_ANCHOR_TERMS
        or anchor in hypothesis.entities
        or anchor in hypothesis.keywords
        or anchor in ACTION_WORDS
    ]
    anchor_hits = [
        anchor
        for anchor in dedupe_keep_order(high_value_anchors)
        if re.sub(r"\s+", "", anchor) in compact_evidence
    ]
    if len(anchor_hits) < 2:
        return False

    has_action_target = any(match.group(1) and match.group(1) in compact_evidence for match in ACTION_TARGET_RE.finditer(question))
    has_action_word = any(word in compact_evidence for word in ACTION_WORDS)
    has_causal_marker = any(marker in compact_evidence for marker in GROUNDING_CAUSAL_MARKERS)
    return has_causal_marker and (has_action_target or has_action_word)
