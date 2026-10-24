from __future__ import annotations

import re

from asa_arknight_story_agent.config import OPERATOR_ALIAS_MAP_PATH
from asa_arknight_story_agent.data.alias_map import load_operator_alias_map
from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    IDENTITY_HINT_WORDS,
)
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import is_entity_candidate
from asa_arknight_story_agent.inference.reveal.detection import is_reveal_question
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def is_identity_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = question + "\n" + hypothesis.expected_answer_type
    return any(token in text for token in IDENTITY_HINT_WORDS)


def primary_entity_anchor_required(question: str, hypothesis: HypothesisDocument) -> str:
    if is_reveal_question(question, hypothesis):
        return ""
    if not any(token in question for token in ("一事", "具体是指", "指的是什么", "指什么", "是谁", "是什么", "身份", "来历")):
        return ""
    for entity in hypothesis.entities:
        normalized = re.sub(r"\s+", "", entity)
        if (
            len(normalized) >= 3
            and is_entity_candidate(normalized)
            and normalized not in COMMON_NON_ENTITY_WORDS
            and normalized not in NOISY_RETRIEVAL_TOKENS
        ):
            return normalized
    return ""


def anchor_aliases(anchor: str) -> list[str]:
    aliases = [anchor]
    try:
        alias_map = load_operator_alias_map(OPERATOR_ALIAS_MAP_PATH)
        if alias_map:
            aliases.extend(alias_map.expand([anchor]))
    except Exception:
        pass
    return dedupe_keep_order([re.sub(r"\s+", "", alias) for alias in aliases if alias])


def unsupported_required_entity_anchor(
    question: str,
    hypothesis: HypothesisDocument,
    evidence_pool: str,
) -> str:
    anchor = primary_entity_anchor_required(question, hypothesis)
    if not anchor:
        return ""
    compact_evidence = re.sub(r"\s+", "", strip_internal_evidence_meta(evidence_pool))
    if not compact_evidence:
        return anchor
    if any(alias and alias in compact_evidence for alias in anchor_aliases(anchor)):
        return ""
    return anchor
