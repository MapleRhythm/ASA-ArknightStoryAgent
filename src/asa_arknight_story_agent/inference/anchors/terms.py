from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.lexicon import (
    ACTION_WORDS,
    COMMON_NON_ENTITY_WORDS,
    DOMAIN_ANCHOR_TERMS,
)
from asa_arknight_story_agent.inference.common.patterns import ACTION_TARGET_BOUNDARY_RE, ACTION_TARGET_RE, QUOTED_TERM_RE
from asa_arknight_story_agent.inference.pipeline.constants import (
    NOISY_RETRIEVAL_TOKENS,
    NOISY_TOKEN_MARKERS,
    PRONOUN_REFERENCES,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import (
    expand_related_retrieval_terms,
    extract_content_tokens,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def clean_anchor_term(term: str) -> str:
    cleaned = re.sub(r"\s+", "", term or "").strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")
    if not cleaned:
        return ""
    cleaned = re.split(
        r"(?:这种题|这类题|这个问题|这种问题|题目|问题|反而|检索|证据|不足|为什么|为何|怎么|如何|吗|呢|啊|吧)",
        cleaned,
        maxsplit=1,
    )[0]
    return cleaned.strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")


def extract_question_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    text = "\n".join(
        [
            question,
            hypothesis.question,
            " ".join(hypothesis.entities),
            " ".join(hypothesis.keywords),
            hypothesis.expected_answer_type,
        ]
    )
    anchors: list[str] = []

    for raw_term in QUOTED_TERM_RE.findall(text):
        anchors.append(clean_anchor_term(raw_term))
    for match in ACTION_TARGET_RE.finditer(text):
        anchors.append(clean_anchor_term(match.group(1)))
    anchors.extend(extract_action_targets(text))
    for action in ACTION_WORDS:
        if action in text:
            anchors.append(action)
    anchors.extend(term for term in DOMAIN_ANCHOR_TERMS if term in text)
    anchors.extend(extract_content_tokens(question))
    anchors.extend(term for term in hypothesis.entities if term)
    anchors.extend(term for term in hypothesis.keywords if term)

    cleaned_anchors: list[str] = []
    for term in anchors:
        cleaned = clean_anchor_term(term)
        if (
            not cleaned
            or cleaned in COMMON_NON_ENTITY_WORDS
            or cleaned in NOISY_RETRIEVAL_TOKENS
            or cleaned in PRONOUN_REFERENCES
            or len(cleaned) == 1 and not cleaned.isascii()
            or any(marker in cleaned for marker in NOISY_TOKEN_MARKERS)
        ):
            continue
        cleaned_anchors.append(cleaned)
    deduped_anchors = dedupe_keep_order(cleaned_anchors)
    related_anchors = expand_related_retrieval_terms(deduped_anchors)
    return dedupe_keep_order(deduped_anchors + related_anchors)[:24]


def anchor_hit_count(text: str, anchors: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not anchors:
        return 0
    return sum(1 for anchor in anchors if re.sub(r"\s+", "", anchor) in compact_text)


def extract_action_targets(text: str) -> list[str]:
    source = text or ""
    targets: list[str] = []
    for match in ACTION_TARGET_RE.finditer(source):
        raw_target = match.group(1)
        raw_target = ACTION_TARGET_BOUNDARY_RE.split(raw_target, maxsplit=1)[0]
        cleaned = clean_anchor_term(raw_target)
        if cleaned:
            targets.append(cleaned)
    if any(action in source for action in ACTION_WORDS):
        targets.extend(term for term in DOMAIN_ANCHOR_TERMS if term in source)
    return dedupe_keep_order(
        target
        for target in targets
        if target and target not in ACTION_WORDS and target not in COMMON_NON_ENTITY_WORDS
    )
