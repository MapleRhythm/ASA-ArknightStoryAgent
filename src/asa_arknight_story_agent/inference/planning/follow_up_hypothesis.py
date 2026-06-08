from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.terms import extract_action_targets
from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    REVEAL_KNOWLEDGE_RETRIEVAL_TERMS,
)
from asa_arknight_story_agent.inference.planning.missing_slot_queries import clean_missing_slots_for_retrieval
from asa_arknight_story_agent.inference.common.patterns import CONSPIRACY_ANCHOR_RE, REAL_NAME_RE
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import (
    expand_related_retrieval_terms,
    extract_content_tokens,
    infer_query_type,
    is_entity_candidate,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def merge_hypotheses(base: HypothesisDocument, follow_up: HypothesisDocument) -> HypothesisDocument:
    return HypothesisDocument(
        question=base.question,
        intent=base.intent,
        query_type=follow_up.query_type or base.query_type,
        entities=dedupe_keep_order(base.entities + follow_up.entities)[:12],
        keywords=dedupe_keep_order(base.keywords + follow_up.keywords)[:20],
        expected_answer_type=follow_up.expected_answer_type or base.expected_answer_type,
        dialogue_context=base.dialogue_context,
    )


def build_heuristic_follow_up_hypothesis(
    question: str,
    current_hypothesis: HypothesisDocument,
    missing_slots: list[str],
) -> HypothesisDocument:
    missing_slots = clean_missing_slots_for_retrieval(missing_slots)
    slot_text = " ".join(slot for slot in missing_slots if slot)
    slot_terms = extract_content_tokens(slot_text)
    slot_entities = [term for term in slot_terms if is_entity_candidate(term)]
    action_targets = extract_action_targets(question + "\n" + current_hypothesis.question)
    is_reason_query = any(token in question for token in ("为什么", "为何", "原因", "目的", "动机", "真正"))

    bridge_terms: list[str] = []
    if is_reason_query:
        bridge_terms.extend(["原因", "目的", "直接原因", "具体原因"])
        for target in action_targets[:3]:
            bridge_terms.extend([f"{target} 目的", f"{target} 原因"])
            if current_hypothesis.entities and current_hypothesis.entities[0] != target:
                bridge_terms.extend(
                    [
                        f"{current_hypothesis.entities[0]} {target}",
                        f"{current_hypothesis.entities[0]} {target} 原因",
                    ]
                )

    focus_terms = dedupe_keep_order(
        action_targets + current_hypothesis.entities[:4] + current_hypothesis.keywords[:8] + slot_terms + bridge_terms
    )
    related_terms = expand_related_retrieval_terms(focus_terms)
    expected_answer_type = current_hypothesis.expected_answer_type
    if is_reason_query and action_targets:
        expected_answer_type = "short_text"

    return HypothesisDocument(
        question=current_hypothesis.question or question,
        intent=current_hypothesis.intent,
        query_type=current_hypothesis.query_type or infer_query_type(
            question,
            current_hypothesis.intent,
            expected_answer_type,
        ),
        entities=dedupe_keep_order(current_hypothesis.entities + slot_entities)[:12],
        keywords=dedupe_keep_order(current_hypothesis.keywords + slot_terms + bridge_terms + related_terms)[:24],
        expected_answer_type=expected_answer_type,
        dialogue_context=current_hypothesis.dialogue_context,
    )


def enrich_follow_up_with_evidence_terms(
    hypothesis: HypothesisDocument,
    *,
    question: str,
    evidence: list[dict[str, Any]],
    missing_slots: list[str],
) -> HypothesisDocument:
    context_text = "\n".join(
        [question, hypothesis.question, *missing_slots]
        + [str(item["document"].get("clean_text") or "") for item in evidence[:4]]
    )
    if "阴谋" not in context_text and "具体" not in context_text:
        return hypothesis

    bridge_entities: list[str] = []
    bridge_keywords: list[str] = []
    for item in evidence[:4]:
        text = str(item["document"].get("clean_text") or "")
        for match in REAL_NAME_RE.finditer(text):
            full_name = match.group(1).strip()
            short_name = full_name.split("·", 1)[0].strip()
            bridge_entities.extend([short_name, full_name])
            bridge_keywords.extend([short_name, full_name])
        for match in CONSPIRACY_ANCHOR_RE.finditer(text):
            location = match.group(1).strip()
            bridge_entities.extend([location, f"{location}城议员", "城议员"])
            bridge_keywords.extend([location, f"{location}城议员", "城议员", "阴谋"])
        for term in REVEAL_KNOWLEDGE_RETRIEVAL_TERMS:
            if term in text:
                bridge_keywords.append(term)

    if "阴谋" in context_text or any(token in hypothesis.query_type for token in ("reveal", "mystery")):
        bridge_keywords.extend(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
        if any("卡拉顿" in term for term in hypothesis.entities + hypothesis.keywords) or "卡拉顿" in context_text:
            bridge_entities.extend(["卡拉顿", "卡拉顿城议员"])

    bridge_entities = dedupe_keep_order(
        [
            term
            for term in bridge_entities
            if term and term not in hypothesis.entities and term not in COMMON_NON_ENTITY_WORDS
        ]
    )
    bridge_keywords = dedupe_keep_order(
        [
            term
            for term in bridge_keywords
            if term and term not in COMMON_NON_ENTITY_WORDS
        ]
    )
    if not bridge_entities and not bridge_keywords:
        return hypothesis

    return HypothesisDocument(
        question=hypothesis.question,
        intent=hypothesis.intent,
        query_type=hypothesis.query_type,
        entities=dedupe_keep_order(hypothesis.entities[:1] + bridge_entities + hypothesis.entities[1:])[:12],
        keywords=dedupe_keep_order(bridge_keywords + hypothesis.keywords)[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )
