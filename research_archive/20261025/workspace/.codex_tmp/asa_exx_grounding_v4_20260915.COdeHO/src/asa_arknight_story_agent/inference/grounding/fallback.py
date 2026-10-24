from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text
from asa_arknight_story_agent.inference.common.lexicon import COMMON_NON_ENTITY_WORDS
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens, extract_entities
from asa_arknight_story_agent.inference.common.text_utils import (
    dedupe_keep_order,
    strip_internal_evidence_meta,
    truncate_text,
)


def build_grounded_fallback_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    missing_tokens: list[str],
) -> str:
    query_terms = [
        term
        for term in dedupe_keep_order(
            extract_entities(question, hypothesis.dialogue_context)
            + hypothesis.entities
            + hypothesis.keywords
            + extract_content_tokens(question)
        )
        if term and term not in COMMON_NON_ENTITY_WORDS and term not in NOISY_RETRIEVAL_TOKENS
    ]
    selected: list[str] = []
    for item in evidence[:6]:
        document = item.get("document") or {}
        text = strip_internal_evidence_meta(
            str(item.get("evidence_chain_text") or document.get("clean_text") or document.get("search_text") or "")
        ).strip()
        if not text:
            continue
        if query_terms and not any(term in text for term in query_terms[:10]):
            continue
        text = truncate_text(re.sub(r"\s+", " ", text), 180)
        if text and text not in selected:
            selected.append(text)
        if len(selected) >= 3:
            break

    if not selected:
        return "现有检索证据不足以确认答案所需的关键表述。"

    answer_lines = ["当前证据只能确认以下片段事实："]
    answer_lines.extend(f"{index}. {text}" for index, text in enumerate(selected, start=1))
    answer_lines.append("缺少足以完整回答用户问题的直接因果或身份绑定证据。")
    return "\n".join(answer_lines)


def has_answerable_evidence(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        if len(strip_internal_evidence_meta(text).strip()) >= 80:
            return True
    return False
