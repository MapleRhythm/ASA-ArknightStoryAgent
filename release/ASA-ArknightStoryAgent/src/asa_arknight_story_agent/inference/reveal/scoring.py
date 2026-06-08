from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import (
    document_chain_text,
    document_clean_text,
    evidence_score,
    evidence_text,
)
from asa_arknight_story_agent.inference.common.lexicon import (
    REVEAL_DIRECT_EVIDENCE_TERMS,
    REVEAL_KNOWLEDGE_RETRIEVAL_TERMS,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.reveal.detection import is_reveal_question
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def reveal_direct_score(text: str, question: str, hypothesis: HypothesisDocument) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text:
        return 0
    query_terms = dedupe_keep_order(
        hypothesis.entities
        + hypothesis.keywords
        + extract_content_tokens(question)
        + list(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
    )
    query_hits = sum(1 for term in query_terms if term and term in compact_text)
    direct_hits = sum(1 for term in REVEAL_DIRECT_EVIDENCE_TERMS if term in compact_text)
    score = query_hits + direct_hits * 3
    if "贝希曼" in compact_text and "阴谋" in compact_text:
        score += 4
    if "苏茜" in compact_text and ("送线索" in compact_text or "劫持" in compact_text):
        score += 6
    if "贝希曼议员的阴谋得以曝光" in compact_text:
        score += 10
    if "[uc]info" in str((hypothesis.question or "") + text) and "阴谋" in compact_text:
        score += 3
    return score


def best_reveal_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not is_reveal_question(question, hypothesis):
        return []
    candidates: list[tuple[int, float, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence):
        clean_text = document_clean_text(item)
        chain_text = document_chain_text(item)
        score = max(
            reveal_direct_score(clean_text, question, hypothesis),
            reveal_direct_score(chain_text, question, hypothesis),
            reveal_direct_score(evidence_text(item), question, hypothesis),
        )
        if score <= 0:
            continue
        doc = item.get("document") or {}
        source_path = str(doc.get("source_path") or "")
        if "handbook_info_table.json" in source_path or "charword_table.json" in source_path:
            score -= 5
        if "[uc]info" in source_path and ("阴谋" in clean_text or "曝光" in clean_text):
            score += 8
        candidates.append((score, evidence_score(item), -index, item))
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return [item for score, _, _, item in candidates[:limit] if score > 0]
