from __future__ import annotations

from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.grounding.quote_match_utils import normalize_for_evidence_match
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


QUOTE_REQUIRED_RELATION_TERMS = (
    "未婚夫",
    "未婚妻",
    "父亲",
    "母亲",
    "亲生",
    "幕后主使",
    "真正原因",
    "建造",
    "开发",
    "制造",
    "创造",
    "设计",
    "源石计划",
    "种族整合",
    "整合统一",
    "仿生学",
    "目的",
    "动机",
    "旨在",
    "服务于",
)


def grounded_supported_fact_texts(conclusion: ConclusionResult) -> list[str]:
    texts: list[str] = []
    for fact in conclusion.supported_facts:
        if not isinstance(fact, dict):
            continue
        fact_text = str(fact.get("fact") or "").strip()
        if fact_text:
            texts.append(fact_text)
        for ref in fact.get("evidence_refs") or []:
            if isinstance(ref, dict):
                quote = str(ref.get("quote") or "").strip()
                if quote:
                    texts.append(quote)
    for fact in conclusion.inferred_facts:
        if isinstance(fact, dict):
            fact_text = str(fact.get("fact") or "").strip()
        else:
            fact_text = str(fact or "").strip()
        if fact_text:
            texts.append(fact_text)
    return dedupe_keep_order(texts)


def grounded_quote_texts(conclusion: ConclusionResult) -> list[str]:
    texts: list[str] = []
    for fact in conclusion.supported_facts:
        if not isinstance(fact, dict):
            continue
        for ref in fact.get("evidence_refs") or []:
            if isinstance(ref, dict):
                quote = str(ref.get("quote") or "").strip()
                if quote:
                    texts.append(quote)
    return dedupe_keep_order(texts)


def claim_has_unsupported_quote_required_terms(claim: str, quote_pool: str) -> list[str]:
    missing: list[str] = []
    for term in QUOTE_REQUIRED_RELATION_TERMS:
        if term in claim and normalize_for_evidence_match(term) not in quote_pool:
            missing.append(term)
    for token in extract_content_tokens(claim):
        if token.isascii() and len(token) >= 3 and normalize_for_evidence_match(token) not in quote_pool:
            missing.append(token)
    return dedupe_keep_order(missing)


def answer_from_grounded_facts(conclusion: ConclusionResult) -> str:
    quote_pool = normalize_for_evidence_match("\n".join(grounded_quote_texts(conclusion)))
    facts = [
        str(fact.get("fact") or "").strip()
        for fact in conclusion.supported_facts
        if (
            isinstance(fact, dict)
            and str(fact.get("fact") or "").strip()
            and not claim_has_unsupported_quote_required_terms(str(fact.get("fact") or ""), quote_pool)
        )
    ]
    inferred = [
        str(fact.get("fact") or "").strip() if isinstance(fact, dict) else str(fact or "").strip()
        for fact in conclusion.inferred_facts
    ]
    inferred = [item for item in inferred if item and not claim_has_unsupported_quote_required_terms(item, quote_pool)]
    selected = dedupe_keep_order([*facts, *inferred])
    if not selected:
        return conclusion.answer
    if len(selected) == 1:
        return selected[0]
    return "根据当前证据可确认：" + "；".join(selected) + "。"
