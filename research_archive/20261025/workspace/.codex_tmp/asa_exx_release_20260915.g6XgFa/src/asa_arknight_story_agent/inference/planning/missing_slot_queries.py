from __future__ import annotations

import re

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, truncate_text


POLLUTED_RETRIEVAL_SLOT_PATTERNS = (
    "supported_fact_",
    "quote_not_found",
    "quote_over_",
    "quote_total_over_",
    "missing_quote",
    "missing_evidence_refs",
    "not_object",
    "answer_directly 缺少可校验 quote 支撑",
    "final_answer_has_terms_outside_supported_facts",
    "has_terms_outside_quotes",
    "grounding 校验",
    "JSON-like 结论已使用启发式续检索",
    "tuple-like 结论已转换为续检索",
    "follow_up_hypothesis 不可用",
    "模型未返回 follow_up_hypothesis",
)


def clean_missing_slots_for_retrieval(missing_slots: list[str]) -> list[str]:
    cleaned: list[str] = []
    for slot in missing_slots:
        text = str(slot or "").strip()
        if not text:
            continue
        if any(pattern in text for pattern in POLLUTED_RETRIEVAL_SLOT_PATTERNS):
            continue
        if re.fullmatch(r"[A-Za-z_0-9:>\-]+", text):
            continue
        cleaned.append(text)
    return dedupe_keep_order(cleaned)[:8]


def build_missing_slot_queries(
    hypothesis: HypothesisDocument,
    missing_slots: list[str],
) -> list[str]:
    missing_slots = clean_missing_slots_for_retrieval(missing_slots)
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""
    queries: list[str] = []
    for slot in missing_slots[:6]:
        slot_parts = [
            part.strip(" ：:，,。；;")
            for part in re.split(r"[|/；;。]+", slot)
            if part.strip(" ：:，,。；;")
        ]
        for slot_part in slot_parts[:3]:
            compact_slot = truncate_text(slot_part, 32)
            queries.append(compact_slot)
            if primary_entity:
                queries.append(f"{primary_entity} {compact_slot}")
    deduped_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:8]
