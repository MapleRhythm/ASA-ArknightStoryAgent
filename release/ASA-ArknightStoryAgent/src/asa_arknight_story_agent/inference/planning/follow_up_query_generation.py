from __future__ import annotations

from asa_arknight_story_agent.inference.anchors.terms import extract_action_targets
from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    IDENTITY_HINT_WORDS,
    REVEAL_KNOWLEDGE_RETRIEVAL_TERMS,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import (
    expand_related_retrieval_terms,
    extract_content_tokens,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def build_follow_up_hypothesis_queries(
    question: str,
    hypothesis: HypothesisDocument,
) -> list[str]:
    queries: list[str] = []
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""
    action_targets = extract_action_targets(question + "\n" + hypothesis.question)
    focus_terms = dedupe_keep_order(
        action_targets
        + extract_content_tokens(question)
        + hypothesis.entities[:4]
        + hypothesis.keywords[:4]
    )
    related_terms = expand_related_retrieval_terms(focus_terms)
    is_reason_query = any(token in question for token in ("为什么", "为何", "原因", "目的", "动机", "真正"))

    if action_targets:
        for target in action_targets[:3]:
            queries.append(target)
            if primary_entity and primary_entity != target:
                queries.append(f"{primary_entity} {target}")
            if is_reason_query:
                queries.append(f"{target} 目的 原因")
                if primary_entity and primary_entity != target:
                    queries.append(f"{primary_entity} {target} 目的 原因")
        if related_terms:
            queries.append(" ".join(dedupe_keep_order([*action_targets[:3], *related_terms[:8]])))

    if "阴谋" in hypothesis.keywords or "阴谋" in question:
        bridge_entities = [
            entity
            for entity in hypothesis.entities[:6]
            if entity != primary_entity and entity not in COMMON_NON_ENTITY_WORDS
        ]
        for entity in bridge_entities[:4]:
            queries.append(f"{entity} 阴谋")
        if any("卡拉顿" in term for term in hypothesis.entities + hypothesis.keywords):
            for entity in bridge_entities[:3]:
                queries.append(f"{entity} 卡拉顿 阴谋")
            queries.extend(
                [
                    "阴云火花 贝希曼 阴谋 曝光",
                    "卡拉顿 贝希曼 议员 阴谋",
                    "苏茜 警备队 送线索 劫持 爆炸",
                    "贝希曼 工厂 地下 设备 物流通道",
                    "贝希曼 议会 拨款 报告损失 钱的窟窿",
                    "贝希曼 栽赃 感染者",
                ]
            )

    if primary_entity and any(token in question for token in IDENTITY_HINT_WORDS):
        queries.extend(
            [
                f"{primary_entity} 身份 来历",
                f"{primary_entity} 身世 真相",
            ]
        )

    if "阴谋" in question or "阴谋" in hypothesis.keywords or hypothesis.query_type in {"reveal", "mystery"}:
        queries.extend(term for term in REVEAL_KNOWLEDGE_RETRIEVAL_TERMS if term in hypothesis.keywords)
    queries.extend(hypothesis.keywords[:8])

    for entity in hypothesis.entities[:4]:
        queries.append(entity)
        for keyword in hypothesis.keywords[:4]:
            if keyword != entity:
                queries.append(f"{entity} {keyword}")
    if related_terms:
        for term in focus_terms[:4]:
            queries.append(" ".join(dedupe_keep_order([term, *related_terms[:6]])))

    deduped_queries: list[str] = []
    seen: set[str] = {question.strip()}
    for query in queries:
        normalized = query.strip()
        query_terms = normalized.split()
        if (
            not normalized
            or normalized in seen
            or len(query_terms) >= 2 and len(set(query_terms)) == 1
        ):
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:14]
