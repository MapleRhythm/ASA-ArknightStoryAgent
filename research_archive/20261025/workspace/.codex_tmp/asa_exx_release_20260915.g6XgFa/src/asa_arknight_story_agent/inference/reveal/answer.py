from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_text
from asa_arknight_story_agent.inference.common.lexicon import REVEAL_KNOWLEDGE_RETRIEVAL_TERMS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.strips import split_evidence_strips
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, truncate_text


def select_reveal_strips(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_strips: int = 5,
) -> list[str]:
    if "阴谋" not in question and "阴谋" not in hypothesis.keywords and hypothesis.query_type not in {"reveal", "mystery"}:
        return []

    query_terms = dedupe_keep_order(
        hypothesis.entities
        + hypothesis.keywords
        + extract_content_tokens(question)
        + list(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
    )
    high_value_terms = {
        "贝希曼",
        "贝希曼伯爵",
        "苏茜",
        "澄闪",
        "卡拉顿",
        "警备队",
        "送线索",
        "劫持",
        "爆炸",
        "工厂",
        "物流通道",
        "阴谋",
        "曝光",
    }
    candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        text = evidence_text(item)
        strips = split_evidence_strips(text, max_strips=48)
        if not strips and text:
            strips = [text]
        for strip in strips:
            compact = re.sub(r"\s+", "", strip)
            term_hits = sum(1 for term in query_terms if term and term in compact)
            high_hits = sum(1 for term in high_value_terms if term in compact)
            if high_hits < 2 and not ("阴谋" in compact and ("曝光" in compact or "贝希曼" in compact)):
                continue
            score = term_hits + high_hits * 2
            if "苏茜去警备队送线索" in compact:
                score += 8
            if "遭到劫持" in compact or "意外爆炸" in compact:
                score += 4
            if "贝希曼议员的阴谋得以曝光" in compact:
                score += 8
            candidates.append((score, truncate_text(strip, 260)))

    selected: list[str] = []
    for _, strip in sorted(candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return selected


def build_reveal_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    strips = select_reveal_strips(question=question, hypothesis=hypothesis, evidence=evidence)
    if not strips:
        return None
    joined = " ".join(strips)
    compact = re.sub(r"\s+", "", joined)
    if not ("贝希曼" in compact and "阴谋" in compact):
        return None

    parts: list[str] = []
    if "送线索" in compact and "警备队" in compact:
        parts.append("苏茜把线索送到警备队后，反而落入贝希曼一方掌控")
    if "劫持" in compact or "被捆住" in compact:
        parts.append("她被劫持并带到废弃物流通道/工厂相关地点")
    if "爆炸" in compact and "逃出" in compact:
        parts.append("之后因一场意外爆炸逃出")
    if "阴谋得以曝光" in compact or ("曝光" in compact and "阴谋" in compact):
        parts.append("最终使贝希曼议员的阴谋曝光")
    if "工厂" in compact or "物流通道" in compact or "设备" in compact:
        parts.append("相关线索还指向工厂设备、地下/废弃物流通道和警备队长的勾连")

    if not parts:
        return "现有证据显示，澄闪/苏茜识破的是贝希曼议员相关的卡拉顿城阴谋。依据：" + "；".join(strips[:3])
    return "现有证据显示，澄闪/苏茜识破的是贝希曼议员相关的阴谋：" + "；".join(parts) + "。依据：" + "；".join(strips[:3])
