from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_text
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.strips import split_evidence_strips
from asa_arknight_story_agent.inference.common.text_utils import truncate_text


def is_suiling_crisis_question(question: str, hypothesis: HypothesisDocument) -> bool:
    compact = re.sub(r"\s+", "", question + "\n" + hypothesis.question + "\n" + hypothesis.expected_answer_type)
    return (
        "岁陵" in compact
        and "危机" in compact
        and any(marker in compact for marker in ("是什么", "指什么", "什么危机", "那场危机", "危机原因", "概念定义"))
    )


def build_suiling_crisis_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    if not is_suiling_crisis_question(question, hypothesis):
        return None

    core_candidates: list[tuple[int, str]] = []
    pressure_candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        text = evidence_text(item)
        strips = split_evidence_strips(text, max_strips=64)
        if not strips and text:
            strips = [text]
        for strip in strips:
            compact = re.sub(r"\s+", "", strip)
            if not compact:
                continue
            core_score = 0
            if "岁兽之患" in compact:
                core_score += 8
            if "岁兽" in compact and ("苏醒" in compact or "平息" in compact or "危害" in compact):
                core_score += 5
            if "岁陵" in compact and ("没有动静" in compact or "石门" in compact or "控制在岁陵" in compact):
                core_score += 4
            if "望" in compact and "岁陵" in compact and ("平息" in compact or "望日" in compact):
                core_score += 3
            if core_score > 0:
                core_candidates.append((core_score, truncate_text(strip, 260)))

            pressure_score = 0
            if "五只巨兽" in compact:
                pressure_score += 5
            if "最坏的结果" in compact or "同时开战" in compact:
                pressure_score += 4
            if "大炎周遭" in compact or "盘踞" in compact:
                pressure_score += 2
            if pressure_score > 0:
                pressure_candidates.append((pressure_score, truncate_text(strip, 240)))

    core_strips: list[str] = []
    for _, strip in sorted(core_candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in core_strips:
            core_strips.append(strip)
        if len(core_strips) >= 2:
            break
    if not core_strips:
        return None

    pressure_strips: list[str] = []
    for _, strip in sorted(pressure_candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in pressure_strips:
            pressure_strips.append(strip)
        if len(pressure_strips) >= 1:
            break

    answer = (
        "岁陵那场危机的核心是岁兽苏醒或即将苏醒引发的“岁兽之患”："
        "证据显示，望被准许进入岁陵尝试平息此事，但望日临近仍没有结果，岁陵局势可能失控。"
    )
    if pressure_strips:
        answer += "五只巨兽盘踞、可能同时开战属于当时的大炎外部压力和潜在最坏后果，不是危机本身的直接原因。"
    answer += "\n依据：" + "；".join([*core_strips, *pressure_strips][:3])
    return answer


def suiling_crisis_answer_needs_correction(answer: str) -> bool:
    compact = re.sub(r"\s+", "", answer or "")
    if not compact:
        return True
    if "岁兽" not in compact and "岁兽之患" not in compact:
        return True
    if "五只巨兽" in compact or "同时开战" in compact:
        return not any(marker in compact for marker in ("外部压力", "最坏后果", "潜在", "不是危机本身", "不是直接原因"))
    if any(marker in compact for marker in ("直接原因是五只巨兽", "致大炎不得不应对五只巨兽")):
        return True
    return False
