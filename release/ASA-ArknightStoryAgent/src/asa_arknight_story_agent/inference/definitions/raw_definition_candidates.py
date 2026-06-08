from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import DATA_ROOT
from asa_arknight_story_agent.inference.definitions.definition_scoring import definition_anchor_score
from asa_arknight_story_agent.inference.definitions.raw_story_cleaning import (
    clean_raw_story_context,
    raw_line_context,
)


def allow_rogue_raw_story_search(question: str) -> bool:
    return any(term in question.lower() for term in ("rogue", "肉鸽", "集成战略", "结局", "萨卡兹的无终奇语"))


def raw_story_relative_path(path: Path) -> Path:
    return path.relative_to(DATA_ROOT.parent.parent.parent.parent) if DATA_ROOT.exists() else path


def build_raw_exact_candidate(
    *,
    path: Path,
    line_index: int,
    context: str,
    anchors: list[str],
    compact_anchors: list[str],
    score: int,
) -> dict[str, Any] | None:
    clean_text = clean_raw_story_context(context)
    if not clean_text:
        return None
    doc_id = f"raw_exact/{path.relative_to(DATA_ROOT).as_posix()}#L{line_index + 1}"
    document = {
        "id": doc_id,
        "activity_name": "ArknightsGameData原文",
        "story_name": path.stem,
        "stage_code": "",
        "avg_tag": "raw_exact",
        "source_path": str(path),
        "clean_text": clean_text,
        "search_text": clean_text,
    }
    return {
        "doc_index": -1,
        "document": document,
        "evidence_chain_score": 100.0 + float(score),
        "fusion_score": 100.0 + float(score),
        "supplemental_source": "raw_exact",
        "raw_exact": {
            "line": line_index + 1,
            "relative_path": str(raw_story_relative_path(path)),
            "anchors": [anchor for anchor in anchors if anchor and anchor in re.sub(r"\s+", "", clean_text)],
            "score": score,
        },
    }


def score_raw_exact_line(context: str, compact_line: str, line: str, compact_anchors: list[str], anchors: list[str]) -> int:
    score = definition_anchor_score(context, anchors)
    if "？" in compact_line or "?" in compact_line or "是什么" in compact_line or "疑问" in compact_line:
        score -= 6
    if "[Sticker" in line or "text=" in line:
        score += 4
    for anchor in compact_anchors[:4]:
        if re.search(
            re.escape(anchor) + r".{0,10}[，,:：/（(].{0,32}(?:系统|产物|机器|设备|本名|全称|身份)",
            compact_line,
        ):
            score += 8
            break
    return score


def raw_exact_candidates_from_file(
    *,
    path: Path,
    anchors: list[str],
    compact_anchors: list[str],
) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    compact_text = re.sub(r"\s+", "", text)
    if not any(anchor and anchor in compact_text for anchor in compact_anchors):
        return None
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        compact_line = re.sub(r"\s+", "", line)
        if not compact_line or not any(anchor and anchor in compact_line for anchor in compact_anchors):
            continue
        context = raw_line_context(lines, line_index, window=2)
        score = score_raw_exact_line(context, compact_line, line, compact_anchors, anchors)
        if score < 7:
            continue
        return build_raw_exact_candidate(
            path=path,
            line_index=line_index,
            context=context,
            anchors=anchors,
            compact_anchors=compact_anchors,
            score=score,
        )
    return None
