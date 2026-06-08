from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.lexicon import WEB_CONTEXT_QUERY_ANCHOR_TERMS
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, truncate_text


def select_web_context_lines(text: str, *, story_name: str, question: str, max_chars: int) -> str:
    if not text:
        return ""
    compact_head = re.sub(r"\s+", "", text[:1200]).lower()
    if (
        compact_head.count("{") >= 6
        and any(marker in compact_head for marker in ("font-family", "--bing", "rgba(", "@media", "display:"))
    ):
        return ""
    compact_text = re.sub(r"\s+", "", text)
    question_terms = extract_content_tokens(question)
    if "岁陵" in question and "危机" in question:
        question_terms = dedupe_keep_order(
            question_terms + ["岁陵", "危机", "岁陵危机", "岁兽之患", "岁兽", "苏醒", "平息", "望", "不反", "真龙"]
        )
    for anchor in WEB_CONTEXT_QUERY_ANCHOR_TERMS:
        if anchor in question:
            question_terms.append(anchor)
    question_terms = dedupe_keep_order(question_terms)
    if story_name and story_name not in compact_text and not any(term and term in compact_text for term in question_terms):
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if re.sub(r"\s+", " ", line).strip()]
    focus_terms = (story_name, *question_terms, "明日方舟", "剧情", "时间线", "解析", "故事集", "活动", "事件", "主线", "时间")
    selected: list[str] = []
    for line in lines:
        if len(line) < 12:
            continue
        if any(term and term in line for term in focus_terms):
            selected.append(line)
        if sum(len(item) for item in selected) >= max_chars:
            break
    if not selected:
        selected = lines[:20]
    return truncate_text("\n".join(dedupe_keep_order(selected)), max_chars)
