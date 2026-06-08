from __future__ import annotations

from asa_arknight_story_agent.inference.planning.dialogue_context import resolve_referential_question
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import build_main_chapter_retrieval_terms


def build_retrieval_query(hypothesis: HypothesisDocument) -> str:
    resolved_question = resolve_referential_question(hypothesis.question, hypothesis.entities)
    lines = [resolved_question]
    if hypothesis.entities:
        lines.append("实体: " + " ".join(hypothesis.entities))
    if hypothesis.keywords:
        lines.append("关键词: " + " ".join(hypothesis.keywords[:10]))
    chapter_terms = build_main_chapter_retrieval_terms(
        "\n".join([hypothesis.question, " ".join(hypothesis.entities), " ".join(hypothesis.keywords)])
    )
    if chapter_terms:
        lines.append("章节限定: " + " ".join(chapter_terms))
    if hypothesis.expected_answer_type:
        lines.append(f"回答类型: {hypothesis.expected_answer_type}")
    return "\n".join(lines)
