from __future__ import annotations

from asa_arknight_story_agent.inference.planning.dialogue_context import extract_context_entities
from asa_arknight_story_agent.inference.planning.entity_extraction import expand_entities_with_aliases
from asa_arknight_story_agent.inference.common.lexicon import STORY_HINT_WORDS
from asa_arknight_story_agent.inference.pipeline.constants import PRONOUN_REFERENCES
from asa_arknight_story_agent.inference.planning.query_tokens import (
    expand_related_retrieval_terms,
    extract_content_tokens,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


CONCEPT_REASONING_KEYWORDS = [
    "本质",
    "原本",
    "一体",
    "苏醒",
    "消灭",
    "代价",
    "动乱",
    "灭顶之灾",
    "开战",
    "平息",
    "解决",
]

STORY_KEYWORDS = ["共同经历", "相遇", "同行", "冲突", "过往"]


def build_hypothesis_keywords(
    *,
    question: str,
    dialogue_context: str,
    entities: list[str],
    answer_type: str,
) -> list[str]:
    question_keywords = extract_content_tokens(question)
    context_entities = extract_context_entities(dialogue_context) if any(
        pronoun in question for pronoun in PRONOUN_REFERENCES
    ) else []

    keywords = dedupe_keep_order(
        context_entities
        + entities
        + question_keywords
    )[:16]
    related_keywords = expand_related_retrieval_terms(entities + question_keywords)
    if related_keywords:
        keywords = dedupe_keep_order(keywords + related_keywords)[:24]

    if any(token in question for token in STORY_HINT_WORDS):
        keywords = dedupe_keep_order(keywords + STORY_KEYWORDS)[:20]
    if answer_type == "概念定义/危机原因":
        keywords = dedupe_keep_order(keywords + CONCEPT_REASONING_KEYWORDS)[:24]
    alias_keywords = expand_entities_with_aliases(entities, keywords)
    if alias_keywords:
        keywords = dedupe_keep_order(keywords + alias_keywords)[:24]
    return keywords
