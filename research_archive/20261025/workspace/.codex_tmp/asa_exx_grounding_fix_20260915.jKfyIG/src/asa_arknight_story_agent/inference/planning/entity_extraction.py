from __future__ import annotations

from asa_arknight_story_agent.config import OPERATOR_ALIAS_MAP_PATH
from asa_arknight_story_agent.data.alias_map import load_operator_alias_map
from asa_arknight_story_agent.inference.planning.dialogue_context import extract_context_entities
from asa_arknight_story_agent.inference.pipeline.constants import PRONOUN_REFERENCES
from asa_arknight_story_agent.inference.planning.query_tokens import extract_content_tokens, is_entity_candidate
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def extract_entities(question: str, dialogue_context: str = "") -> list[str]:
    question_entities = [token for token in extract_content_tokens(question) if is_entity_candidate(token)]
    if any(pronoun in question for pronoun in PRONOUN_REFERENCES):
        return dedupe_keep_order(extract_context_entities(dialogue_context) + question_entities)[:12]
    return dedupe_keep_order(question_entities)[:12]


def expand_entities_with_aliases(entities: list[str], existing_keywords: list[str]) -> list[str]:
    alias_map = load_operator_alias_map(OPERATOR_ALIAS_MAP_PATH)
    if not alias_map:
        return []
    return [alias for alias in alias_map.expand(entities) if alias not in existing_keywords]
