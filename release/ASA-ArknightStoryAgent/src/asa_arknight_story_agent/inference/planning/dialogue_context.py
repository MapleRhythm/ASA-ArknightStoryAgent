from __future__ import annotations

from asa_arknight_story_agent.inference.common.patterns import DIALOGUE_ROLE_PREFIX_RE
from asa_arknight_story_agent.inference.pipeline.constants import PRONOUN_REFERENCES, ROLE_LABEL_MAP
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def parse_dialogue_context(dialogue_context: str) -> list[tuple[str | None, str]]:
    entries: list[tuple[str | None, str]] = []
    for raw_line in dialogue_context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        role_match = DIALOGUE_ROLE_PREFIX_RE.match(line)
        if role_match:
            role = role_match.group(1).lower()
            content = role_match.group(2).strip()
            if content:
                entries.append((role, content))
            continue
        entries.append((None, line))
    return entries


def sanitize_dialogue_context(dialogue_context: str, *, for_prompt: bool = False) -> str:
    rendered_lines: list[str] = []
    for role, content in parse_dialogue_context(dialogue_context):
        if not content:
            continue
        if for_prompt and role in ROLE_LABEL_MAP:
            rendered_lines.append(f"{ROLE_LABEL_MAP[role]}: {content}")
        else:
            rendered_lines.append(content)
    return "\n".join(rendered_lines).strip()


def render_dialogue_context_for_prompt(dialogue_context: str) -> str:
    normalized = sanitize_dialogue_context(dialogue_context, for_prompt=True)
    if not normalized:
        return "无"
    return normalized


def extract_context_entities(dialogue_context: str) -> list[str]:
    from asa_arknight_story_agent.inference.planning.query_tokens import extract_content_tokens, is_entity_candidate

    parsed_entries = parse_dialogue_context(dialogue_context)
    prioritized_texts = [content for role, content in parsed_entries if role == "user"]
    prioritized_texts.extend(content for role, content in parsed_entries if role == "assistant")
    prioritized_texts.extend(content for role, content in parsed_entries if role is None)

    entities: list[str] = []
    for content in prioritized_texts[-4:]:
        entities.extend(token for token in extract_content_tokens(content) if is_entity_candidate(token))
    return dedupe_keep_order(entities)[:6]


def resolve_referential_question(question: str, entities: list[str]) -> str:
    normalized_question = question.strip()
    if not normalized_question or not entities:
        return normalized_question
    anchor = "和".join(entities[:2]) if len(entities) >= 2 else entities[0]
    resolved = normalized_question
    for pronoun in sorted(PRONOUN_REFERENCES, key=len, reverse=True):
        if pronoun in resolved:
            resolved = resolved.replace(pronoun, anchor, 1)
            break
    return resolved
