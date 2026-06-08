from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.terms import extract_question_anchor_terms
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def render_minirag_hints_for_prompt(evidence: list[dict[str, Any]], hypothesis: HypothesisDocument) -> str:
    entities: list[str] = []
    relations: list[str] = []
    neighbors: list[str] = []
    for item in evidence[:12]:
        doc = item.get("document") or {}
        for value in (
            doc.get("activity_name"),
            doc.get("story_name"),
            doc.get("stage_code"),
            doc.get("stage_name"),
            doc.get("zone_name"),
        ):
            text = str(value or "").strip()
            if text:
                entities.append(text)
        for role in item.get("evidence_chain_roles") or []:
            text = str(role or "").strip()
            if text:
                neighbors.append(text)
        chain_text = str(item.get("evidence_chain_text") or "").strip()
        if chain_text:
            for token in extract_question_anchor_terms(chain_text, hypothesis)[:4]:
                entities.append(token)
    entities = dedupe_keep_order(hypothesis.entities + entities)[:12]
    keywords = dedupe_keep_order(hypothesis.keywords + neighbors)[:12]
    if len(entities) >= 2:
        relations = [f"{entities[index]}-相关-{entities[index + 1]}" for index in range(min(len(entities) - 1, 6))]
    return (
        "entities="
        + ",".join(entities)
        + " | relations="
        + ";".join(relations)
        + " | neighbors="
        + ",".join(keywords)
    )
