from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.terms import extract_question_anchor_terms
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import expand_related_retrieval_terms
from asa_arknight_story_agent.inference.common.text_utils import (
    dedupe_keep_order,
    strip_internal_evidence_meta,
    truncate_text,
)


def build_minirag_expansion_queries(
    question: str,
    hypothesis: HypothesisDocument,
    minirag_hits: list[dict[str, Any]],
    *,
    chapter_scope_label: str,
    top_k: int = 8,
) -> list[str]:
    anchors = extract_question_anchor_terms(question, hypothesis)[:12]
    metadata_terms: list[str] = []
    snippets: list[str] = []
    for item in minirag_hits[:top_k]:
        doc = item.get("document") or {}
        if not isinstance(doc, dict):
            continue
        for key in ("activity_name", "story_name", "stage_code", "stage_name", "zone_name"):
            value = str(doc.get(key) or "").strip()
            if value:
                metadata_terms.append(value)
        text = strip_internal_evidence_meta(
            str(doc.get("clean_text") or doc.get("search_text") or "")
        ).strip()
        if text:
            snippets.append(truncate_text(text, 260))

    related_terms = expand_related_retrieval_terms(anchors)
    compact_terms = dedupe_keep_order([*anchors, *related_terms, *metadata_terms])[:32]
    evidence_blob = "\n".join(snippets[:top_k])
    queries = [
        "\n".join(
            [
                question,
                f"章节限定: {chapter_scope_label}",
                "关系图扩展线索: " + " ".join(compact_terms),
                "关系图扩展证据:",
                truncate_text(evidence_blob, 1400),
            ]
        ).strip(),
        " ".join([question, chapter_scope_label, *compact_terms]).strip(),
    ]
    return dedupe_keep_order([query for query in queries if query])[:3]
