from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import (
    best_prompt_text,
    document_chain_text,
)
from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta, truncate_text


def render_evidence_blocks(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    blocks = []
    total_chars = 0
    seen_chain_texts: set[str] = set()
    seen_doc_ids: set[str] = set()
    for item in evidence:
        doc = item["document"]
        clean_text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        chain_text = document_chain_text(item)
        if bool(item.get("prompt_prefer_clean_text")):
            chain_text = "" if clean_text else chain_text
        if chain_text:
            if chain_text in seen_chain_texts:
                continue
            seen_chain_texts.add(chain_text)
            clean_text = chain_text
        else:
            doc_id = str(doc.get("id") or "")
            if doc_id and doc_id in seen_doc_ids:
                continue
            if doc_id:
                seen_doc_ids.add(doc_id)
            clean_text = strip_internal_evidence_meta(str(doc["clean_text"]))
        if max_chars_per_doc is not None:
            clean_text = truncate_text(clean_text, max_chars_per_doc)
        block = [
            f"[E{len(blocks) + 1}]",
            f"id: {doc['id']}",
            f"activity_name: {doc.get('activity_name') or ''}",
            f"story_name: {doc.get('story_name') or ''}",
            f"stage_code: {doc.get('stage_code') or ''}",
            f"avg_tag: {doc.get('avg_tag') or ''}",
            f"source_path: {doc.get('source_path') or ''}",
            f"chain_roles: {','.join(item.get('evidence_chain_roles') or [])}",
            "clean_text:",
            clean_text,
        ]
        rendered_block = "\n".join(block)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def render_short_evidence_brief(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int = 260,
    max_total_chars: int = 2200,
    preserve_complete_evidence: bool = False,
    label_on_own_line: bool = False,
) -> str:
    lines: list[str] = []
    total_chars = 0
    seen: set[str] = set()
    for item in evidence:
        doc = item.get("document") or {}
        doc_id = str(doc.get("id") or item.get("doc_index") or "").strip()
        text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        text = re.sub(r"\s+", " ", strip_internal_evidence_meta(text)).strip()
        if not text:
            continue
        key = doc_id or text[:160]
        if key in seen:
            continue
        seen.add(key)
        if not preserve_complete_evidence:
            text = truncate_text(text, max_chars_per_doc)
        # 用 [E编号] 作为证据的稳定可引用编号(替代长 doc_id), 供 grounding 校验定位
        line = (
            f"[E{len(lines) + 1}]\n{text}"
            if label_on_own_line
            else f"[E{len(lines) + 1}] {text}"
        )
        # In Exx mode evidence is atomic: never cut a passage in the middle.
        # When the remaining context budget is insufficient, drop the whole
        # passage. This also applies to rank 1: returning no evidence is safer
        # than silently exceeding the model context window.
        if max_total_chars > 0 and total_chars + len(line) > max_total_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)


def evidence_id_text_map(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int | None = None,
    max_total_chars: int | None = None,
) -> dict[str, str]:
    """构建与 render_short_evidence_brief 一致的 [E编号] -> 证据文本 映射。
    校验时用它把 evidence_id 定位到具体证据文本。"""
    mapping: dict[str, str] = {}
    total_chars = 0
    seen: set[str] = set()
    for item in evidence:
        doc = item.get("document") or {}
        doc_id = str(doc.get("id") or item.get("doc_index") or "").strip()
        text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        text = re.sub(r"\s+", " ", strip_internal_evidence_meta(text)).strip()
        if not text:
            continue
        key = doc_id or text[:160]
        if key in seen:
            continue
        seen.add(key)
        if max_chars_per_doc is not None:
            text = truncate_text(text, max_chars_per_doc)
        line_chars = len(f"[E{len(mapping) + 1}] {text}")
        if max_total_chars is not None and mapping and total_chars + line_chars > max_total_chars:
            break
        mapping[f"E{len(mapping) + 1}"] = text
        total_chars += line_chars
    return mapping


def evidence_id_text_map_from_prompt(evidence_prompt_text: str | None) -> dict[str, str]:
    """Parse the exact ``[E#]`` blocks that were visible to the generator.

    Validation must not use an untruncated document when the prompt only
    contained a shortened excerpt.  This parser supports both the one-line
    minimal evidence brief and the multi-line full evidence blocks.
    """
    if not evidence_prompt_text:
        return {}
    matches = list(re.finditer(r"(?m)^\[(E\d+)\][ \t]*", evidence_prompt_text))
    mapping: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence_prompt_text)
        text = evidence_prompt_text[match.end() : end].strip()
        if text:
            mapping[match.group(1)] = text
    return mapping


def summarize_evidence_for_trace(
    evidence: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for item in evidence[:limit]:
        doc = item["document"]
        snippet = re.sub(r"\s+", " ", doc["clean_text"]).strip()[:80]
        summary.append(
            {
                "id": str(doc["id"]),
                "activity_name": str(doc.get("activity_name") or ""),
                "story_name": str(doc.get("story_name") or ""),
                "stage_code": str(doc.get("stage_code") or ""),
                "snippet": snippet,
            }
        )
    return summary
