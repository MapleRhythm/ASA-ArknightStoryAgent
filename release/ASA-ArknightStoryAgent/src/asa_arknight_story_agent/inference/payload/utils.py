from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[、,，;；]\s*", value) if re.search(r"[、,，;；]", value) else [value]
    elif isinstance(value, list):
        items = [item for item in value if isinstance(item, (str, int, float))]
    else:
        return []
    return dedupe_keep_order([str(item).strip() for item in items if str(item).strip()])[:limit]


def compact_supported_facts_payload(value: Any, *, max_facts: int = 6, max_refs: int = 2) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        refs: list[dict[str, str]] = []
        raw_refs = item.get("evidence_refs")
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                quote = str(ref.get("quote") or "").strip()
                evidence_id = str(ref.get("evidence_id") or "").strip()
                if quote:
                    quote = quote[:80].rstrip()
                new_ref: dict[str, str] = {}
                if evidence_id:
                    new_ref["evidence_id"] = evidence_id
                if quote:
                    new_ref["quote"] = quote
                if new_ref:
                    refs.append(new_ref)
                if len(refs) >= max_refs:
                    break
        compact.append({"fact": fact, "evidence_refs": refs})
        if len(compact) >= max_facts:
            break
    return compact


def compact_inferred_facts_payload(value: Any, *, max_items: int = 2) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value:
        fact = str(item.get("fact") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        if fact:
            compact.append({"fact": fact})
        if len(compact) >= max_items:
            break
    return compact


def answer_from_structured_facts(
    supported_facts: list[dict[str, Any]],
    inferred_facts: list[dict[str, Any]],
    *,
    max_chars: int = 280,
) -> str:
    facts = [
        str(item.get("fact") or "").strip()
        for item in [*supported_facts, *inferred_facts]
        if isinstance(item, dict) and str(item.get("fact") or "").strip()
    ]
    facts = dedupe_keep_order(facts)[:3]
    answer = "；".join(facts)
    if len(answer) > max_chars:
        answer = answer[: max_chars - 1].rstrip("；，。 ") + "。"
    return answer
