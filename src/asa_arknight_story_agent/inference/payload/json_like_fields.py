from __future__ import annotations

import json
import re
from typing import Any

from asa_arknight_story_agent.inference.payload.utils import normalize_string_list
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def extract_json_like_bare_field(text: str, field: str) -> str:
    match = re.search(
        rf'"?{re.escape(field)}"?\s*:\s*"?([A-Za-z_][A-Za-z0-9_\-/]*)"?',
        text,
    )
    return match.group(1).strip() if match else ""


def extract_json_like_string_field(text: str, field: str) -> str:
    match = re.search(rf'"?{re.escape(field)}"?\s*:\s*"', text)
    if match:
        start = match.end()
        escape = False
        chars: list[str] = []
        for char in text[start:]:
            if escape:
                chars.append("\\" + char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                raw_value = "".join(chars)
                try:
                    parsed = json.loads(f'"{raw_value}"')
                    return parsed if isinstance(parsed, str) else str(parsed)
                except json.JSONDecodeError:
                    return raw_value.replace(r"\"", '"').replace(r"\\", "\\").strip()
            chars.append(char)
    bare_match = re.search(
        rf'"?{re.escape(field)}"?\s*:\s*([^,\}}\n\r]+)',
        text,
    )
    if not bare_match:
        return ""
    value = bare_match.group(1).strip().strip('"')
    return "" if value == "null" else value


def extract_json_like_missing_slots(text: str, *, limit: int = 8) -> list[str]:
    string_value = extract_json_like_string_field(text, "missing_slots")
    if string_value and not string_value.startswith("["):
        return normalize_string_list(string_value, limit=limit)

    match = re.search(r'"?missing_slots"?\s*:?\s*\[', text)
    if not match:
        return []
    start = match.end()
    end = text.find("]", start)
    follow_up_match = re.search(r'"?follow_up_hypothesis"?\s*:?\s*\{', text[start:])
    if end == -1 or (follow_up_match and start + follow_up_match.start() < end):
        end = start + follow_up_match.start() if follow_up_match else min(len(text), start + 500)
    body = text[start:end]

    items: list[str] = []
    for quoted in re.finditer(r'"((?:\\.|[^"\\])*)"', body):
        raw_value = quoted.group(1)
        try:
            value = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            value = raw_value
        if isinstance(value, str) and value.strip():
            items.append(value.strip())

    bare_body = re.sub(r'"(?:\\.|[^"\\])*"', "", body)
    for part in re.split(r"[、,，;；]\s*", bare_body):
        value = part.strip().strip('"').strip()
        value = re.sub(r"^[\[\s]+|[\]\s]+$", "", value).strip()
        if value and value not in {":", "null", "None"}:
            items.append(value)
    return dedupe_keep_order(items)[:limit]


def extract_json_like_repeated_string_field(text: str, field: str, *, limit: int = 4) -> list[str]:
    values: list[str] = []
    pattern = re.compile(rf'"?{re.escape(field)}"?\s*:\s*"')
    for match in pattern.finditer(text):
        start = match.end()
        escape = False
        chars: list[str] = []
        for char in text[start:]:
            if escape:
                chars.append("\\" + char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                raw_value = "".join(chars)
                try:
                    value = json.loads(f'"{raw_value}"')
                except json.JSONDecodeError:
                    value = raw_value.replace(r"\"", '"').replace(r"\\", "\\")
                value = str(value).strip()
                if value:
                    values.append(value)
                break
            chars.append(char)
        if len(values) >= limit:
            break
    return dedupe_keep_order(values)[:limit]


def extract_truncated_supported_facts(text: str, *, limit: int = 2) -> list[dict[str, Any]]:
    facts = extract_json_like_repeated_string_field(text, "fact", limit=limit)
    quotes = extract_json_like_repeated_string_field(text, "quote", limit=limit)
    evidence_ids = extract_json_like_repeated_string_field(text, "evidence_id", limit=limit)
    supported: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        item: dict[str, Any] = {"fact": fact}
        if index < len(quotes):
            ref: dict[str, Any] = {"quote": quotes[index]}
            if index < len(evidence_ids):
                ref["evidence_id"] = evidence_ids[index]
            item["evidence_refs"] = [ref]
        supported.append(item)
    return supported
