from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def repair_json_like_output(text: str) -> str:
    candidate = text.lstrip()
    if not candidate:
        return text
    if candidate.startswith('"'):
        return "{" + candidate
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*"\s*:', candidate):
        return '{"' + candidate
    if re.match(r'^"[A-Za-z_][A-Za-z0-9_]*"\s*:', candidate):
        return "{" + candidate
    return text


def repair_common_json_syntax(text: str) -> str:
    repaired = text.strip()
    # Common 4B error: {("question": ...} with a stray parenthesis after the
    # opening brace.
    repaired = re.sub(r"^\{\s*\(", "{", repaired)
    # Common 4B errors: missing colon after list/object fields.
    repaired = re.sub(
        r'([{\[,]\s*)missing_slots\s*\[',
        r'\1"missing_slots":[',
        repaired,
    )
    repaired = re.sub(
        r'([{\[,]\s*)follow_up_hypothesis\s*\{',
        r'\1"follow_up_hypothesis":{',
        repaired,
    )
    # Common 4B error: {question": "..."} where the opening quote is missing.
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:',
        r'\1"\2":',
        repaired,
    )
    # Common 4B error: {next_action:"answer_directly"} with unquoted object keys.
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        repaired,
    )

    def quote_bare_value(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2).strip()
        if value in {"null", "true", "false"} or re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            return prefix + value
        return prefix + json.dumps(value, ensure_ascii=False)

    repaired = re.sub(
        r'("(?:next_action|query_type|intent|expected_answer_type)"\s*:\s*)'
        r'([A-Za-z_][A-Za-z0-9_\-/]*)\b(?=\s*[,}])',
        quote_bare_value,
        repaired,
    )

    def quote_bare_list_item(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2).strip()
        if (
            not value
            or value in {"null", "true", "false"}
            or re.fullmatch(r"-?\d+(?:\.\d+)?", value)
        ):
            return prefix + value
        return prefix + json.dumps(value, ensure_ascii=False)

    repaired = re.sub(
        r'([,\[]\s*)(?!["{\[\]])([^,\]\{\}\n\r:]{1,120})(?=\s*[,]])',
        quote_bare_list_item,
        repaired,
    )
    # Common 4B error: ["a",b", "c"] where the opening quote after comma is missing.
    repaired = re.sub(
        r'([,\[]\s*)([\u4e00-\u9fffA-Za-z_][^"\[\]\{\}:,\n\r]*?)"\s*(?=[,\]])',
        r'\1"\2"',
        repaired,
    )
    # Same issue for object values: "key":value", followed by comma or closing brace.
    repaired = re.sub(
        r'(:\s*)([\u4e00-\u9fffA-Za-z_][^"\[\]\{\}:,\n\r]*?)"\s*(?=[,\}])',
        r'\1"\2"',
        repaired,
    )
    # Missing value for optional nullable fields is safer as null than invalid JSON.
    repaired = re.sub(r'(:\s*)(?=[,\}])', r'\1null', repaired)
    return repaired


def extract_json_object(text: str) -> dict[str, Any] | None:
    fenced_match = JSON_BLOCK_RE.search(text)
    candidate = fenced_match.group(1) if fenced_match else text.strip()
    for candidate_variant in (candidate, repair_common_json_syntax(candidate)):
        try:
            parsed = json.loads(candidate_variant)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                object_candidate = candidate[start : index + 1]
                for candidate_variant in (object_candidate, repair_common_json_syntax(object_candidate)):
                    try:
                        parsed = json.loads(candidate_variant)
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        pass
                return None

    # Tolerate truncated JSON that is otherwise structurally valid except for
    # missing closing braces at the end of generation.
    if depth > 0 and not in_string:
        repaired_candidate = candidate[start:] + ("}" * depth)
        for candidate_variant in (repaired_candidate, repair_common_json_syntax(repaired_candidate)):
            try:
                parsed = json.loads(candidate_variant)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        return None
    return None
