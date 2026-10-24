"""Lossless input conversion and strict final-answer extraction; no inference."""
import hashlib
import json
import re


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def messages_from_prompt(prompt):
    pattern = (
        r"\A<\|im_start\|>system\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n\{\Z"
    )
    match = re.fullmatch(pattern, prompt, re.DOTALL)
    if not match:
        raise ValueError("Unexpected input layout; refuse lossy conversion")
    messages = [{"role": "system", "content": match[1]},
                {"role": "user", "content": match[2]}]
    if any("<|im_start|>" in x["content"] for x in messages):
        raise ValueError("Nested ChatML marker")
    return messages


def answer_after_thinking(raw, thinking):
    # Same strict contract as the historical v31 extractor, plus nested-marker check.
    if not thinking:
        if "<think>" in raw or "</think>" in raw:
            raise ValueError("Unexpected reasoning markers in non-thinking output")
        if not raw.strip():
            raise ValueError("Empty final answer")
        return raw.strip(), ""
    if raw.count("</think>") != 1:
        raise ValueError("Missing or ambiguous thinking terminator")
    reasoning, answer = raw.split("</think>", 1)
    if "<think>" in answer or "<think>" in reasoning or not answer.strip():
        raise ValueError("Missing or malformed final answer")
    return answer.strip(), reasoning
