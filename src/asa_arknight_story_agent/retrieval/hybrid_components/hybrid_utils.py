from __future__ import annotations

import json
from pathlib import Path

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import ASCII_TOKEN_RE, MAIN_CHAPTER_REF_RE


def tokenize_for_bm25(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = ASCII_TOKEN_RE.findall(lowered)
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [f"{cjk_chars[i]}{cjk_chars[i + 1]}" for i in range(len(cjk_chars) - 1)]
    return ascii_tokens + cjk_bigrams


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_main_chapter_number(value: str) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        number = int(raw)
        return number if 0 < number < 100 else None
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        number = tens * 10 + ones
        return number if 0 < number < 100 else None
    if len(raw) == 1 and raw in digits:
        return digits[raw]
    return None


def extract_main_chapter_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in MAIN_CHAPTER_REF_RE.finditer(text or ""):
        raw_number = next((group for group in match.groups() if group), "")
        number = parse_main_chapter_number(raw_number)
        if number is not None:
            numbers.append(number)
    return list(dict.fromkeys(numbers))
