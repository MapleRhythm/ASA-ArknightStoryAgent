from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.patterns import INTERNAL_EVIDENCE_META_RE, MAIN_CHAPTER_REF_RE


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def strip_internal_evidence_meta(text: str) -> str:
    return INTERNAL_EVIDENCE_META_RE.sub("", text or "")


def truncate_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def parse_chapter_number(value: str) -> int | None:
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
        number = parse_chapter_number(raw_number)
        if number is not None:
            numbers.append(number)
    return list(dict.fromkeys(numbers))


def build_main_chapter_retrieval_terms(text: str) -> list[str]:
    terms: list[str] = []
    for number in extract_main_chapter_numbers(text):
        padded = f"{number:02d}"
        terms.extend(
            [
                f"第{number}章",
                f"{number}章",
                f"main_{padded}",
                f"level_main_{padded}",
                f"main_{number}",
                f"level_main_{number}",
                f"EPISODE {number}",
            ]
        )
    return dedupe_keep_order(terms)


def expand_queries_with_main_chapter_terms(queries: list[str]) -> list[str]:
    expanded: list[str] = []
    for query in queries:
        if not query:
            continue
        expanded.append(query)
        chapter_terms = build_main_chapter_retrieval_terms(query)
        if chapter_terms and "章节限定:" not in query:
            expanded.append(query + "\n章节限定: " + " ".join(chapter_terms))
    return dedupe_keep_order(expanded)
