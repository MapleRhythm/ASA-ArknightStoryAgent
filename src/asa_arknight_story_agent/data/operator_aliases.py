from __future__ import annotations

import json
import re
from pathlib import Path

from asa_arknight_story_agent.config import MANUAL_OPERATOR_ALIAS_SOURCE_PATH
from asa_arknight_story_agent.data.story_text import normalize_text


ALIAS_NAME_PATTERN = r"[A-Za-z\u4e00-\u9fff·\.\-]{1,32}"
PROFILE_CODENAME_RE = re.compile(rf"^【代号】\s*({ALIAS_NAME_PATTERN})$", re.MULTILINE)
REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:本名|真名)(?:为|是|叫|：|:)?\s*({ALIAS_NAME_PATTERN})(?:[，。；\n]|$)"
)
CODENAME_REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:干员)?({ALIAS_NAME_PATTERN})，本名({ALIAS_NAME_PATTERN})(?:[，。；\n]|$)"
)
REAL_NAME_OPERATOR_CODENAME_RE = re.compile(
    rf"(?:^|[。；\n])({ALIAS_NAME_PATTERN})，[\s\S]{{0,200}}?以干员[\"“”']+({ALIAS_NAME_PATTERN})[\"“”']+身份"
)
SELF_INTRO_REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:您好|你好)?[，,]?(?:我叫|我是)\s*({ALIAS_NAME_PATTERN})(?:[，,。]|$)"
)
SELF_INTRO_CODENAME_RE = re.compile(
    rf"(?:代号[^。\n]{{0,40}}?)?(?:您叫我|您可以叫我|就叫我|叫我)"
    rf"[，,:：\s\"“”']*({ALIAS_NAME_PATTERN})[\"“”']*(?:就好|吧|即可|好了|。|，|$)"
)
OPERATOR_IDENTITY_CODENAME_RE = re.compile(
    rf"以干员[\"“”']+({ALIAS_NAME_PATTERN})[\"“”']+身份"
)


def clean_alias_name(candidate: str, *, min_len: int = 2) -> str:
    cleaned = normalize_text(candidate).strip("，。；：:、,.;!?？！()（）[]【】<>《》\"' ")
    if len(cleaned) < min_len:
        return ""
    cleaned = cleaned.removesuffix("就好").removesuffix("好了").removesuffix("即可")
    cleaned = cleaned.removesuffix("吧").removesuffix("啊").removesuffix("呀").removesuffix("吗").removesuffix("呢").removesuffix("了")
    if len(cleaned) > 12:
        return ""
    if any(token in cleaned for token in ("干员", "本名", "真名", "代号", "小姐", "先生")):
        return ""
    if "的" in cleaned:
        return ""
    if any(
        token in cleaned
        for token in (
            "代表我",
            "开始吧",
            "可行不通",
            "怎么样",
            "的话",
            "不用担心",
            "多多指教",
            "什么",
        )
    ):
        return ""
    return cleaned


def extract_real_name_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in CODENAME_REAL_NAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(2))
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_OPERATOR_CODENAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    for match in SELF_INTRO_REAL_NAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def extract_codename_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in PROFILE_CODENAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in CODENAME_REAL_NAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_OPERATOR_CODENAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(2), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in SELF_INTRO_CODENAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in OPERATOR_IDENTITY_CODENAME_RE.finditer(text):
        cleaned = clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def build_operator_alias_lookup(excel_root: Path) -> dict[str, list[str]]:
    cache_key = f"{excel_root.resolve().as_posix()}::{MANUAL_OPERATOR_ALIAS_SOURCE_PATH.resolve().as_posix()}"
    cached = getattr(build_operator_alias_lookup, "_cache", {})
    if cache_key in cached:
        return cached[cache_key]

    raw_groups = json.loads(MANUAL_OPERATOR_ALIAS_SOURCE_PATH.read_text(encoding="utf-8"))
    alias_lookup: dict[str, set[str]] = {}
    for primary_alias, related_aliases in raw_groups.items():
        if not isinstance(related_aliases, list):
            continue
        group = [primary_alias, *related_aliases]
        normalized_group: list[str] = []
        for candidate in group:
            cleaned = normalize_text(str(candidate or ""))
            if cleaned and cleaned not in normalized_group:
                normalized_group.append(cleaned)
        if len(normalized_group) <= 1:
            continue
        for alias in normalized_group:
            related = alias_lookup.setdefault(alias, set())
            related.update(item for item in normalized_group if item != alias)

    normalized_lookup = {alias: sorted(related) for alias, related in sorted(alias_lookup.items()) if related}
    cached[cache_key] = normalized_lookup
    build_operator_alias_lookup._cache = cached  # type: ignore[attr-defined]
    return normalized_lookup


def collect_related_aliases(text: str, alias_lookup: dict[str, list[str]]) -> list[str]:
    if not text or not alias_lookup:
        return []
    related_terms: list[str] = []
    for alias, related in alias_lookup.items():
        if alias in text:
            related_terms.extend(item for item in related if item not in text)
    return list(dict.fromkeys(related_terms))
