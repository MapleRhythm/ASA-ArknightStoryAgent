from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ENTITY_RUN_RE = re.compile(r"[\u4e00-\u9fff·]{2,32}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
SPEAKER_PREFIX_RE = re.compile(r"(?m)^([\u4e00-\u9fff·A-Za-z0-9_.\-]{2,16})[：:]")
TITLE_RE = re.compile(r"[《「“]([^》」”]{2,24})[》」”]")
GENERIC_ENTITY_STOP_WORDS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "如何",
        "这里",
        "那里",
        "这个",
        "那个",
        "这些",
        "那些",
        "自己",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "博士",
        "干员",
        "作战",
        "行动",
        "当前",
        "证据",
        "剧情",
        "事情",
        "一个",
        "一些",
        "不是",
        "没有",
        "已经",
        "因为",
        "所以",
        "但是",
        "然后",
        "如果",
        "只是",
        "可以",
        "知道",
        "觉得",
        "还是",
        "不会",
        "不能",
        "必须",
        "具体",
        "一事",
        "指什么",
        "是什么",
        "为什么",
        "怎么样",
        "怎么办",
        "哪里",
        "哪位",
    }
)
METADATA_ENTITY_PREFIXES = ("activity:", "story:", "stage:")
RELATION_GATE_STOP_WORDS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "如何",
        "是谁",
        "哪里",
        "哪位",
        "这个",
        "那个",
        "这些",
        "那些",
        "事情",
        "关系",
        "原因",
        "目的",
        "动机",
        "真相",
        "秘密",
    }
)
RELATION_GATE_KEYWORDS = frozenset(
    {
        "父亲",
        "母亲",
        "儿子",
        "女儿",
        "老师",
        "学生",
        "同伴",
        "朋友",
        "敌人",
        "上司",
        "下属",
        "领袖",
        "成员",
        "属于",
        "来自",
        "控制",
        "背叛",
        "保护",
        "杀死",
        "刺杀",
        "攻击",
        "阻止",
        "帮助",
        "合作",
        "交易",
        "约定",
        "计划",
        "导致",
        "引发",
        "揭露",
        "识破",
        "隐瞒",
        "伪装",
        "身份",
        "真身",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_alias_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): [str(item) for item in value if str(item).strip()]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def build_alias_lookup(alias_map: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        names = [canonical, *aliases]
        for name in names:
            normalized = name.strip()
            if len(normalized) >= 2:
                lookup[normalized] = canonical
    return lookup


@lru_cache(maxsize=16)
def _compile_alias_regex(alias_items: tuple[tuple[str, str], ...]) -> re.Pattern[str] | None:
    aliases = [alias for alias, _canonical in alias_items]
    pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    return re.compile(pattern) if pattern else None


@lru_cache(maxsize=16)
def _compile_alias_automaton(alias_items: tuple[tuple[str, str], ...]) -> Any:
    try:
        import ahocorasick  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    automaton = ahocorasick.Automaton()
    for alias, canonical in alias_items:
        automaton.add_word(alias, (alias, canonical))
    automaton.make_automaton()
    return automaton


def extract_alias_entities(text: str, alias_lookup: dict[str, str]) -> list[str]:
    """Exact alias matching using cached Aho-Corasick or regex fallback."""
    if not alias_lookup:
        return []
    alias_items = tuple(sorted(alias_lookup.items()))
    automaton = _compile_alias_automaton(alias_items)
    if automaton is None:
        regex = _compile_alias_regex(alias_items)
        if regex is None:
            return []
        found: list[str] = []
        seen: set[str] = set()
        for match in regex.finditer(text):
            canonical = alias_lookup.get(match.group(0))
            if canonical and canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
        return found

    found: list[str] = []
    seen: set[str] = set()
    for _end, (_alias, canonical) in automaton.iter(text):
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)
    return found


def metadata_entities(document: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    for key, prefix in (
        ("activity_name", "activity"),
        ("story_name", "story"),
        ("stage_code", "stage"),
        ("stage_name", "stage"),
    ):
        value = str(document.get(key) or "").strip()
        if value:
            entities.append(f"{prefix}:{value}")
    return entities


def relation_gate_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in ENTITY_RUN_RE.finditer(text):
        token = match.group(0).strip()
        if len(token) < 2 or token in RELATION_GATE_STOP_WORDS:
            continue
        if any(stop in token for stop in RELATION_GATE_STOP_WORDS):
            continue
        terms.add(token)
        if "\u4e00" <= token[0] <= "\u9fff" and len(token) >= 4:
            for size in (2, 3, 4):
                for index in range(0, len(token) - size + 1):
                    gram = token[index : index + size]
                    if gram not in RELATION_GATE_STOP_WORDS:
                        terms.add(gram)
    for keyword in RELATION_GATE_KEYWORDS:
        if keyword in text:
            terms.add(keyword)
    return terms


def is_generic_entity_candidate(token: str) -> bool:
    token = token.strip()
    if len(token) < 2 or len(token) > 16:
        return False
    if token in GENERIC_ENTITY_STOP_WORDS:
        return False
    if any(marker in token for marker in ("什么", "怎么", "为何", "为什么", "如何")):
        return False
    if token.endswith(("的", "了", "吗", "呢", "吧", "啊", "着", "过")):
        return False
    return True


def extract_generic_text_entities(text: str, *, limit: int = 48) -> list[str]:
    """Extract lightweight text entities without an LLM."""
    candidates: dict[str, tuple[float, int]] = {}

    def add(raw: str, *, boost: float = 0.0, pos: int = 10**9) -> None:
        token = raw.strip()
        if not is_generic_entity_candidate(token):
            return
        score = float(len(token)) + boost
        old = candidates.get(token)
        if old is None:
            candidates[token] = (score, pos)
        else:
            candidates[token] = (old[0] + score, min(old[1], pos))

    for pattern in (SPEAKER_PREFIX_RE, TITLE_RE):
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            add(value, boost=6.0, pos=match.start())

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1][0], item[1][1], -len(item[0]), item[0]),
    )
    return [token for token, _ in ranked[:limit]]
