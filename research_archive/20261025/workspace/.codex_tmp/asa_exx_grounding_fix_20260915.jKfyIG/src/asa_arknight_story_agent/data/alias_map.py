"""Operator alias lookup used by hypothesis-stage query expansion.

Loads ``indexes/arknights_story/operator_aliases.json`` (a dict mapping
each canonical / known form to a list of equivalents) and exposes a
readable-name -> alias-list view. Resource-prefix identifiers like
``char_xxx`` / ``avatar_yyy`` are filtered out because they are not
useful for retrieval keywords.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

RESOURCE_PREFIX_PATTERNS = (
    re.compile(r"^(avatar|char|npc|trap|enemy|skin|act|story|stage|building)_", re.IGNORECASE),
)
NON_READABLE_RE = re.compile(r"^[a-z0-9_]+$", re.IGNORECASE)
MIN_ALIAS_LEN = 2


def _is_readable(token: str) -> bool:
    token = token.strip()
    if len(token) < MIN_ALIAS_LEN:
        return False
    for pattern in RESOURCE_PREFIX_PATTERNS:
        if pattern.match(token):
            return False
    if NON_READABLE_RE.match(token) and len(token) <= 4:
        # Short ASCII like "Dr" / "W" — keep only if uppercase initial; lowercase blobs are noise.
        if not token[0].isupper():
            return False
    return True


def _filter_aliases(aliases: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in aliases:
        if not isinstance(raw, str):
            continue
        token = raw.strip()
        if not token or token in seen:
            continue
        if not _is_readable(token):
            continue
        seen.add(token)
        output.append(token)
    return output


class OperatorAliasMap:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping: dict[str, list[str]] = {}
        for key, aliases in mapping.items():
            if not isinstance(key, str):
                continue
            key_stripped = key.strip()
            if not _is_readable(key_stripped):
                continue
            filtered = _filter_aliases(aliases)
            if filtered:
                self._mapping[key_stripped] = filtered

    @classmethod
    def from_path(cls, path: Path) -> "OperatorAliasMap":
        if not path.exists():
            return cls({})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls({})
        if not isinstance(payload, dict):
            return cls({})
        return cls({k: v for k, v in payload.items() if isinstance(v, list)})

    def __bool__(self) -> bool:
        return bool(self._mapping)

    def lookup(self, entity: str) -> list[str]:
        if not entity:
            return []
        return list(self._mapping.get(entity.strip(), []))

    def expand(self, entities: list[str], *, limit_per_entity: int = 6) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set(entity.strip() for entity in entities if entity)
        for entity in entities:
            aliases = self.lookup(entity)
            for alias in aliases[:limit_per_entity]:
                if alias in seen:
                    continue
                seen.add(alias)
                expanded.append(alias)
        return expanded


@lru_cache(maxsize=4)
def load_operator_alias_map(path: Path) -> OperatorAliasMap:
    return OperatorAliasMap.from_path(path)
