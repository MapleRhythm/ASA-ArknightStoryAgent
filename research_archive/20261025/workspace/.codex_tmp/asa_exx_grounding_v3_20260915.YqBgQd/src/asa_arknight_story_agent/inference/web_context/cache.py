from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def cache_key_for_web_context(story_name: str, queries: list[str]) -> str:
    raw = json.dumps(
        {"version": 7, "story_name": story_name, "queries": queries},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def read_web_context_cache(cache_dir: Path | None, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    created_at = float(payload.get("created_at") or 0)
    if ttl_seconds > 0 and created_at and time.time() - created_at > ttl_seconds:
        return None
    return payload if isinstance(payload, dict) else None


def write_web_context_cache(cache_dir: Path | None, cache_key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps({"created_at": time.time(), **payload}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return
