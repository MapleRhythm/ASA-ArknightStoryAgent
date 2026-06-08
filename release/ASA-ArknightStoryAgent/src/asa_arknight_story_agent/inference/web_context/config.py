from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.inference.common.lexicon import (
    WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES,
    WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES,
)


@dataclass(slots=True)
class WebContextConfig:
    enabled: bool = False
    cache_dir: Path | None = None
    cache_ttl_seconds: int = 604800
    timeout_seconds: float = 6.0
    max_elapsed_seconds: float = 18.0
    max_first_round_evidence: int = 24
    min_story_hits: int = 2
    max_search_queries: int = 2
    max_search_results: int = 6
    max_pages: int = 3
    max_chars_per_page: int = 2200
    max_total_chars: int = 6000
    rerank_top_k: int = 2
    rerank_min_score: float = 1.0
    require_story_or_question_hit: bool = True
    force_prompt_evidence: bool = False
    query_templates: tuple[str, ...] = WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES
    search_url_templates: tuple[str, ...] = WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES
    user_agent: str = "Mozilla/5.0 ASA-ArknightStoryAgent/1.0"


def _normalize_string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return default
    normalized = tuple(str(item).strip() for item in value if str(item).strip())
    return normalized or default


def build_web_context_config(payload: dict[str, Any] | None) -> WebContextConfig:
    cfg = payload if isinstance(payload, dict) else {}
    cache_dir_value = cfg.get("cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value else None
    return WebContextConfig(
        enabled=bool(cfg.get("enabled", False)),
        cache_dir=cache_dir,
        cache_ttl_seconds=max(0, int(cfg.get("cache_ttl_seconds", 604800))),
        timeout_seconds=max(1.0, float(cfg.get("timeout_seconds", 6.0))),
        max_elapsed_seconds=max(3.0, float(cfg.get("max_elapsed_seconds", 18.0))),
        max_first_round_evidence=max(1, int(cfg.get("max_first_round_evidence", 24))),
        min_story_hits=max(1, int(cfg.get("min_story_hits", 2))),
        max_search_queries=max(1, int(cfg.get("max_search_queries", 2))),
        max_search_results=max(1, int(cfg.get("max_search_results", 6))),
        max_pages=max(1, int(cfg.get("max_pages", 3))),
        max_chars_per_page=max(300, int(cfg.get("max_chars_per_page", 2200))),
        max_total_chars=max(800, int(cfg.get("max_total_chars", 6000))),
        rerank_top_k=max(0, int(cfg.get("rerank_top_k", 2))),
        rerank_min_score=float(cfg.get("rerank_min_score", 1.0)),
        require_story_or_question_hit=bool(cfg.get("require_story_or_question_hit", True)),
        force_prompt_evidence=bool(cfg.get("force_prompt_evidence", False)),
        query_templates=_normalize_string_tuple(cfg.get("query_templates"), WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES),
        search_url_templates=_normalize_string_tuple(
            cfg.get("search_url_templates"),
            WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES,
        ),
        user_agent=str(cfg.get("user_agent") or "Mozilla/5.0 ASA-ArknightStoryAgent/1.0"),
    )
