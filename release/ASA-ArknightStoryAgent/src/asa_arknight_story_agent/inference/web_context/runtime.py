from __future__ import annotations

from asa_arknight_story_agent.inference.web_context.cache import (
    cache_key_for_web_context,
    read_web_context_cache,
    write_web_context_cache,
)
from asa_arknight_story_agent.inference.web_context.rendering import (
    build_web_context_text,
    make_web_context_evidence_item,
)
from asa_arknight_story_agent.inference.web_context.search import (
    deadline_expired,
    decode_bing_redirect,
    decode_duckduckgo_redirect,
    extract_search_results,
    http_get_text,
    is_usable_web_url,
    normalize_search_href,
    remaining_timeout,
    resolve_search_redirect_url,
    strip_html_to_text,
    web_search_results,
)

__all__ = [
    "cache_key_for_web_context",
    "read_web_context_cache",
    "write_web_context_cache",
    "build_web_context_text",
    "make_web_context_evidence_item",
    "deadline_expired",
    "decode_bing_redirect",
    "decode_duckduckgo_redirect",
    "extract_search_results",
    "http_get_text",
    "is_usable_web_url",
    "normalize_search_href",
    "remaining_timeout",
    "resolve_search_redirect_url",
    "strip_html_to_text",
    "web_search_results",
]
