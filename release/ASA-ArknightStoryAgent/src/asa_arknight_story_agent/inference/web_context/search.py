from __future__ import annotations

import re
from urllib.parse import quote_plus

from asa_arknight_story_agent.inference.web_context.config import WebContextConfig
from asa_arknight_story_agent.inference.web_context.html import strip_html_to_text
from asa_arknight_story_agent.inference.web_context.http import (
    deadline_expired,
    http_get_text,
    remaining_timeout,
    resolve_search_redirect_url,
)
from asa_arknight_story_agent.inference.web_context.url import (
    decode_bing_redirect,
    decode_duckduckgo_redirect,
    is_usable_web_url,
    normalize_search_href,
)


def extract_search_results(search_html: str, *, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    result_pattern = re.compile(
        r'''<a[^>]+href=["'](?P<href>[^"']+)["'][^>]*>(?P<title>.*?)</a>''',
        re.IGNORECASE | re.DOTALL,
    )
    for match in result_pattern.finditer(search_html or ""):
        title = strip_html_to_text(match.group("title"))
        if "剧情" not in title:
            continue
        href = normalize_search_href(match.group("href"))
        if not is_usable_web_url(href) or href in seen_urls:
            continue
        seen_urls.add(href)
        results.append({"url": href, "title": title[:160]})
        if len(results) >= limit:
            return results
    return results


def web_search_results(query: str, config: WebContextConfig, *, deadline: float | None = None) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    encoded_query = quote_plus(query)
    for template in config.search_url_templates:
        if deadline_expired(deadline):
            break
        search_url = template.format(query=encoded_query)
        raw_html = http_get_text(
            search_url,
            timeout=remaining_timeout(deadline, config.timeout_seconds),
            user_agent=config.user_agent,
        )
        if not raw_html:
            continue
        for result in extract_search_results(raw_html, limit=config.max_search_results):
            url = result["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                results.append(result)
            if len(results) >= config.max_search_results:
                return results
    return results


__all__ = [
    "strip_html_to_text",
    "decode_duckduckgo_redirect",
    "decode_bing_redirect",
    "normalize_search_href",
    "is_usable_web_url",
    "extract_search_results",
    "http_get_text",
    "resolve_search_redirect_url",
    "remaining_timeout",
    "deadline_expired",
    "web_search_results",
]
