from __future__ import annotations

import base64
import html
from urllib.parse import parse_qs, unquote, urlparse

from asa_arknight_story_agent.inference.common.lexicon import (
    WEB_CONTEXT_BLOCKED_URL_HOSTS,
    WEB_CONTEXT_STATIC_URL_SUFFIXES,
)


def decode_duckduckgo_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def decode_bing_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    encoded = next((query[key][0] for key in ("u", "url") if query.get(key)), "")
    if not encoded:
        return url
    if encoded.startswith("a1") and len(encoded) > 4:
        payload = encoded[2:]
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8", "ignore")
            if decoded.startswith(("http://", "https://")):
                return decoded
        except Exception:
            pass
    return unquote(encoded)


def normalize_search_href(href: str) -> str:
    href = html.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/l/?") or "duckduckgo.com/l/?" in href:
        href = decode_duckduckgo_redirect(href)
    if href.startswith(("/url?", "/ck/a")) or "bing.com/ck/a" in href:
        href = decode_bing_redirect(href)
    return unquote(href)


def is_usable_web_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return False
    if host in WEB_CONTEXT_BLOCKED_URL_HOSTS:
        if host in {"baidu.com", "www.baidu.com"} and parsed.path == "/link":
            return True
        return False
    normalized_url = url.lower().split("?", 1)[0]
    if any(normalized_url.endswith(suffix) for suffix in WEB_CONTEXT_STATIC_URL_SUFFIXES):
        return False
    if "/rs/" in normalized_url or "/assets/" in normalized_url or "/static/" in normalized_url:
        return False
    return True
