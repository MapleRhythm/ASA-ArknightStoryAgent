from __future__ import annotations

import time
from urllib.parse import urljoin, urlparse

from asa_arknight_story_agent.inference.web_context.url import is_usable_web_url


def http_get_text(url: str, *, timeout: float, user_agent: str) -> str:
    try:
        import requests
    except Exception:
        return ""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.9,*/*;q=0.8"},
        )
        if response.status_code >= 400:
            return ""
        content_type = response.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type and content_type:
            return ""
        response.encoding = response.encoding or "utf-8"
        return response.text
    except Exception:
        return ""


def resolve_search_redirect_url(url: str, *, timeout: float, user_agent: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"baidu.com", "www.baidu.com"} or parsed.path != "/link":
        return url
    try:
        import requests
    except Exception:
        return url
    try:
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=False,
            headers={"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.9,*/*;q=0.8"},
        )
        location = response.headers.get("location") or ""
        if not location:
            return url
        resolved = urljoin(url, location)
        return resolved if is_usable_web_url(resolved) else url
    except Exception:
        return url


def remaining_timeout(deadline: float | None, default_timeout: float) -> float:
    if deadline is None:
        return default_timeout
    return max(0.5, min(default_timeout, deadline - time.time()))


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline
