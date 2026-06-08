from __future__ import annotations

import html
import re


def strip_html_to_text(payload: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|template)[^>]*>.*?</\1>", " ", payload or "")
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>|</div\s*>|</li\s*>|</h[1-6]\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) >= 8)
