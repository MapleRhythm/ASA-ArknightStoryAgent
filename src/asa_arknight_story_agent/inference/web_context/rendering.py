from __future__ import annotations

import hashlib
from typing import Any

from asa_arknight_story_agent.inference.common.text_utils import truncate_text


def build_web_context_text(
    *,
    story_name: str,
    queries: list[str],
    pages: list[dict[str, str]],
    max_total_chars: int,
) -> str:
    sections = [
        "联网补充资料：以下内容来自运行时网页检索，主要用于补足剧情解析与时间线线索；若与本地游戏文本证据冲突，以本地证据为准。",
        f"召回链命中最多的剧情集：{story_name}",
        "联网检索词：" + " | ".join(queries),
    ]
    for index, page in enumerate(pages, start=1):
        excerpt = page.get("excerpt", "").strip()
        if not excerpt:
            continue
        sections.append(
            "\n".join(
                [
                    f"[联网来源 {index}] {page.get('title') or page.get('url') or ''}",
                    f"url: {page.get('url') or ''}",
                    "摘录:",
                    excerpt,
                ]
            )
        )
    return truncate_text("\n\n".join(sections), max_total_chars)


def make_web_context_evidence_item(
    story_name: str,
    text: str,
    urls: list[str],
    *,
    item_key: str = "",
    title: str = "",
) -> dict[str, Any]:
    key = item_key or story_name
    doc_id = "web_context/" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    source_path = " | ".join(urls[:3])
    title_prefix = f"{title}\n" if title else ""
    return {
        "doc_index": -1,
        "document": {
            "id": doc_id,
            "activity_name": "联网补充资料",
            "story_name": f"{story_name} 剧情解析/时间线",
            "stage_code": "",
            "avg_tag": "联网资料",
            "source_path": source_path,
            "clean_text": title_prefix + text,
            "search_text": title_prefix + text,
        },
        "fusion_score": 0.0,
        "rerank_score": 0.0,
        "dense_score": None,
        "sparse_score": None,
        "minirag_score": None,
        "supplemental_source": "web_context",
        "prompt_prefer_clean_text": True,
    }
