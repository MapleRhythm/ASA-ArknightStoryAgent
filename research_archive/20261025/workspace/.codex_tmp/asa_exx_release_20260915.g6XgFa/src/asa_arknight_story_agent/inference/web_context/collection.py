from __future__ import annotations

import re

from asa_arknight_story_agent.inference.web_context.config import WebContextConfig
from asa_arknight_story_agent.inference.web_context.pages import select_web_context_lines
from asa_arknight_story_agent.inference.web_context.rendering import make_web_context_evidence_item
from asa_arknight_story_agent.inference.web_context.search import (
    deadline_expired,
    http_get_text,
    remaining_timeout,
    resolve_search_redirect_url,
    strip_html_to_text,
    web_search_results,
)


def build_web_context_queries(
    *,
    story_name: str,
    question: str,
    question_terms: list[str],
    config: WebContextConfig,
) -> list[str]:
    question_terms_text = " ".join(question_terms)
    return [
        template.format(
            story_name=story_name,
            question=question,
            question_terms=question_terms_text,
        )
        for template in config.query_templates
    ][: config.max_search_queries]


def collect_web_search_results(config: WebContextConfig, queries: list[str], deadline: float) -> tuple[list[dict[str, str]], list[str]]:
    candidate_results: list[dict[str, str]] = []
    candidate_urls: list[str] = []
    for query in queries:
        if deadline_expired(deadline):
            break
        for result in web_search_results(query, config, deadline=deadline):
            url = result["url"]
            if url not in candidate_urls:
                candidate_urls.append(url)
                candidate_results.append(result)
            if len(candidate_urls) >= config.max_search_results:
                break
        if len(candidate_urls) >= config.max_search_results:
            break
    return candidate_results, candidate_urls


def fetch_web_context_pages(
    *,
    story_name: str,
    question: str,
    candidate_results: list[dict[str, str]],
    config: WebContextConfig,
    deadline: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pages: list[dict[str, str]] = []
    rejected_pages: list[dict[str, str]] = []
    for result in candidate_results:
        if deadline_expired(deadline):
            break
        url = result["url"]
        search_title = result.get("title", "")
        resolved_url = resolve_search_redirect_url(
            url,
            timeout=remaining_timeout(deadline, min(config.timeout_seconds, 2.0)),
            user_agent=config.user_agent,
        )
        raw_text = http_get_text(
            resolved_url,
            timeout=remaining_timeout(deadline, config.timeout_seconds),
            user_agent=config.user_agent,
        )
        if not raw_text:
            continue
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
        page_title = strip_html_to_text(title_match.group(1)) if title_match else ""
        title = page_title or search_title
        if "剧情" not in title:
            rejected_pages.append(
                {
                    "url": resolved_url,
                    "title": title[:120],
                    "search_title": search_title[:120],
                    "reason": "title_missing_剧情",
                }
            )
            continue
        stripped = strip_html_to_text(raw_text)
        excerpt = select_web_context_lines(
            stripped,
            story_name=story_name,
            question=question,
            max_chars=config.max_chars_per_page,
        )
        if not excerpt:
            continue
        pages.append({"url": resolved_url, "title": title[:120], "excerpt": excerpt})
        if len(pages) >= config.max_pages:
            break
    return pages, rejected_pages


def make_web_page_candidates(story_name: str, pages: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        make_web_context_evidence_item(
            story_name,
            page["excerpt"],
            [page["url"]],
            item_key=f"{story_name}:{page.get('url') or index}",
            title=page.get("title") or "",
        )
        for index, page in enumerate(pages, start=1)
    ]
