#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = "https://moegirl.icu/%E6%98%8E%E6%97%A5%E6%96%B9%E8%88%9F/%E4%B8%96%E7%95%8C%E8%A7%82"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "moegirl_arknights_lore"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "documents.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.json"
API_URL = "https://moegirl.icu/api.php"
SOURCE_NAME = "萌娘百科"
LICENSE_NOTE = (
    "资料引自萌娘百科。页面声明要求转载标注来源并声明引自萌娘百科，且不得用于商业用途；"
    "本脚本在每个 chunk 元数据中保留 source_url/source_name/license_note。"
)

ANCHOR_RE = re.compile(r"<a\s+([^>]*?)>", re.IGNORECASE | re.DOTALL)
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
HEADING_RE = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}$")
REF_RE = re.compile(r"\[\d+\]")
BLANK_RE = re.compile(r"\n{3,}")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
SKIP_SECTION_RE = re.compile(
    r"^(?:关联条目|注释|外部链接|参考资料|导航菜单|个人工具|命名空间|变体|页面工具|分类)"
)
INFORMATIVE_PUNCT_RE = re.compile(r"[。！？；：,.，、（）()《》“”]")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_title(title: str) -> str:
    cleaned = html.unescape(title).strip()
    cleaned = cleaned.replace("_", " ")
    return cleaned


def title_to_url(title: str) -> str:
    return "https://moegirl.icu/" + requests.utils.quote(title.replace(" ", "_"), safe="/()·")


def extract_anchor_attrs(anchor: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in ATTR_RE.findall(anchor):
        attrs[key.lower()] = html.unescape(value)
    return attrs


def collect_titles_from_html_section(section_html: str) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for match in ANCHOR_RE.finditer(section_html):
        attrs = extract_anchor_attrs(match.group(1))
        href = attrs.get("href", "")
        title = attrs.get("title", "")
        class_name = attrs.get("class", "")
        if not href or not title:
            continue
        if "redlink=1" in href or "new" in class_name:
            continue
        if title.startswith(("File:", "Template:", "Category:", "Special:", "Help:", "Talk:")):
            continue
        if "页面不存在" in title:
            continue
        if href.startswith("http") and "moegirl.icu" not in href:
            continue
        normalized = canonical_title(title)
        if normalized and normalized not in seen:
            seen.add(normalized)
            titles.append(normalized)
    return titles


def collect_related_titles(source_html: str, *, max_story_character_pages: int | None) -> tuple[list[str], dict[str, Any]]:
    related_index = source_html.find('id="关联条目"')
    if related_index < 0:
        related_index = source_html.find("关联条目")
    if related_index < 0:
        raise RuntimeError("Could not find 关联条目 section in source page.")

    navbox_index = source_html.find('<table class="navbox"', related_index)
    direct_html = source_html[related_index: navbox_index if navbox_index > related_index else related_index + 8000]
    world_titles = ["明日方舟/世界观"]
    for title in collect_titles_from_html_section(direct_html):
        if title not in world_titles:
            world_titles.append(title)

    story_marker = source_html.find("明日方舟/剧情角色", related_index)
    story_titles: list[str] = []
    navbox_world_titles: list[str] = []
    if story_marker >= 0:
        next_marker = source_html.find("明日方舟/跨媒体作品", story_marker)
        if next_marker < 0:
            next_marker = source_html.find("<h2", story_marker)
        section_html = source_html[story_marker: next_marker if next_marker > story_marker else story_marker + 100000]
        section_titles = collect_titles_from_html_section(section_html)
        try:
            story_heading_index = section_titles.index("明日方舟/剧情角色")
        except ValueError:
            story_heading_index = len(section_titles)
        story_titles = section_titles[:story_heading_index]
        if max_story_character_pages is not None:
            story_titles = story_titles[:max_story_character_pages]

        world_tail = section_titles[story_heading_index + 1 :]
        for title in world_tail:
            if title in {"明日方舟/地图/常规地图", "明日方舟/序章 黑暗时代 上"}:
                break
            navbox_world_titles.append(title)

    titles: list[str] = []
    for title in [*world_titles, *story_titles, *navbox_world_titles]:
        if title not in titles:
            titles.append(title)
    manifest = {
        "world_titles": world_titles,
        "navbox_world_titles": navbox_world_titles,
        "story_character_titles": story_titles,
        "world_title_count": len(world_titles),
        "navbox_world_title_count": len(navbox_world_titles),
        "story_character_title_count": len(story_titles),
        "total_title_count": len(titles),
    }
    return titles, manifest


def request_json(
    session: requests.Session,
    params: dict[str, Any],
    *,
    request_interval: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("MediaWiki API returned non-object JSON.")
            if request_interval > 0:
                time.sleep(request_interval)
            return payload
        except Exception as exc:  # Keep crawling through transient CDN/API failures.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_extracts(
    session: requests.Session,
    titles: list[str],
    *,
    request_interval: float,
    retries: int,
    retry_sleep: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    pages: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for title in titles:
        try:
            payload = request_json(
                session,
                {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|info",
                    "explaintext": "1",
                    "inprop": "url",
                    "redirects": "1",
                    "titles": title,
                },
                request_interval=request_interval,
                retries=retries,
                retry_sleep=retry_sleep,
            )
        except Exception as exc:
            failed.append({"title": title, "error": str(exc)})
            continue
        query = payload.get("query") or {}
        for page in (query.get("pages") or {}).values():
            if not isinstance(page, dict) or "missing" in page:
                continue
            extract = str(page.get("extract") or "").strip()
            if not extract:
                continue
            pages.append(page)
    pages.sort(key=lambda item: canonical_title(str(item.get("title") or "")))
    return pages, failed


def clean_extract_text(text: str) -> str:
    lines: list[str] = []
    skip_section = False
    for raw_line in text.splitlines():
        line = html.unescape(raw_line).strip()
        if not line:
            lines.append("")
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading = heading_match.group(1).strip()
            skip_section = bool(SKIP_SECTION_RE.match(heading))
            if not skip_section:
                lines.append(heading)
            continue
        if skip_section:
            continue
        if line.startswith(("本页面", "欢迎来到", "萌娘百科")) and len(line) < 80:
            continue
        line = REF_RE.sub("", line)
        line = SPACE_RE.sub(" ", line)
        if line:
            lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = BLANK_RE.sub("\n\n", cleaned).strip()
    return cleaned


def split_paragraphs(text: str, *, max_chars: int, overlap_paragraphs: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        if overlap_paragraphs > 0:
            current = current[-overlap_paragraphs:]
            current_len = sum(len(item) + 2 for item in current)
        else:
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            sentences = re.split(r"(?<=[。！？；])", paragraph)
            segment = ""
            for sentence in sentences:
                if not sentence:
                    continue
                if segment and len(segment) + len(sentence) > max_chars:
                    chunks.append(segment.strip())
                    segment = sentence
                else:
                    segment += sentence
            if segment.strip():
                chunks.append(segment.strip())
            continue
        if current and current_len + len(paragraph) + 2 > max_chars:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + 2
    flush()
    return [chunk for chunk in chunks if chunk]


def stable_page_id(title: str) -> str:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
    safe_title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff_.·-]+", "_", title).strip("_")
    return f"{safe_title[:48]}-{digest}"


def is_trivial_chunk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 12:
        return True
    if len(stripped) < 24 and not INFORMATIVE_PUNCT_RE.search(stripped):
        return True
    return False


def build_documents(pages: list[dict[str, Any]], *, max_chars: int, overlap_paragraphs: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for page in pages:
        title = canonical_title(str(page.get("title") or ""))
        source_url = str(page.get("fullurl") or title_to_url(title))
        clean_text = clean_extract_text(str(page.get("extract") or ""))
        if not clean_text:
            continue
        page_id = stable_page_id(title)
        chunks = [
            chunk
            for chunk in split_paragraphs(clean_text, max_chars=max_chars, overlap_paragraphs=overlap_paragraphs)
            if not is_trivial_chunk(chunk)
        ]
        for chunk_index, chunk in enumerate(chunks):
            search_parts = [
                SOURCE_NAME,
                "明日方舟",
                "萌百世界观资料",
                title,
                chunk,
            ]
            documents.append(
                {
                    "id": f"moegirl/{page_id}#chunk-{chunk_index:04d}",
                    "chunk_index": chunk_index,
                    "source_path": source_url,
                    "source_url": source_url,
                    "source_name": SOURCE_NAME,
                    "source_license_note": LICENSE_NOTE,
                    "clean_text": chunk,
                    "search_text": "\n".join(part for part in search_parts if part).strip(),
                    "segments": [
                        {
                            "speaker": None,
                            "text": chunk,
                            "segment_type": "moegirl_lore",
                        }
                    ],
                    "story_key": f"moegirl/{title}",
                    "story_id": f"moegirl/{title}",
                    "activity_id": "moegirl_lore",
                    "activity_name": "萌百世界观资料",
                    "story_name": title,
                    "story_code": None,
                    "avg_tag": "外部资料",
                    "story_sort": chunk_index,
                    "stage_id": None,
                    "stage_code": None,
                    "stage_name": None,
                    "stage_type": "EXTERNAL_LORE",
                    "zone_id": None,
                    "zone_name": None,
                    "chapter_name": None,
                    "trigger_type": "MOEGIRL_LORE",
                }
            )
    return documents


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Moegirl Arknights worldview/story-character related entries and export RAG documents."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--max-chars", type=int, default=520)
    parser.add_argument("--overlap-paragraphs", type=int, default=1)
    parser.add_argument(
        "--max-story-character-pages",
        type=int,
        default=0,
        help="Limit story-character navbox pages. 0 means no limit.",
    )
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--user-agent", default="ASA-ArknightStoryAgent/0.1 (+local RAG enrichment)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)
    manifest_path = resolve_path(args.manifest)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    source_response = session.get(args.source_url, timeout=60)
    source_response.raise_for_status()
    if args.request_interval > 0:
        time.sleep(args.request_interval)

    max_story = args.max_story_character_pages if args.max_story_character_pages > 0 else None
    titles, title_manifest = collect_related_titles(source_response.text, max_story_character_pages=max_story)
    pages, failed_pages = fetch_extracts(
        session,
        titles,
        request_interval=args.request_interval,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    documents = build_documents(pages, max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)
    write_jsonl(output_path, documents)

    page_titles = [canonical_title(str(page.get("title") or "")) for page in pages]
    manifest = {
        "source_url": args.source_url,
        "source_name": SOURCE_NAME,
        "license_note": LICENSE_NOTE,
        "requested_titles": len(titles),
        "fetched_pages": len(pages),
        "failed_pages": len(failed_pages),
        "failed_page_errors": failed_pages,
        "documents": len(documents),
        "max_chars": args.max_chars,
        "overlap_paragraphs": args.overlap_paragraphs,
        "title_manifest": title_manifest,
        "fetched_page_titles": page_titles,
        "output": str(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
