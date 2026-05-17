#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_PATH = PROJECT_ROOT / ".python_packages" / "third_party"
if THIRD_PARTY_PATH.exists():
    sys.path.insert(0, str(THIRD_PARTY_PATH))

from bs4 import BeautifulSoup

CHARACTER_TABLE_PATH = PROJECT_ROOT / "data" / "ArknightsGameData" / "zh_CN" / "gamedata" / "excel" / "character_table.json"
MANUAL_ALIAS_PATH = PROJECT_ROOT / "data" / "processed" / "operator_aliases_manual.json"
PRTS_OPERATOR_LIST_URL = "https://prts.wiki/w/%E5%B9%B2%E5%91%98%E4%B8%80%E8%A7%88"
PRTS_OPERATOR_PAGE_URL = "https://prts.wiki/w/{title}"

TAG_RE = re.compile(r"<[^>]+>")
NAME_TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff·\-.]{2,32}")
REAL_NAME_RE = re.compile(r"本名([A-Za-z\u4e00-\u9fff·\-.]{2,32})")
CODENAME_RE = re.compile(r"代号([A-Za-z\u4e00-\u9fff·\-.]{1,32})")
CODENAME_REAL_NAME_RE = re.compile(r"([A-Za-z\u4e00-\u9fff·\-.]{1,32})，本名([A-Za-z\u4e00-\u9fff·\-.]{2,32})")
LEADING_NAME_RE = re.compile(r"^([A-Za-z\u4e00-\u9fff·\-.]{2,16})，")

NON_NAME_MARKERS = (
    "本名",
    "真名",
    "代号",
    "原名",
    "区分",
    "干员",
    "公民",
    "专家",
    "研究",
    "主任",
    "领袖",
    "工程",
    "顾问",
    "骑士",
    "成员",
    "博士",
    "小姐",
    "先生",
    "人",
    "者",
    "家",
    "师",
    "员",
    "事件",
    "组织",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = cleaned.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    cleaned = TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def clean_name(value: str) -> str:
    cleaned = normalize_text(value).strip("，。；：:、,.;!?？！()（）[]【】<>《》\"' ")
    return cleaned


def looks_like_name(value: str, *, min_len: int = 2) -> bool:
    candidate = clean_name(value)
    if len(candidate) < min_len or len(candidate) > 16:
        return False
    return not any(marker in candidate for marker in NON_NAME_MARKERS)


def load_operator_names() -> list[str]:
    character_table = load_json(CHARACTER_TABLE_PATH)
    names: list[str] = []
    for payload in character_table.values():
        if not isinstance(payload, dict):
            continue
        name = clean_name(str(payload.get("name") or ""))
        if name and name not in names:
            names.append(name)
    return names


def fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def fetch_operator_titles(session: requests.Session) -> set[str]:
    html_text = fetch(session, PRTS_OPERATOR_LIST_URL)
    soup = BeautifulSoup(html_text, "html.parser")
    titles: set[str] = set()
    for link in soup.select('a[title]'):
        href = str(link.get("href") or "")
        if not href.startswith("/w/"):
            continue
        if any(prefix in href for prefix in ("/w/PRTS:", "/w/Help:", "/w/Special:", "/w/File:", "/w/Template:")):
            continue
        title = str(link.get("title") or "")
        cleaned = clean_name(title)
        if cleaned:
            titles.add(cleaned)
    return titles


def extract_resume_text(page_html: str) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    for marker in soup.find_all(["th", "p"]):
        marker_text = normalize_text(marker.get_text(" ", strip=True))
        if marker_text != "客观履历":
            continue

        row = marker.find_parent("tr")
        if row is None:
            continue

        current = row.find_next_sibling("tr")
        while current is not None:
            td = current.find("td")
            if td is not None:
                text = normalize_text(td.get_text("\n", strip=True))
                if text:
                    return text
            current = current.find_next_sibling("tr")
    return ""


def extract_aliases_from_resume(codename: str, resume_text: str) -> list[str]:
    aliases: list[str] = []

    pair_match = CODENAME_REAL_NAME_RE.search(resume_text)
    if pair_match:
        pair_codename = clean_name(pair_match.group(1))
        real_name = clean_name(pair_match.group(2))
        if pair_codename and pair_codename != codename:
            aliases.append(pair_codename)
        if looks_like_name(real_name, min_len=2) and real_name != codename:
            aliases.append(real_name)

    for match in REAL_NAME_RE.finditer(resume_text):
        real_name = clean_name(match.group(1))
        if looks_like_name(real_name, min_len=2) and real_name != codename:
            aliases.append(real_name)

    for match in CODENAME_RE.finditer(resume_text):
        alt_codename = clean_name(match.group(1))
        if looks_like_name(alt_codename, min_len=1) and alt_codename != codename:
            aliases.append(alt_codename)

    leading_match = LEADING_NAME_RE.match(resume_text)
    if leading_match:
        leading_name = clean_name(leading_match.group(1))
        if looks_like_name(leading_name, min_len=2) and leading_name != codename:
            aliases.append(leading_name)

    deduped: list[str] = []
    for alias in aliases:
        if alias and alias not in deduped:
            deduped.append(alias)
    return deduped


def build_prts_alias_map(operator_names: list[str], *, delay_seconds: float = 0.0) -> dict[str, list[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    available_titles = fetch_operator_titles(session)

    alias_map: dict[str, list[str]] = {}
    for index, codename in enumerate(operator_names, start=1):
        title = codename if codename in available_titles else codename
        url = PRTS_OPERATOR_PAGE_URL.format(title=quote(title, safe=""))
        try:
            page_html = fetch(session, url)
        except requests.RequestException:
            continue

        resume_text = extract_resume_text(page_html)
        aliases = extract_aliases_from_resume(codename, resume_text)
        if aliases:
            alias_map[codename] = aliases
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if index % 50 == 0:
            print(f"processed={index} aliases={len(alias_map)}")
    return alias_map


def merge_alias_maps(base_map: dict[str, list[str]], extra_map: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    all_keys = sorted(set(base_map) | set(extra_map))
    for key in all_keys:
        values: list[str] = []
        for source in (base_map.get(key, []), extra_map.get(key, [])):
            for item in source:
                cleaned = clean_name(item)
                if cleaned and cleaned != key and cleaned not in values:
                    values.append(cleaned)
        if values:
            merged[key] = values
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync operator alias map from PRTS.")
    parser.add_argument("--write", action="store_true", help="Write merged aliases back to operator_aliases_manual.json")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional delay between requests.")
    parser.add_argument("--only", nargs="*", default=[], help="Only fetch the provided operator codenames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operator_names = args.only or load_operator_names()
    current_manual_map = load_json(MANUAL_ALIAS_PATH)
    prts_alias_map = build_prts_alias_map(operator_names, delay_seconds=args.delay_seconds)
    merged_alias_map = merge_alias_maps(current_manual_map, prts_alias_map)

    if args.write:
        MANUAL_ALIAS_PATH.write_text(
            json.dumps(merged_alias_map, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "operators_scanned": len(operator_names),
                "prts_aliases_found": len(prts_alias_map),
                "manual_aliases_before": len(current_manual_map),
                "manual_aliases_after": len(merged_alias_map),
                "sample": {key: prts_alias_map[key] for key in sorted(prts_alias_map)[:10]},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
