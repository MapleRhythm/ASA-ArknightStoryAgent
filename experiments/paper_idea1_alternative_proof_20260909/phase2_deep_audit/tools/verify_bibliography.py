#!/usr/bin/env python3
"""Capture primary-source responses/metadata without trusting search snippets.

An inaccessible page is unverified, not evidence a paper does not exist.
Only bibliographic identity is checked; method/novelty requires full-paper review.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("cue_r", "https://arxiv.org/abs/2604.05467", "CUE-R"),
    ("pair_id", "https://arxiv.org/abs/2608.08944", "Pair-ID"),
    ("grab_rag", "https://arxiv.org/abs/2608.22228", "Prompt-Based Abstention"),
    ("old_false_negative_citation", "https://aclanthology.org/2021.emnlp-main.315/", "Mitigating False-Negative Contexts"),
    ("musique", "https://aclanthology.org/2022.tacl-1.31/", "MuSiQue"),
]


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta, self.title, self.in_title = {}, [], False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            name = attrs.get("name", attrs.get("property", ""))
            if name.startswith("citation_") or name in {"og:title", "description"}:
                self.meta.setdefault(name, []).append(attrs.get("content", ""))

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)


def fetch(source):
    key, url, expected = source
    result = subprocess.run(
        ["curl", "-L", "--max-time", "20", "-sS", "-w", "\n__HTTP_STATUS__:%{http_code}", url],
        capture_output=True, timeout=25)
    body, _, status = result.stdout.rpartition(b"\n__HTTP_STATUS__:")
    code = status.decode(errors="replace").strip()
    parsed = Metadata()
    parsed.feed(body.decode("utf-8", errors="replace"))
    actual = (parsed.meta.get("citation_title") or ["".join(parsed.title)])[0]
    success = result.returncode == 0 and code == "200" and bool(actual)
    expected_match = expected.lower() in actual.lower() if success else None
    out = ROOT / "audits/bibliography"
    out.mkdir(parents=True, exist_ok=True)
    # Store retrieved source as an audit artifact, never execute its contents.
    if body:
        (out / f"{key}.html").write_bytes(body)
    return {
        "key": key, "url": url, "checked_at": datetime.datetime.now().astimezone().isoformat(),
        "expected_title_fragment": expected, "actual_title": actual,
        "status": "verified_title_match" if expected_match else "verified_title_mismatch" if success else "unverified",
        "http_status": code, "curl_exit_code": result.returncode,
        "stderr": result.stderr.decode(errors="replace")[:1000],
        "metadata": parsed.meta, "body_sha256": hashlib.sha256(body).hexdigest() if body else None,
        "body_file": f"audits/bibliography/{key}.html" if body else None,
        "full_paper_method_claims_verified": False,
    }


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, SOURCES))
    output = {
        "web_tool_note": "web.run returned empty results during this audit; direct primary HTTP used as fallback",
        "records": records, "novelty_search_complete": False,
    }
    path = ROOT / "audits/bibliography/verification.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for r in records:
        print(r["key"], r["status"], r["actual_title"])


if __name__ == "__main__":
    main()
