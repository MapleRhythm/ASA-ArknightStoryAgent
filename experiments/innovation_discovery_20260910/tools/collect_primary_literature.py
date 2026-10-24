#!/usr/bin/env python3
"""Retrieve publication-owned metadata and selected pages with explicit failures."""
from __future__ import annotations

import concurrent.futures
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    **{year: f"https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{year}.xml"
       for year in ("2023.emnlp", "2022.emnlp", "2021.naacl", "2022.naacl", "2023.acl")},
    "cue_r_unverified": "https://arxiv.org/html/2604.05467v1",
    "pair_id_unverified": "https://arxiv.org/html/2608.08944v1",
    "grab_rag_unverified": "https://arxiv.org/html/2608.22228v1",
}
RECENT = {
    key: f"https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{key}.xml"
    for key in ("2025.acl", "2025.emnlp", "2026.acl", "2026.eacl")
}
FOCUSED = {
    key: f"https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{key}.xml"
    for key in ("2024.tacl", "2025.tacl", "2025.acl", "2026.acl")
}
PAPERS = {
    key: f"https://aclanthology.org/{key}.pdf"
    for key in ("2023.emnlp-main.322", "2022.emnlp-main.679",
                "2021.naacl-main.383", "2022.naacl-main.236",
                "2023.acl-short.64", "2023.emnlp-main.398")
}
NEAR_NEIGHBORS = {
    key: f"https://aclanthology.org/{key}.pdf"
    for key in ("2026.acl-long.1362", "2026.acl-long.814",
                "2025.acl-long.837", "2025.acl-long.828",
                "2026.acl-long.1054")
}
PATTERN = re.compile(
    r"query rewriting|CONQRR|FactGraph|FRANK:|FANToM|RARR:|citations|"
    r"false.negative contexts|attributable|attribution|factual consistency|"
    r"retrieval.*marginalization|sufficient context|evidence selection|"
    r"passage select|retriev.*reinforcement|query.*reinforcement|"
    r"counterfactual.*retrieval|retrieval.*counterfactual|belief.*graph|"
    r"RAG.*abst|abst.*RAG|evidence.*intervention|factuality|"
    r"epistemic|modal.*dependency|reported speech|multi.hop", re.I)


def fetch(key, url, out=None, timeout_seconds=25):
    result = subprocess.run(
        ["curl", "-L", "--max-time", str(timeout_seconds), "-sS", "-w", "\n__HTTP__:%{http_code}", url],
        capture_output=True, timeout=timeout_seconds + 5)
    body, _, status = result.stdout.rpartition(b"\n__HTTP__:")
    status = status.decode(errors="replace").strip()
    success = result.returncode == 0 and status == "200"
    out = out or ROOT / "sources"
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".xml" if url.endswith(".xml") else ".pdf" if url.endswith(".pdf") else ".html"
    path = out / (key + suffix)
    if body:
        path.write_bytes(body)
    record = {
        "key": key, "url": url, "http_status": status, "curl_exit_code": result.returncode,
        "retrieved_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "retrieved" if success else "unverified",
        "sha256": hashlib.sha256(body).hexdigest() if body else None,
        "file": str(path.relative_to(ROOT)) if body else None,
        "error": result.stderr.decode(errors="replace")[:1200],
    }
    if success and path.suffix == ".xml":
        root = ET.fromstring(body)
        matches = []
        for volume in root.findall("volume"):
            for paper in volume.findall("paper"):
                title = "".join(paper.find("title").itertext()) if paper.find("title") is not None else ""
                if not PATTERN.search(title):
                    continue
                paper_id = f"{root.attrib['id']}-{volume.attrib['id']}.{paper.attrib['id']}"
                matches.append({
                    "id": paper_id, "title": title,
                    "authors": [" ".join(filter(None, (x.findtext("first"), x.findtext("last"))))
                                for x in paper.findall("author")],
                    "abstract": "".join(paper.find("abstract").itertext()) if paper.find("abstract") is not None else "",
                    "url": f"https://aclanthology.org/{paper_id}/",
                })
        record["title_matches"] = matches
    if success and path.suffix == ".pdf":
        text_path = path.with_suffix(".txt")
        parsed = subprocess.run(["pdftotext", "-layout", str(path), str(text_path)],
                                capture_output=True, timeout=25)
        record["pdftotext_exit_code"] = parsed.returncode
        record["text_file"] = str(text_path.relative_to(ROOT)) if parsed.returncode == 0 else None
        record["text_sha256"] = hashlib.sha256(text_path.read_bytes()).hexdigest() if parsed.returncode == 0 else None
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("initial", "recent", "papers", "focused", "near_neighbors"), default="initial")
    parser.add_argument("--run-tag", help="New subdirectory preserving previous attempts; refuses reuse")
    parser.add_argument("--keys", nargs="+", help="Fetch only these exact IDs in the chosen group")
    parser.add_argument("--timeout-seconds", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 55:
        parser.error("timeout-seconds must be between 1 and 55")
    targets = {"initial": TARGETS, "recent": RECENT, "papers": PAPERS, "focused": FOCUSED,
               "near_neighbors": NEAR_NEIGHBORS}[args.group]
    if args.keys:
        unknown = set(args.keys) - targets.keys()
        if unknown:
            parser.error(f"unknown keys: {sorted(unknown)}")
        targets = {key: targets[key] for key in args.keys}
    out = ROOT / "sources"
    if args.run_tag:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_tag):
            parser.error("run-tag must be a simple directory name")
        out = out / args.run_tag
        out.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda kv: fetch(*kv, out=out, timeout_seconds=args.timeout_seconds),
                                targets.items()))
    manifest = "manifest.json" if args.group == "initial" else f"manifest_{args.group}.json"
    (out / manifest).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(record["key"], record["status"], record["http_status"])
        for p in record.get("title_matches", []):
            print(" ", p["id"], p["title"])


if __name__ == "__main__":
    main()
