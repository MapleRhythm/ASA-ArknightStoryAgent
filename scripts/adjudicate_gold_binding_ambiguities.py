#!/usr/bin/env python3
"""Resolve merged-vs-individual GLM binding contradictions.

This is deliberately a third, ambiguity-aware pass.  It never changes the
strict recalibration artifact; it emits a sidecar adjudication file that can
be used to build either a strict or an ambiguity-preserving training set.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def contradiction_kind(row: dict[str, Any]) -> str | None:
    merged = (row.get("merged") or {}).get("verdict")
    individual = row.get("individual") or {}
    eids = [str(eid) for eid in row.get("cited_eids") or []]
    supported = [
        eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "supported"
    ]
    if merged == "unsupported" and supported:
        return "merged_unsupported_ind_supported"
    if merged == "supported" and not supported:
        return "merged_supported_no_ind_supported"
    return None


def messages(
    claim: str,
    cited_eids: list[str],
    cited_texts: dict[str, str],
    merged: dict[str, Any],
    individual: dict[str, Any],
) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[{eid}]\n{str(cited_texts.get(eid) or '')[:1800]}" for eid in cited_eids
    )
    indiv = "\n".join(
        f"{eid}: {str((individual.get(eid) or {}).get('verdict'))}; "
        f"{str((individual.get(eid) or {}).get('reason') or '')[:500]}"
        for eid in cited_eids
    )
    system = (
        "你是证据绑定争议的第三方裁决员。只能依据给出的证据文本，不能使用外部知识。"
        "要区分证据逐段支持和多段合并支持；主体可由证据明确指代时算支持，"
        "但不能凭常识补全。"
    )
    user = (
        "请裁决这条主张的证据绑定状态。输出一个 JSON 对象，不要 Markdown。\n"
        "标签只能是：supported_by_union（多段合并后直接支持）、"
        "supported_by_some（至少一段直接支持但整体主张含未支持部分）、"
        "unsupported（证据不支持）、ambiguous（指代/语境存在无法可靠消解的歧义）。\n"
        "判定对象是整条主张，不要因为某个子句相关就判整条支持。\n\n"
        f"主张：{claim}\n\n证据：\n{evidence}\n\n"
        f"原 merged 判定：{merged.get('verdict')}；理由：{str(merged.get('reason') or '')[:900]}\n"
        f"原 individual 判定：\n{indiv}\n\n"
        '格式：{"label":"supported_by_union|supported_by_some|unsupported|ambiguous",'
        '"reason":"一句中文理由","keep_eids":["E1"]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            for item in value
            if isinstance(item, dict)
        )
    return str(value or "")


def parse_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if "```" in raw:
        raw = next(
            (part.strip() for part in raw.split("```") if part.strip().startswith("{")),
            raw,
        )
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def call(row: dict[str, Any], fact: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": messages(
            str(fact.get("claim") or ""),
            [str(eid) for eid in row.get("cited_eids") or []],
            fact.get("cited_texts") or {},
            row.get("merged") or {},
            row.get("individual") or {},
        ),
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "medium",
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    ca_bundle = "/etc/ssl/certs/ca-certificates.crt"
    ssl_ctx = (
        ssl.create_default_context(cafile=ca_bundle)
        if os.path.exists(ca_bundle)
        else ssl.create_default_context()
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl_ctx),
    )
    last = ""
    for attempt in range(1, 4):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            content = parse_content((choices[0].get("message") or {}).get("content")) if choices else ""
            parsed = parse_object(content)
            label = str((parsed or {}).get("label") or "")
            if label not in {
                "supported_by_union",
                "supported_by_some",
                "unsupported",
                "ambiguous",
            }:
                raise RuntimeError("invalid_label")
            keep = [
                str(eid)
                for eid in (parsed.get("keep_eids") or [])
                if str(eid) in {str(x) for x in row.get("cited_eids") or []}
            ]
            return {
                "fact_id": fact.get("fact_id"),
                "row_id": row.get("row_id"),
                "fact_index": row.get("fact_index"),
                "kind": contradiction_kind(row),
                "label": label,
                "reason": str((parsed or {}).get("reason") or "")[:1200],
                "keep_eids": keep,
                "status": "ok",
                "attempts": attempt,
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last = str(exc)
            if attempt < 3:
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            return {
                "fact_id": fact.get("fact_id"),
                "row_id": row.get("row_id"),
                "fact_index": row.get("fact_index"),
                "kind": contradiction_kind(row),
                "label": "error",
                "reason": f"HTTP {exc.code}: {detail}",
                "keep_eids": [],
                "status": "error",
                "attempts": attempt,
            }
    return {
        "fact_id": fact.get("fact_id"),
        "row_id": row.get("row_id"),
        "fact_index": row.get("fact_index"),
        "kind": contradiction_kind(row),
        "label": "error",
        "reason": last[:1200],
        "keep_eids": [],
        "status": "error",
        "attempts": 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing {args.api_key_env}")
    judgements = [row for row in read_jsonl(args.judgements) if contradiction_kind(row)]
    facts = {
        str(row.get("fact_id")): row
        for row in read_jsonl(args.facts)
        if row.get("fact_id")
    }
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in read_jsonl(args.output):
            if row.get("status") == "ok" and row.get("fact_id"):
                existing[str(row["fact_id"])] = row
    pending = [row for row in judgements if str(row.get("fact_id")) not in existing]
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(3, args.workers))) as pool:
            futures = [
                pool.submit(call, row, facts[str(row["fact_id"])], api_key, args.timeout)
                for row in pending
                if str(row.get("fact_id")) in facts
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(compact(result) + "\n")
                    handle.flush()
                    existing[str(result.get("fact_id"))] = result
                if index == 1 or index % 10 == 0 or index == len(futures):
                    print(f"progress {index}/{len(futures)}", flush=True)
    counts = Counter(str(row.get("label") or "missing") for row in existing.values())
    report = {
        "total_contradictions": len(judgements),
        "complete": len(existing),
        "pending_after_run": len(judgements) - len(existing),
        "label_counts": dict(sorted(counts.items())),
        "kind_counts": dict(
            Counter(
                f"{row.get('kind')}:{row.get('label')}"
                for row in existing.values()
            )
        ),
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
