#!/usr/bin/env python3
"""Resolve unsafe GLM gold-binding judgements against complete evidence.

By default this remains the third, ambiguity-aware pass for merged-vs-
individual contradictions.  ``--include-truncated`` additionally selects any
fact whose original judge saw fewer characters than the manifest contains.
Those rows are judged blindly without the unsafe prior verdicts.

The script never changes the strict recalibration artifact.  It emits a
sidecar with a SHA256 fingerprint of the complete claim/evidence payload so
downstream builders can reject stale or truncated adjudications.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
ADJUDICATION_LABELS = {
    "supported_by_union",
    "supported_by_some",
    "unsupported",
    "ambiguous",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_fingerprint(fact: dict[str, Any]) -> str:
    eids = [str(item) for item in fact.get("cited_eids") or []]
    texts = fact.get("cited_texts") or {}
    payload = {
        "claim": str(fact.get("claim") or ""),
        "cited_eids": eids,
        "cited_texts": {eid: str(texts.get(eid) or "") for eid in eids},
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def full_adjudication_matches(
    fact: dict[str, Any], adjudication: dict[str, Any] | None
) -> bool:
    if not (
        adjudication
        and adjudication.get("status") == "ok"
        and adjudication.get("full_evidence") is True
        and adjudication.get("evidence_sha256") == evidence_fingerprint(fact)
    ):
        return False
    label = str(adjudication.get("label") or "")
    if label not in ADJUDICATION_LABELS:
        return False
    cited = {str(item) for item in fact.get("cited_eids") or []}
    kept = [str(item) for item in adjudication.get("keep_eids") or [] if str(item)]
    if any(eid not in cited for eid in kept):
        return False
    if label in {"supported_by_union", "supported_by_some"}:
        return bool(kept)
    return not kept


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


def truncated_eids(row: dict[str, Any], fact: dict[str, Any]) -> list[str]:
    try:
        limit = int(row.get("chunk_chars"))
    except (TypeError, ValueError):
        return [str(item) for item in row.get("cited_eids") or []]
    texts = fact.get("cited_texts") or {}
    return [
        str(eid)
        for eid in row.get("cited_eids") or []
        if len(str(texts.get(str(eid)) or "")) > limit
    ]


def adjudication_kind(
    row: dict[str, Any], fact: dict[str, Any], include_truncated: bool
) -> str | None:
    conflict = contradiction_kind(row)
    truncated = truncated_eids(row, fact) if include_truncated else []
    if truncated and conflict:
        return f"truncated_evidence+{conflict}"
    if truncated:
        return "truncated_evidence"
    return conflict


def messages(
    claim: str,
    cited_eids: list[str],
    cited_texts: dict[str, str],
    merged: dict[str, Any],
    individual: dict[str, Any],
    *,
    include_prior: bool,
) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[{eid}]\n{str(cited_texts.get(eid) or '')}" for eid in cited_eids
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
    prior = ""
    if include_prior:
        prior = (
            f"\n原 merged 判定：{merged.get('verdict')}；"
            f"理由：{str(merged.get('reason') or '')[:900]}\n"
            f"原 individual 判定：\n{indiv}\n"
        )
    user = (
        "请裁决这条主张的证据绑定状态。输出一个 JSON 对象，不要 Markdown。\n"
        "标签只能是：supported_by_union（多段合并后直接支持）、"
        "supported_by_some（至少一段直接支持但整体主张含未支持部分）、"
        "unsupported（证据不支持）、ambiguous（指代/语境存在无法可靠消解的歧义）。\n"
        "判定对象是整条主张，不要因为某个子句相关就判整条支持。\n\n"
        f"主张：{claim}\n\n证据：\n{evidence}\n"
        f"{prior}\n"
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


def call(
    row: dict[str, Any],
    fact: dict[str, Any],
    api_key: str,
    timeout: float,
    kind: str,
) -> dict[str, Any]:
    is_truncated = "truncated_evidence" in kind
    cited_eids = [str(eid) for eid in row.get("cited_eids") or []]
    cited_texts = fact.get("cited_texts") or {}
    body = {
        "model": MODEL,
        "messages": messages(
            str(fact.get("claim") or ""),
            cited_eids,
            cited_texts,
            row.get("merged") or {},
            row.get("individual") or {},
            include_prior=not is_truncated,
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
    attempts = 0
    for attempt in range(1, 4):
        attempts = attempt
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            content = parse_content((choices[0].get("message") or {}).get("content")) if choices else ""
            parsed = parse_object(content)
            label = str((parsed or {}).get("label") or "")
            if label not in ADJUDICATION_LABELS:
                raise RuntimeError("invalid_label")
            keep = [
                str(eid)
                for eid in (parsed.get("keep_eids") or [])
                if str(eid) in {str(x) for x in row.get("cited_eids") or []}
            ]
            if label in {"supported_by_union", "supported_by_some"} and not keep:
                raise RuntimeError("supported_label_without_keep_eids")
            if label in {"unsupported", "ambiguous"}:
                keep = []
            return {
                "fact_id": fact.get("fact_id"),
                "row_id": row.get("row_id"),
                "fact_index": row.get("fact_index"),
                "kind": kind,
                "label": label,
                "reason": str((parsed or {}).get("reason") or "")[:1200],
                "keep_eids": keep,
                "status": "ok",
                "attempts": attempt,
                "full_evidence": True,
                "evidence_sha256": evidence_fingerprint(fact),
                "evidence_chars": {
                    eid: len(str(cited_texts.get(eid) or "")) for eid in cited_eids
                },
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            last = f"HTTP {exc.code}: {detail}"
            if attempt < 3 and exc.code == 429:
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))
                continue
            break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last = str(exc)
            if attempt < 3:
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))
    return {
        "fact_id": fact.get("fact_id"),
        "row_id": row.get("row_id"),
        "fact_index": row.get("fact_index"),
        "kind": kind,
        "label": "error",
        "reason": last[:1200],
        "keep_eids": [],
        "status": "error",
        "attempts": attempts,
        "full_evidence": True,
        "evidence_sha256": evidence_fingerprint(fact),
        "evidence_chars": {
            eid: len(str((fact.get("cited_texts") or {}).get(eid) or ""))
            for eid in cited_eids
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--include-truncated",
        action="store_true",
        help="Also rejudge rows whose recorded chunk_chars did not cover full evidence.",
    )
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing {args.api_key_env}")
    facts = {
        str(row.get("fact_id")): row
        for row in read_jsonl(args.facts)
        if row.get("fact_id")
    }
    candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row in read_jsonl(args.judgements):
        fact = facts.get(str(row.get("fact_id") or ""))
        if fact is None:
            continue
        kind = adjudication_kind(row, fact, args.include_truncated)
        if kind:
            candidates.append((row, fact, kind))
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in read_jsonl(args.output):
            if row.get("status") == "ok" and row.get("fact_id"):
                existing[str(row["fact_id"])] = row
    complete: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for row, fact, kind in candidates:
        previous = existing.get(str(row.get("fact_id") or ""))
        if full_adjudication_matches(fact, previous):
            complete[str(row["fact_id"])] = previous
        else:
            pending.append((row, fact, kind))
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(3, args.workers))) as pool:
            futures = [
                pool.submit(call, row, fact, api_key, args.timeout, kind)
                for row, fact, kind in pending
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(compact(result) + "\n")
                    handle.flush()
                    if result.get("status") == "ok":
                        complete[str(result.get("fact_id"))] = result
                if index == 1 or index % 10 == 0 or index == len(futures):
                    print(f"progress {index}/{len(futures)}", flush=True)
    # The append-only journal is convenient for interruption recovery, but
    # duplicate/error rows must not be consumed as a completed sidecar.  Once
    # this invocation has accounted for every candidate currently available,
    # rewrite a canonical latest-ok view in candidate order.
    canonical_rows = [
        complete[str(row.get("fact_id") or "")]
        for row, _, _ in candidates
        if str(row.get("fact_id") or "") in complete
    ]
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as handle:
        for result in canonical_rows:
            handle.write(compact(result) + "\n")
    tmp_output.replace(args.output)
    counts = Counter(str(row.get("label") or "missing") for row in complete.values())
    report = {
        "total_candidates": len(candidates),
        "include_truncated": args.include_truncated,
        "complete": len(complete),
        "pending_after_run": len(candidates) - len(complete),
        "label_counts": dict(sorted(counts.items())),
        "kind_counts": dict(
            Counter(
                f"{row.get('kind')}:{row.get('label')}"
                for row in complete.values()
            )
        ),
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["pending_after_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
