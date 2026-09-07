#!/usr/bin/env python3
"""Rejudge truncated binding hard-negative candidates with complete evidence.

The historical hard-negative builder clipped every selected candidate to
2,000 characters before GLM judging.  This script joins that artifact with
the original grounded prompts, submits only quarantined candidates with their
complete source text, and writes one provenance-bearing result per candidate.
Old verdicts are deliberately omitted from the prompt.
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
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from audit_binding_hardneg import candidate_rows, read_records, source_evidence
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_binding_hardneg import candidate_rows, read_records, source_evidence


ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"
VERDICTS = {"supported", "unsupported"}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_fingerprint(
    record_id: str,
    fact_index: int,
    claim: str,
    eid: str,
    text: str,
) -> str:
    payload = {
        "record_id": record_id,
        "fact_index": fact_index,
        "claim": claim,
        "eid": eid,
        "text": text,
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def prompt_messages(
    claim: str, candidates: list[tuple[str, str, str]]
) -> list[dict[str, str]]:
    payload = [
        {"eid": eid, "candidate_role": role, "text": text}
        for eid, role, text in candidates
    ]
    system = (
        "你是严格的证据绑定审核员。只能依据每条给出的完整证据文本，"
        "独立判断它是否支持主张，不得使用外部知识，也不能把同主题、"
        "同人物或背景相关误判为支持。"
    )
    user = (
        "请逐条判断候选证据是否支持主张。证据已经是完整文本；不要假设"
        "未提供的上下文。只有证据能直接证明主张的核心事实、主体和关键"
        "关系时 supports=true；仅相关、背景、缺少关键断言或主体不明时"
        "supports=false。每个 eid 必须输出一次，只输出 JSON，不要 Markdown。\n\n"
        f"主张：{claim}\n\n候选证据：\n{compact(payload)}\n\n"
        '格式：{"verdicts":[{"eid":"E1","supports":false,"reason":"一句中文理由"}]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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
        parts = [part.strip() for part in raw.split("```") if part.strip()]
        raw = next((part[4:].strip() if part.lower().startswith("json") else part for part in parts if part.startswith("{")), raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def call(
    api_key: str,
    claim: str,
    candidates: list[tuple[str, str, str]],
    *,
    timeout: float,
    attempts: int,
) -> dict[str, dict[str, Any]]:
    body = {
        "model": MODEL,
        "messages": prompt_messages(claim, candidates),
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "medium",
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ca_bundle = "/etc/ssl/certs/ca-certificates.crt"
    context = (
        ssl.create_default_context(cafile=ca_bundle)
        if os.path.exists(ca_bundle)
        else ssl.create_default_context()
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )
    expected = {eid for eid, _, _ in candidates}
    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            content = (
                parse_content((choices[0].get("message") or {}).get("content"))
                if choices
                else ""
            )
            parsed = parse_object(content)
            raw = (parsed or {}).get("verdicts")
            if not isinstance(raw, list):
                raise RuntimeError("missing_verdicts")
            result: dict[str, dict[str, Any]] = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                eid = str(item.get("eid") or "")
                if eid not in expected:
                    continue
                supports = item.get("supports")
                if isinstance(supports, bool):
                    verdict = "supported" if supports else "unsupported"
                else:
                    verdict = str(item.get("verdict") or "").strip().lower()
                    if verdict in {"support", "entailed", "true", "yes"}:
                        verdict = "supported"
                    elif verdict in {"unsupported", "false", "no", "not_supported"}:
                        verdict = "unsupported"
                if verdict not in VERDICTS:
                    continue
                result[eid] = {
                    "glm_verdict": verdict,
                    "glm_reason": str(item.get("reason") or "")[:800],
                }
            if set(result) != expected:
                raise RuntimeError(f"missing_eids:{sorted(expected - set(result))}")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            last = f"HTTP {exc.code}: {detail}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last = str(exc)
        if attempt < max(1, attempts):
            time.sleep(min(12.0, 2 ** (attempt - 1) + random.random()))
    return {
        eid: {"glm_verdict": "error", "glm_reason": last[:800]}
        for eid, _, _ in candidates
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing {args.api_key_env}")

    hardneg_rows = read_records(args.hardneg)
    source = source_evidence(read_records(args.source_rows))
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in hardneg_rows:
        record_id = str(row.get("record_id") or "")
        fact_index = int(row.get("fact_index", -1))
        claim = str(row.get("claim") or "")
        visible = source.get(record_id)
        if visible is None:
            errors.append(f"missing source row: {record_id}")
            continue
        for role, candidate in candidate_rows(row):
            eid = str(candidate.get("eid") or "")
            shown = str(candidate.get("text") or "")
            full = visible.get(eid)
            if full is None:
                errors.append(f"{record_id}#{fact_index}: missing E-ID {eid}")
                continue
            if shown != full[: len(shown)]:
                errors.append(f"{record_id}#{fact_index}/{eid}: non-prefix candidate")
                continue
            if len(full) <= len(shown):
                continue
            candidates.append(
                {
                    "record_id": record_id,
                    "fact_index": fact_index,
                    "claim": claim,
                    "eid": eid,
                    "candidate_role": role,
                    "text": full,
                    "evidence_sha256": evidence_fingerprint(
                        record_id, fact_index, claim, eid, full
                    ),
                    "evidence_chars": len(full),
                }
            )

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in read_records(args.output):
            if row.get("status") == "ok" and row.get("evidence_sha256"):
                key = "|".join(
                    [
                        str(row.get("record_id") or ""),
                        str(row.get("fact_index", -1)),
                        str(row.get("eid") or ""),
                        str(row.get("candidate_role") or ""),
                    ]
                )
                existing[key] = row

    pending = [
        item
        for item in candidates
        if "|".join(
            [
                item["record_id"],
                str(item["fact_index"]),
                item["eid"],
                item["candidate_role"],
            ]
        )
        not in existing
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for item in pending:
        groups.setdefault((item["record_id"], item["fact_index"], item["claim"]), []).append(item)

    results: dict[str, dict[str, Any]] = dict(existing)
    lock = threading.Lock()

    def run_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        verdicts = call(
            api_key,
            group[0]["claim"],
            [
                (item["eid"], item["candidate_role"], item["text"])
                for item in group
            ],
            timeout=args.timeout,
            attempts=args.attempts,
        )
        output: list[dict[str, Any]] = []
        for item in group:
            verdict = verdicts.get(item["eid"]) or {
                "glm_verdict": "error",
                "glm_reason": "missing_candidate_verdict",
            }
            output.append(
                {
                    **item,
                    **verdict,
                    "status": "ok" if verdict.get("glm_verdict") in VERDICTS else "error",
                    "full_evidence": True,
                    "attempts": args.attempts,
                }
            )
        return output

    if groups:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(8, args.workers))
        ) as pool:
            futures = [pool.submit(run_group, group) for group in groups.values()]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                for result in future.result():
                    key = "|".join(
                        [
                            str(result["record_id"]),
                            str(result["fact_index"]),
                            str(result["eid"]),
                            str(result["candidate_role"]),
                        ]
                    )
                    with lock:
                        if result.get("status") == "ok":
                            results[key] = result
                if index == 1 or index % 10 == 0 or index == len(futures):
                    print(f"progress {index}/{len(futures)}", flush=True)

    ordered = []
    for item in candidates:
        key = "|".join(
            [
                item["record_id"],
                str(item["fact_index"]),
                item["eid"],
                item["candidate_role"],
            ]
        )
        if key in results:
            ordered.append(results[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(compact(row) + "\n")

    counts = Counter(str(row.get("glm_verdict") or "missing") for row in ordered)
    report = {
        "candidates": len(candidates),
        "groups": len(groups),
        "completed": len(ordered),
        "pending_after_run": len(candidates) - len(ordered),
        "verdicts": dict(sorted(counts.items())),
        "validation_errors": errors,
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["pending_after_run"] or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
