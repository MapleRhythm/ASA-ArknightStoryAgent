#!/usr/bin/env python3
"""Collect independent GLM-5.3 support judgements for every pair."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The GLM endpoint is known to fail if inherited proxy variables point at a
# non-listening local proxy.  Clear all common spellings in-process as well as
# in the launch command.
for _proxy_name in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ.pop(_proxy_name, None)

ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"
PROXY_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content or "")


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if "```" in raw:
        pieces = raw.split("```")
        raw = next((p.strip() for p in pieces if p.strip().startswith("{")), raw)
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


def normalize_support(value: Any, raw: str) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "supported", "support", "entailed", "partial", "支持"}:
        return True
    if text in {"false", "no", "unsupported", "contradicted", "not_supported", "不支持"}:
        return False
    first = raw.strip().lower()[:80]
    if "不支持" in first or "unsupported" in first or "contradicted" in first:
        return False
    if "支持" in first or "supported" in first or "entailed" in first:
        return True
    return None


def messages(pair: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是严格、独立的证据绑定审核员。只能依据给出的证据文本判断，"
        "不能使用外部知识；同主题但没有直接证明主张核心事实时必须判不支持。"
    )
    user = (
        "请判断下面这段【剧情证据】是否直接支持这条【主张】。\n"
        "支持的标准：证据必须直接证明主张的核心事实、主体以及关键关系；"
        "仅背景相关、同人物/同主题、缺少关键断言，或与主张相反，都判不支持。\n\n"
        f"主张：{pair['claim']}\n\n"
        f"剧情证据：{pair['chunk_text']}\n\n"
        "只输出一个 JSON 对象，不要输出 Markdown。格式："
        '{"supports":true或false,"reason":"一句中文理由"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")


def call_one(
    pair: dict[str, Any],
    api_key: str,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": messages(pair),
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl_context())
    )
    last_error = ""
    terminal_code = ""
    for attempt in range(1, attempts + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError("no_choices")
            content = parse_content((choices[0].get("message") or {}).get("content"))
            parsed = parse_json_object(content)
            if parsed is None:
                raise RuntimeError("invalid_json")
            support = normalize_support(parsed.get("supports", parsed.get("verdict")), content)
            if support is None:
                raise RuntimeError("unparseable_support")
            return {
                "glm_verdict": "supported" if support else "unsupported",
                "glm_supports": support,
                "glm_reason": str(parsed.get("reason") or "")[:1000],
                "glm_raw_content": content[:4000],
                "glm_attempts": attempt,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            last_error = f"HTTP {exc.code}: {detail}"
            business_code = ""
            try:
                decoded = json.loads(detail)
                business_code = str((decoded.get("error") or {}).get("code") or decoded.get("code") or "")
            except (json.JSONDecodeError, AttributeError):
                pass
            terminal_code = business_code or str(exc.code)
            if attempt < attempts and (exc.code == 429 or business_code == "1301"):
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
                continue
            if business_code == "1301":
                return {
                    "glm_verdict": "skip",
                    "glm_supports": None,
                    "glm_reason": "content_filter_1301",
                    "glm_raw_content": detail,
                    "glm_attempts": attempt,
                }
            if exc.code in {400, 401, 402, 403}:
                break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    return {
        "glm_verdict": "error",
        "glm_supports": None,
        "glm_reason": (last_error or f"terminal:{terminal_code}")[:1000],
        "glm_raw_content": "",
        "glm_attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    api_key = os.environ.get("BIGMODEL_API_KEY", "")
    if not api_key:
        raise SystemExit("BIGMODEL_API_KEY is missing; invoke with 'source ~/.bigmodel_env'")
    pairs = load_jsonl(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if Path(args.output).exists():
        for row in load_jsonl(args.output):
            if row.get("pair_id"):
                existing[str(row["pair_id"])] = row
    pending = [p for p in pairs if str(p["pair_id"]) not in existing]
    lock = threading.Lock()
    print(f"GLM pairs={len(pairs)} existing={len(existing)} pending={len(pending)} workers={args.workers}", flush=True)

    def run(pair: dict[str, Any]) -> dict[str, Any]:
        result = call_one(pair, api_key, args.timeout, args.attempts)
        merged = dict(pair)
        merged.update(result)
        merged.update(
            {
                "glm_model": MODEL,
                "glm_reasoning_effort": "medium",
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return merged

    with open(args.output, "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(3, args.workers))) as pool:
            futures = [pool.submit(run, pair) for pair in pending]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                with lock:
                    handle.write(compact(row) + "\n")
                    handle.flush()
                    existing[str(row["pair_id"])] = row
                if count == 1 or count % 10 == 0 or count == len(futures):
                    print(f"GLM progress {count}/{len(futures)}", flush=True)
    print(
        json.dumps(
            {
                "output": args.output,
                "total": len(pairs),
                "written": len(existing),
                "supported": sum(r.get("glm_verdict") == "supported" for r in existing.values()),
                "unsupported": sum(r.get("glm_verdict") == "unsupported" for r in existing.values()),
                "skip": sum(r.get("glm_verdict") == "skip" for r in existing.values()),
                "error": sum(r.get("glm_verdict") == "error" for r in existing.values()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
