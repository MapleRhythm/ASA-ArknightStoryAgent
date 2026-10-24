#!/usr/bin/env python3
"""Use GLM as a gold-free full-candidate reranking teacher for ASA.

This script shares the answer_support_v2 protocol with the DeepSeek labeller.
It sends the complete candidate pool but never sends gold evidence, old
relevance labels, retrieval ranks, or retrieval scores.

Expected use on the 248 mounted project:

    export BIGMODEL_API_KEY=...
    python scripts/label_coverage_candidates_with_glm.py \
      --input data/processed/evidence_chain_reranker/rank_v8_coverage/train_candidates.jsonl \
      --output data/processed/evidence_chain_reranker/rank_v8_coverage/glm_5_3/train_teacher.jsonl

Do not put API keys in command-line arguments; pass them via the environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from answer_support_teacher_protocol import (
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    build_claim_messages,
    build_ranking_messages,
    build_result_record,
    parse_json_content,
    record_key,
    safe_int,
    validate_claim_judgement,
    validate_input_record,
    validate_ranking_judgement,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TeacherError(Exception):
    """Error with a bounded, JSON-serialisable diagnostic payload."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        detail: str = "",
        terminal: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.attempts = attempts
        self.detail = detail
        self.terminal = terminal

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.detail:
            return f"{self.message}: {self.detail}"
        return self.message


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_request_body(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    use_response_format: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if use_response_format:
        body["response_format"] = {"type": "json_object"}
    return body


def call_json_phase(
    messages: list[dict[str, str]],
    *,
    validator: Any,
    phase: str,
    api_key: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    temperature: float,
    top_p: float | None,
    use_response_format: bool,
) -> tuple[Any, dict[str, Any], str]:
    body = build_request_body(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        use_response_format=use_response_format,
    )
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_detail = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise ValueError(f"API response has no choices: {json.dumps(payload)[:1200]}")
            choice = choices[0]
            finish_reason = str(choice.get("finish_reason") or "")
            if finish_reason in {"length", "max_tokens"}:
                raise ValueError(f"{phase} response truncated: finish_reason={finish_reason}")
            content = str(choice.get("message", {}).get("content") or "")
            return validator(parse_json_content(content)), payload.get("usage") or {}, finish_reason
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:4000]
            last_detail = f"HTTP {exc.code}: {error_body}"
            if exc.code in {400, 401, 402, 403}:
                raise TeacherError(
                    f"GLM terminal API error during {phase}",
                    attempts=attempt,
                    detail=last_detail,
                    terminal=True,
                ) from exc
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_detail = str(exc)[:4000]

        if attempt < retries:
            time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))

    raise TeacherError(f"GLM {phase} phase failed", attempts=retries, detail=last_detail)


def call_teacher(
    record: dict[str, Any],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    claim_max_tokens: int,
    max_tokens: int,
    timeout: int,
    retries: int,
    temperature: float,
    top_p: float | None,
    expected_candidates: int | None,
    use_response_format: bool,
) -> dict[str, Any]:
    validate_input_record(record, expected_candidates)
    claims, claim_usage, claim_finish_reason = call_json_phase(
        build_claim_messages(record),
        validator=validate_claim_judgement,
        phase="claim_decomposition",
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        max_tokens=claim_max_tokens,
        timeout=timeout,
        retries=retries,
        temperature=temperature,
        top_p=top_p,
        use_response_format=use_response_format,
    )
    judgement, ranking_usage, ranking_finish_reason = call_json_phase(
        build_ranking_messages(record, claims, expected_candidates),
        validator=lambda raw: validate_ranking_judgement(record, claims, raw),
        phase="candidate_ranking",
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        temperature=temperature,
        top_p=top_p,
        use_response_format=use_response_format,
    )
    usage = {
        key: safe_int(claim_usage.get(key), 0) + safe_int(ranking_usage.get(key), 0)
        for key in {"prompt_tokens", "completion_tokens", "total_tokens"}
    }
    result = build_result_record(
        record,
        judgement=judgement,
        provider="bigmodel",
        model=model,
        usage=usage,
        finish_reason=ranking_finish_reason,
    )
    result["phase_usage"] = {
        "claim_decomposition": claim_usage,
        "candidate_ranking": ranking_usage,
    }
    result["phase_finish_reason"] = {
        "claim_decomposition": claim_finish_reason,
        "candidate_ranking": ranking_finish_reason,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--endpoint", default="https://open.bigmodel.cn/api/paas/v4/chat/completions")
    parser.add_argument("--api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--claim-max-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--allow-non120", action="store_true")
    parser.add_argument("--disable-response-format", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    failures_path = resolve(args.failures_output) if args.failures_output else output_path.with_suffix(".failures.jsonl")
    expected_candidates = None if args.allow_non120 else args.expected_candidates

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    records = read_jsonl(input_path)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    completed: set[str] = set()
    if output_path.exists():
        completed = {record_key(record) for record in read_jsonl(output_path)}
    pending = [record for record in records if record_key(record) not in completed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failure_count = 0

    with output_path.open("a", encoding="utf-8") as output_handle, failures_path.open(
        "a", encoding="utf-8"
    ) as failure_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_record = {
                executor.submit(
                    call_teacher,
                    record,
                    api_key=api_key,
                    endpoint=args.endpoint,
                    model=args.model,
                    claim_max_tokens=args.claim_max_tokens,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    expected_candidates=expected_candidates,
                    use_response_format=not args.disable_response_format,
                ): record
                for record in pending
            }
            for completed_count, future in enumerate(
                concurrent.futures.as_completed(future_to_record), start=1
            ):
                record = future_to_record[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - continue other records
                    failure_count += 1
                    attempts = exc.attempts if isinstance(exc, TeacherError) else 0
                    failure = {
                        "source_name": record.get("source_name"),
                        "query": record.get("query"),
                        "query_type": record.get("query_type"),
                        "candidate_count": len(record.get("candidates") or []),
                        "teacher_provider": "bigmodel",
                        "teacher_model": args.model,
                        "retry_count": attempts,
                        "error": str(exc)[:5000],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_handle.flush()
                    print(f"[failed] {record.get('query')}: {exc}", flush=True)
                    continue

                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                usage = result.get("usage") or {}
                for key in total_usage:
                    total_usage[key] += safe_int(usage.get(key), 0)
                print(
                    f"[{completed_count}/{len(pending)}] candidates={len(result['candidates'])} "
                    f"tokens={usage.get('total_tokens', 0)} query={str(result['query'])[:36]}",
                    flush=True,
                )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "failures_output": str(failures_path),
        "input_records": len(records),
        "previously_completed": len(completed),
        "attempted": len(pending),
        "succeeded": len(pending) - failure_count,
        "failed": failure_count,
        "usage": total_usage,
        "provider": "bigmodel",
        "model": args.model,
        "protocol": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "endpoint": args.endpoint,
        "expected_candidates": expected_candidates,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
