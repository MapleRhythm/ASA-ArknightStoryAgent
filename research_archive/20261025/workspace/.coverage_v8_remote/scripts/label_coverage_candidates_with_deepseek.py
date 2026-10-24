#!/usr/bin/env python3
"""Label ASA full-candidate pools with DeepSeek, without gold evidence leakage.

Every request contains the query, reference answer, and all candidate passages.
Gold evidence IDs, old relevance labels, retrieval ranks, and retrieval scores
are excluded from the prompt.  Successful records are appended so interrupted
jobs can safely resume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
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
    """Teacher failure with retry metadata and a bounded diagnostic."""

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

    def __str__(self) -> str:
        return f"{self.message}: {self.detail}" if self.detail else self.message


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
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def _add_validation_repair(
    messages: list[dict[str, str]], raw_content: str, error: Exception
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": raw_content[:16000]},
        {
            "role": "user",
            "content": (
                "上一个 JSON 未通过确定性校验："
                f"{str(error)[:1000]}。请严格保持任务与 schema，只修正错误并重新输出完整 JSON；"
                "不要解释。"
            ),
        },
    ]


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
) -> tuple[Any, dict[str, Any], str]:
    active_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        body = build_request_body(active_messages, model=model, max_tokens=max_tokens)
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw_content = ""
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
            raw_content = str(choice.get("message", {}).get("content") or "")
            parsed = parse_json_content(raw_content)
            validated = validator(parsed)
            return validated, payload.get("usage") or {}, finish_reason
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:4000]
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if exc.code in {400, 401, 402, 403}:
                raise TeacherError(
                    f"DeepSeek terminal API error during {phase}",
                    attempts=attempt,
                    detail=str(last_error),
                    terminal=True,
                ) from last_error
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if raw_content and attempt < retries:
                active_messages = _add_validation_repair(messages, raw_content, exc)
        if attempt < retries:
            time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    raise TeacherError(
        f"DeepSeek {phase} phase failed",
        attempts=retries,
        detail=str(last_error)[:4000],
    )


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
    expected_candidates: int | None,
    frozen_claims: list[dict[str, str]] | None = None,
    claim_sink: Any | None = None,
) -> dict[str, Any]:
    validate_input_record(record, expected_candidates)
    if frozen_claims is None:
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
        )
        if claim_sink is not None:
            claim_sink(claims, claim_usage)
    else:
        claims = frozen_claims
        claim_usage = {}
        claim_finish_reason = "cached"
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
    )
    combined_usage = {
        key: safe_int(claim_usage.get(key), 0) + safe_int(ranking_usage.get(key), 0)
        for key in {"prompt_tokens", "completion_tokens", "total_tokens"}
    }
    result = build_result_record(
        record,
        judgement=judgement,
        provider="deepseek",
        model=model,
        usage=combined_usage,
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
    parser.add_argument(
        "--claims-output",
        type=Path,
        help="Resumable JSONL cache for the candidate-independent claim decomposition.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--claim-max-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--allow-non120", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    failures_path = (
        resolve(args.failures_output)
        if args.failures_output
        else output_path.with_suffix(".failures.jsonl")
    )
    claims_path = (
        resolve(args.claims_output)
        if args.claims_output
        else output_path.with_suffix(".claims.jsonl")
    )
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
    claim_cache: dict[str, list[dict[str, str]]] = {}
    if claims_path.exists():
        for cached in read_jsonl(claims_path):
            raw_claims = cached.get("claims")
            if isinstance(raw_claims, list):
                claim_cache[record_key(cached)] = raw_claims

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failure_count = 0
    attempted_count = 0
    succeeded_count = 0
    terminal_error: TeacherError | None = None

    # Submit at most max_workers records at a time so a terminal account error
    # does not enqueue hundreds of doomed calls.
    with output_path.open("a", encoding="utf-8") as output_handle, failures_path.open(
        "a", encoding="utf-8"
    ) as failure_handle, claims_path.open("a", encoding="utf-8") as claims_handle:
        pending_iter = iter(pending)
        claim_write_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

            def submit_next() -> bool:
                nonlocal attempted_count
                try:
                    record = next(pending_iter)
                except StopIteration:
                    return False
                key = record_key(record)

                def persist_claims(
                    claims: list[dict[str, str]], usage: dict[str, Any]
                ) -> None:
                    with claim_write_lock:
                        if key in claim_cache:
                            return
                        claims_handle.write(
                            json.dumps(
                                {
                                    "source_name": record.get("source_name"),
                                    "query": record.get("query"),
                                    "answer": record.get("answer"),
                                    "teacher_protocol": PROTOCOL_VERSION,
                                    "teacher_prompt_version": PROMPT_VERSION,
                                    "teacher_model": args.model,
                                    "claims": claims,
                                    "usage": usage,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        claims_handle.flush()
                        claim_cache[key] = claims

                future = executor.submit(
                    call_teacher,
                    record,
                    api_key=api_key,
                    endpoint=args.endpoint,
                    model=args.model,
                    claim_max_tokens=args.claim_max_tokens,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    expected_candidates=expected_candidates,
                    frozen_claims=claim_cache.get(key),
                    claim_sink=persist_claims,
                )
                futures[future] = record
                attempted_count += 1
                return True

            for _ in range(max(args.max_workers, 1)):
                if not submit_next():
                    break

            completed_count = 0
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    record = futures.pop(future)
                    completed_count += 1
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - persist diagnostic
                        failure_count += 1
                        attempts = exc.attempts if isinstance(exc, TeacherError) else 0
                        failure = {
                            "source_name": record.get("source_name"),
                            "query": record.get("query"),
                            "candidate_count": len(record.get("candidates") or []),
                            "teacher_protocol": PROTOCOL_VERSION,
                            "teacher_prompt_version": PROMPT_VERSION,
                            "teacher_provider": "deepseek",
                            "teacher_model": args.model,
                            "retry_count": attempts,
                            "terminal": bool(
                                isinstance(exc, TeacherError) and exc.terminal
                            ),
                            "error": str(exc)[:5000],
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_handle.flush()
                        print(f"[failed] {record.get('query')}: {exc}", flush=True)
                        if isinstance(exc, TeacherError) and exc.terminal:
                            terminal_error = exc
                    else:
                        succeeded_count += 1
                        key = record_key(result)
                        output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        usage = result.get("usage") or {}
                        for key in total_usage:
                            total_usage[key] += safe_int(usage.get(key), 0)
                        print(
                            f"[{completed_count}/{len(pending)}] "
                            f"candidates={len(result['candidates'])} "
                            f"tokens={usage.get('total_tokens', 0)} "
                            f"query={str(result['query'])[:36]}",
                            flush=True,
                        )
                    if terminal_error is None:
                        submit_next()
                if terminal_error is not None:
                    for future in futures:
                        future.cancel()
                    break

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "failures_output": str(failures_path),
        "claims_output": str(claims_path),
        "input_records": len(records),
        "previously_completed": len(completed),
        "attempted": attempted_count,
        "succeeded": succeeded_count,
        "failed": failure_count,
        "stopped_on_terminal_error": terminal_error is not None,
        "usage": total_usage,
        "provider": "deepseek",
        "model": args.model,
        "protocol": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "expected_candidates": expected_candidates,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
