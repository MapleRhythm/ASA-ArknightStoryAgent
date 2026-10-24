#!/usr/bin/env python3
"""Label full candidate pools for query-only answerability via an OpenAI API."""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from query_answerability_teacher_protocol import (
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    build_messages,
    build_result_record,
    parse_json_content,
    record_key,
    safe_int,
    validate_input_record,
    validate_judgement,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TeacherError(Exception):
    def __init__(
        self, message: str, *, attempts: int, detail: str = "", terminal: bool = False
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
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = reasoning_effort
    return body


def add_validation_repair(
    messages: list[dict[str, str]], raw_content: str, error: Exception
) -> list[dict[str, str]]:
    # The original response can be large.  Preserve the beginning and end so
    # the model can repair both schema fields and a truncated/incorrect tail.
    if len(raw_content) > 30000:
        quoted = raw_content[:15000] + "\n...<middle omitted>...\n" + raw_content[-15000:]
    else:
        quoted = raw_content
    return [
        *messages,
        {"role": "assistant", "content": quoted},
        {
            "role": "user",
            "content": (
                "上一个 JSON 未通过确定性校验："
                f"{str(error)[:1200]}。只修正这个错误，重新输出完整 JSON；"
                "保持每条候选的原判断，不要解释。"
            ),
        },
    ]


def call_teacher(
    record: dict[str, Any],
    *,
    api_key: str,
    endpoint: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
    timeout: int,
    retries: int,
    expected_candidates: int | None,
) -> dict[str, Any]:
    validate_input_record(record, expected_candidates)
    messages = build_messages(record, expected_candidates)
    active_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        body = build_request_body(
            active_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
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
                raise ValueError(f"response truncated: finish_reason={finish_reason}")
            raw_content = str(choice.get("message", {}).get("content") or "")
            judgement = validate_judgement(record, parse_json_content(raw_content))
            return build_result_record(
                record,
                judgement=judgement,
                provider=provider,
                model=model,
                usage=payload.get("usage") or {},
                finish_reason=finish_reason,
            )
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:4000]
            last_error = RuntimeError(f"HTTP {exc.code}: {body_text}")
            try:
                business_code = str((json.loads(body_text).get("error") or {}).get("code") or "")
            except (json.JSONDecodeError, AttributeError):
                business_code = ""
            if exc.code in {400, 401, 402, 403} or business_code == "1113":
                raise TeacherError(
                    f"{provider} terminal API error",
                    attempts=attempt,
                    detail=str(last_error),
                    terminal=True,
                ) from last_error
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if raw_content and attempt < retries:
                active_messages = add_validation_repair(messages, raw_content, exc)
        if attempt < retries:
            time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    raise TeacherError(
        f"{provider} answerability labelling failed",
        attempts=retries,
        detail=str(last_error)[:4000],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=5)
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
    expected_candidates = None if args.allow_non120 else args.expected_candidates
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {args.api_key_env}")

    records = read_jsonl(input_path)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]
    for record in records:
        validate_input_record(record, expected_candidates)

    completed: set[str] = set()
    if output_path.exists():
        completed = {record_key(record) for record in read_jsonl(output_path)}
    pending = [record for record in records if record_key(record) not in completed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempted = succeeded = failed = 0
    terminal_error: TeacherError | None = None

    with output_path.open("a", encoding="utf-8") as output_handle, failures_path.open(
        "a", encoding="utf-8"
    ) as failure_handle:
        pending_iter = iter(pending)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as executor:
            futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

            def submit_next() -> bool:
                nonlocal attempted
                try:
                    record = next(pending_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    call_teacher,
                    record,
                    api_key=api_key,
                    endpoint=args.endpoint,
                    provider=args.provider,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout,
                    retries=args.retries,
                    expected_candidates=expected_candidates,
                )
                futures[future] = record
                attempted += 1
                return True

            for _ in range(max(args.max_workers, 1)):
                if not submit_next():
                    break
            finished = 0
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    record = futures.pop(future)
                    finished += 1
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        failure = {
                            "source_name": record.get("source_name"),
                            "query": record.get("query"),
                            "candidate_count": len(record.get("candidates") or []),
                            "teacher_protocol": PROTOCOL_VERSION,
                            "teacher_prompt_version": PROMPT_VERSION,
                            "teacher_provider": args.provider,
                            "teacher_model": args.model,
                            "retry_count": exc.attempts if isinstance(exc, TeacherError) else 0,
                            "terminal": bool(isinstance(exc, TeacherError) and exc.terminal),
                            "error": str(exc)[:5000],
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_handle.flush()
                        print(f"[failed] {record.get('query')}: {exc}", flush=True)
                        if isinstance(exc, TeacherError) and exc.terminal:
                            terminal_error = exc
                    else:
                        succeeded += 1
                        output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        usage = result.get("usage") or {}
                        for key in total_usage:
                            total_usage[key] += safe_int(usage.get(key), 0)
                        labels: dict[str, int] = {}
                        for item in result["teacher"]["ranking"]:
                            label = str(item["label"])
                            labels[label] = labels.get(label, 0) + 1
                        print(
                            f"[{finished}/{len(pending)}] labels={labels} "
                            f"tokens={usage.get('total_tokens', 0)} query={str(result['query'])[:36]}",
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
        "input_records": len(records),
        "previously_completed": len(completed),
        "pending_at_start": len(pending),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "stopped_on_terminal_error": terminal_error is not None,
        "usage": total_usage,
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "protocol": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "expected_candidates": expected_candidates,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
