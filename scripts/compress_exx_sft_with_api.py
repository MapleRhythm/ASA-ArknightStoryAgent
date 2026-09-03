#!/usr/bin/env python3
"""Compress long Exx SFT targets into the minimum sufficient fact set.

The source dataset is immutable.  Only ``answer_directly`` rows whose target
contains more than ``--max-facts`` facts are sent to a teacher API.  The
teacher sees the complete question, round, evidence and current target, and
must remove redundant/irrelevant facts without inventing facts or evidence
IDs.  Every response is validated locally against the current Exx protocol.

The run is resumable through ``labels.jsonl``.  A failed API call, malformed
response, or locally invalid repair copies the original row unchanged, so a
partial network outage cannot silently delete training data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

SOURCE_ROOT_CANDIDATES = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "release" / "ASA-ArknightStoryAgent" / "src",
)
SOURCE_ROOT = next(
    (
        candidate
        for candidate in SOURCE_ROOT_CANDIDATES
        if (candidate / "asa_arknight_story_agent").is_dir()
    ),
    SOURCE_ROOT_CANDIDATES[0],
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asa_arknight_story_agent.inference.generation.exx_prompt import (  # noqa: E402
    EXX_PROTOCOL,
    EXX_SYSTEM_PROMPT,
)


ALLOWED_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    line = compact_json(value) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def assistant_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    conversations = row.get("conversations") or []
    if not conversations:
        return None
    try:
        value = json.loads(str(conversations[-1].get("value") or ""))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prompt_text(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    return str(conversations[0].get("value") or "") if conversations else ""


def visible_evidence_ids(prompt: str) -> set[str]:
    import re

    return set(re.findall(r"^\[(E\d+)\]\s*$", prompt, re.MULTILINE))


def validate_payload(payload: dict[str, Any] | None, prompt: str) -> list[str]:
    if not isinstance(payload, dict):
        return ["invalid_json_object"]
    if payload.get("next_action") not in ALLOWED_ACTIONS:
        return ["invalid_action"]
    visible = visible_evidence_ids(prompt)
    action = payload["next_action"]
    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            return ["answer_top_schema"]
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return ["answer_fact_count"]
        seen: set[str] = set()
        for index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                return [f"fact_{index}_schema"]
            text = str(fact.get("fact") or "").strip()
            ids = fact.get("evidence_ids")
            normalized = "".join(ch for ch in text.lower() if ch.isalnum() or "\u3400" <= ch <= "\u9fff")
            if not text or not normalized:
                return [f"fact_{index}_empty"]
            if normalized in seen:
                return [f"fact_{index}_duplicate"]
            seen.add(normalized)
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or len({str(item) for item in ids}) != len(ids)
                or any(str(item) not in visible for item in ids)
            ):
                return [f"fact_{index}_evidence_ids"]
        return []
    if action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            return ["retrieve_top_schema"]
        follow = payload.get("follow_up_hypothesis")
        if not isinstance(follow, dict) or not str(follow.get("question") or "").strip():
            return ["retrieve_follow_up"]
        return []
    if set(payload) != {"next_action", "reason"} or not str(payload.get("reason") or "").strip():
        return ["abstain_schema"]
    return []


def parse_json_object(value: Any) -> dict[str, Any] | None:
    raw = str(value or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def normalize_api_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


SYSTEM_PROMPT = """你是 Exx grounded_action_exx_v1 训练数据的严格压缩器。
你的任务是把已有 answer_directly 标签压缩成“完整回答问题所需的最少充分原子事实”。
只能使用输入中当前可见 evidence 的内容，不能使用游戏常识，不能补写证据外事实。
只输出一个 JSON 对象，不要 markdown、解释或思维过程。"""


def build_user_prompt(row: dict[str, Any], *, max_facts: int) -> str:
    payload = assistant_payload(row)
    if payload is None:
        raise ValueError("row has no valid assistant payload")
    return "\n".join(
        (
            f"目标协议：{EXX_PROTOCOL}",
            f"最大事实条数：{max_facts}（只有确实能覆盖问题核心且被证据支持时才保留）",
            "硬规则：",
            "1. 保持 next_action 为 answer_directly。",
            "2. 删除重复、近义改写、只回答枝节或与问题无关的事实；不要为了凑条数输出事实。",
            "3. 不得删除回答问题核心所必需的不同事实；不要把互相矛盾的证据强行合并。",
            "4. 每个 fact 必须是简短、可核验的原子陈述，evidence_ids 只能从当前 prompt 的 E 编号中选择。",
            "5. 不复制引文，不增加 quote/final_answer/inferred_facts/evidence_refs 等字段。",
            "6. 如果原标签事实确实超过上限且每条都不可安全删除，仍输出最小充分集合，不得截断证据正文。",
            "输入 user prompt（证据完整，不得假设隐藏证据）：",
            prompt_text(row),
            "当前 assistant 标签：",
            compact_json(payload),
            "只输出修复后的 answer_directly JSON。",
        )
    )


def call_api(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {args.api_key_env}")
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, max_facts=args.max_facts)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        normalize_api_url(args.endpoint),
        data=encoded,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            with urlopen(request, timeout=args.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            content = value["choices"][0]["message"]["content"]
            parsed = parse_json_object(content)
            if parsed is None:
                raise ValueError("API response is not a JSON object")
            return parsed
        except (HTTPError, URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"API call failed: {type(last_error).__name__}: {last_error}")


def copy_row(row: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(row)


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("id"):
            completed[str(value["id"])] = value
    return completed


def process_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rid = str(row.get("id") or "")
    payload = assistant_payload(row)
    if (
        row.get("task_type") != "grounded_action_generation"
        or not isinstance(payload, dict)
        or payload.get("next_action") != "answer_directly"
        or not isinstance(payload.get("supported_facts"), list)
        or len(payload["supported_facts"]) <= args.max_facts
    ):
        return {"id": rid, "status": "copied", "row": copy_row(row)}
    original_count = len(payload["supported_facts"])
    try:
        candidate = call_api(row, args)
        issues = validate_payload(candidate, prompt_text(row))
        if issues:
            raise ValueError("invalid compressed payload: " + ",".join(issues))
        if candidate.get("next_action") != "answer_directly":
            raise ValueError("compressed action changed")
        facts = candidate.get("supported_facts") or []
        if len(facts) > args.max_facts:
            raise ValueError(f"compressed fact count remains {len(facts)}>{args.max_facts}")
        repaired = copy_row(row)
        repaired["conversations"][-1]["value"] = compact_json(candidate)
        meta = repaired.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {"source_meta": meta}
        if not isinstance(meta, dict):
            meta = {}
        meta["minimal_fact_compression"] = {
            "provider": args.provider,
            "model": args.model,
            "original_fact_count": original_count,
            "compressed_fact_count": len(facts),
            "policy": "minimum_sufficient_nonredundant_facts",
        }
        repaired["meta"] = compact_json(meta)
        return {
            "id": rid,
            "status": "compressed",
            "original_fact_count": original_count,
            "compressed_fact_count": len(facts),
            "row": repaired,
        }
    except Exception as exc:  # noqa: BLE001 - preserve source row on any failure.
        return {
            "id": rid,
            "status": "fallback_copy",
            "original_fact_count": original_count,
            "error": f"{type(exc).__name__}:{exc}",
            "row": copy_row(row),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train.json,val.json,test.json")
    parser.add_argument("--max-facts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--endpoint", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--provider", default="deepseek_flash_minimal_compression")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    args = parser.parse_args()
    if args.max_facts < 1:
        parser.error("--max-facts must be positive")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be in [1,16]")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = input_dir / split
        rows_by_split[split] = read_json(path) if path.exists() else []

    all_rows: list[dict[str, Any]] = []
    for split in splits:
        all_rows.extend(rows_by_split[split])
    random.Random(args.seed).shuffle(all_rows)
    pending_rows = all_rows if args.limit <= 0 else all_rows[: args.limit]
    checkpoint_path = output_dir / "labels.jsonl"
    completed = load_completed(checkpoint_path) if args.resume else {}
    pending_rows = [row for row in pending_rows if str(row.get("id") or "") not in completed]

    lock = threading.Lock()
    stats: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, row, args): row for row in pending_rows}
        for future in as_completed(futures):
            result = future.result()
            rid = str(result.get("id") or "")
            completed[rid] = result
            stats[str(result.get("status") or "unknown")] += 1
            if result.get("compressed_fact_count") is not None:
                stats[f"compressed_{result['original_fact_count']}_to_{result['compressed_fact_count']}"] += 1
            append_jsonl(checkpoint_path, result, lock)
            done = sum(stats.values())
            if done % 10 == 0 or done == len(pending_rows):
                print(
                    json.dumps(
                        {"done_this_run": done, "pending_this_run": len(pending_rows), "stats": dict(stats)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    # Materialize in original split/source order, never in completion order.
    output_counts: dict[str, int] = {}
    for split in splits:
        materialized: list[dict[str, Any]] = []
        for row in rows_by_split[split]:
            rid = str(row.get("id") or "")
            result = completed.get(rid)
            materialized.append(copy_row(result["row"]) if result and result.get("row") else copy_row(row))
        write_json(output_dir / split, materialized)
        output_counts[split] = len(materialized)

    write_json(
        output_dir / "dataset_info.json",
        {
            f"exx_grounding_v3_sft_minimal_{split.removesuffix('.json')}": {
                "file_name": split,
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "observation_tag": "observation",
                    "function_tag": "function_call",
                },
            }
            for split in splits
        },
    )
    audit = {
        "protocol": EXX_PROTOCOL,
        "source_dir": str(input_dir),
        "source_sha256": {
            split: sha256_file(input_dir / split)
            for split in splits
            if (input_dir / split).is_file()
        },
        "output_counts": output_counts,
        "max_facts": args.max_facts,
        "provider": args.provider,
        "model": args.model,
        "workers": args.workers,
        "seed": args.seed,
        "resume": args.resume,
        "stats": dict(sorted(stats.items())),
        "checkpoint": str(checkpoint_path),
        "policy": {
            "source_immutable": True,
            "complete_evidence_preserved": True,
            "api_failure_copies_source": True,
        },
    }
    write_json(output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
