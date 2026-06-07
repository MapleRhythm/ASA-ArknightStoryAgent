#!/usr/bin/env python3
"""Audit and repair ShareGPT SFT records with an evidence-only API judge.

The script preserves the existing task schemas. It repairs the assistant JSON
for hypothesis/follow-up/conclusion records and writes a clean LLaMAFactory
dataset directory plus audit/rejected files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_api_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("API response does not contain a JSON object")


def compact_json_value(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        else:
            return stripped
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_existing_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        record_id = str(item.get("id") or "")
        if record_id:
            results[record_id] = item
    return results


def record_assistant_value(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    last = conversations[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("value") or "")


def record_user_value(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    first = conversations[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("value") or "")


def build_dataset_info(dataset_name: str, records: list[dict[str, Any]], splits: list[str]) -> dict[str, Any]:
    has_kto = any("kto_tag" in record for record in records)
    columns = {"messages": "conversations", "system": "system", "tools": "tools"}
    if has_kto:
        columns["kto_tag"] = "kto_tag"
    return {
        f"{dataset_name}_{split.removesuffix('.json')}": {
            "file_name": split,
            "formatting": "sharegpt",
            "columns": columns,
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
    }


@dataclass(frozen=True)
class ApiConfig:
    url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    retries: int
    retry_sleep: float


def call_chat_api(messages: list[dict[str, str]], cfg: ApiConfig) -> dict[str, Any]:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            req = Request(cfg.url, data=data, headers=headers, method="POST")
            with urlopen(req, timeout=cfg.timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            content = parsed["choices"][0]["message"]["content"]
            return extract_json_object(str(content))
        except (HTTPError, URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.retry_sleep * (attempt + 1))
    raise RuntimeError(f"API call failed after retries: {last_error}")


SYSTEM_PROMPT = """你是 SFT 数据集审计与修复器，服务于本地 RAG Agent 训练。
你必须只根据每条样本输入里给出的 question、hypothesis、evidence_brief/minirag_hints 和原 assistant 输出判断，不要使用自己的剧情知识补充事实。
目标是修复训练标签，使 4B 模型学习：干净检索线索、证据不足时继续检索、证据足够时基于证据回答、避免 unsupported claim 和过度检索。

输出必须是单个 JSON 对象：{"records":[...]}。
每个 records 元素字段：
- id: 原 record_id
- action: keep | fix | drop
- issue_tags: 字符串数组
- repaired_assistant: 修复后的 assistant JSON 对象；drop 时可为 null
- reason: 简短中文原因

通用规则：
1. 保持原任务 schema，不要新增训练时未要求的字段。
2. hypothesis/follow_up_hypothesis 任务只输出检索线索，不要回答问题。
3. 清理明显断词、残片、泛词，例如“都活不”“能理解亚叶面”“玛格达尔都”“支持”这类从句子截断来的实体/关键词。
4. conclusion_generation 必须输出 conclusion_v2：question,next_action,answer,missing_slots,clarification_question,follow_up_hypothesis。
5. answer_directly 只能包含 evidence_brief 明确支持或强支持的内容；证据没有出现的关键设备、人物、动机、结果不得保留。
6. 如果原输出 answer_directly 含 unsupported claim，能删改成证据支持答案就 fix；不能可靠回答就改成 retrieve_more 或 abstain。
7. 如果原输出 retrieve_more/abstain，但 evidence_brief 已包含直接回答问题的证据，应 fix 为 answer_directly。
8. retrieve_more 的 missing_slots 必须具体，follow_up_hypothesis 的 entities/keywords 必须短、干净、可检索。
9. 无法修复、输入输出 schema 坏、或证据严重不足且标签会误导训练时 action=drop。
"""


def build_batch_prompt(records: list[dict[str, Any]], max_input_chars: int, max_output_chars: int) -> str:
    compact_records: list[dict[str, Any]] = []
    for record in records:
        compact_records.append(
            {
                "id": str(record.get("id") or ""),
                "task_type": str(record.get("task_type") or ""),
                "system": truncate_text(str(record.get("system") or ""), 500),
                "user_input": truncate_text(record_user_value(record), max_input_chars),
                "assistant_output": truncate_text(record_assistant_value(record), max_output_chars),
            }
        )
    return (
        "请逐条审计并修复以下 SFT 样本。返回 JSON，不要 markdown。\n\n"
        + json.dumps({"records": compact_records}, ensure_ascii=False, indent=2)
    )


def apply_repair(record: dict[str, Any], audit: dict[str, Any], model: str) -> dict[str, Any] | None:
    action = str(audit.get("action") or "keep").strip()
    if action == "drop":
        return None
    repaired = json.loads(json.dumps(record, ensure_ascii=False))
    conversations = repaired.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    assistant = conversations[-1]
    if not isinstance(assistant, dict):
        return None
    if action == "fix":
        repaired_value = audit.get("repaired_assistant")
        if repaired_value is None:
            return None
        assistant["value"] = compact_json_value(repaired_value)
    meta = repaired.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["api_sft_qc_v1"] = {
            "action": action,
            "issue_tags": audit.get("issue_tags") or [],
            "reason": str(audit.get("reason") or ""),
            "model": model,
        }
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an API model to QC/repair ShareGPT SFT records.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--splits", nargs="+", default=["train.json", "val.json"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=9000)
    parser.add_argument("--max-output-chars", type=int, default=2500)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {}
    all_records: list[tuple[str, dict[str, Any]]] = []
    for split in args.splits:
        path = input_dir / split
        if not path.exists():
            continue
        records = read_json(path)
        if not isinstance(records, list):
            raise SystemExit(f"{path} is not a JSON list")
        split_records[split] = [record for record in records if isinstance(record, dict)]
        for record in split_records[split]:
            all_records.append((split, record))

    if args.shuffle_seed:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(all_records)
    if args.limit_records is not None:
        all_records = all_records[: args.limit_records]

    summary_input = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "splits": {split: len(records) for split, records in split_records.items()},
        "records_to_score": len(all_records),
        "batch_size": args.batch_size,
        "model": args.model,
        "max_input_chars": args.max_input_chars,
        "max_output_chars": args.max_output_chars,
    }
    write_json(output_dir / "input_summary.json", summary_input)
    print(json.dumps(summary_input, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")
    cfg = ApiConfig(
        url=normalize_api_url(args.api_base),
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    audit_path = output_dir / "audit.jsonl"
    existing = load_existing_audit(audit_path)
    pending = [(split, record) for split, record in all_records if str(record.get("id") or "") not in existing]
    print(f"[resume] existing={len(existing)} pending={len(pending)}", flush=True)

    for start in range(0, len(pending), args.batch_size):
        batch_pairs = pending[start : start + args.batch_size]
        batch = [record for _, record in batch_pairs]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(batch, args.max_input_chars, args.max_output_chars)},
        ]
        response = call_chat_api(messages, cfg)
        items = response.get("records")
        if not isinstance(items, list):
            raise RuntimeError(f"API response missing records array: {response}")
        by_id = {str(item.get("id") or ""): item for item in items if isinstance(item, dict)}
        for split, record in batch_pairs:
            record_id = str(record.get("id") or "")
            audit = by_id.get(record_id) or {
                "id": record_id,
                "action": "drop",
                "issue_tags": ["missing_api_verdict"],
                "repaired_assistant": None,
                "reason": "API 未返回该样本的判定。",
            }
            audit["_split"] = split
            append_jsonl(audit_path, audit)
        done = min(start + args.batch_size, len(pending))
        print(f"[batch] {done}/{len(pending)}", flush=True)

    audits = load_existing_audit(audit_path)
    stats: Counter[str] = Counter()
    issue_stats: Counter[str] = Counter()
    all_fixed_records: list[dict[str, Any]] = []
    rejected_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_records}
    for split, records in split_records.items():
        fixed_records: list[dict[str, Any]] = []
        for record in records:
            record_id = str(record.get("id") or "")
            audit = audits.get(record_id)
            if audit is None:
                fixed = json.loads(json.dumps(record, ensure_ascii=False))
                fixed_records.append(fixed)
                all_fixed_records.append(fixed)
                stats["unscored_kept_original"] += 1
                continue
            stats[str(audit.get("action") or "keep")] += 1
            for tag in audit.get("issue_tags") or []:
                issue_stats[str(tag)] += 1
            fixed = apply_repair(record, audit, args.model)
            if fixed is None:
                rejected = json.loads(json.dumps(record, ensure_ascii=False))
                rejected.setdefault("meta", {})["api_sft_qc_v1"] = {
                    "action": "drop",
                    "issue_tags": audit.get("issue_tags") or [],
                    "reason": str(audit.get("reason") or ""),
                    "model": args.model,
                }
                rejected_by_split[split].append(rejected)
            else:
                fixed_records.append(fixed)
                all_fixed_records.append(fixed)
        write_json(output_dir / split, fixed_records)
        if rejected_by_split[split]:
            write_json(output_dir / f"rejected_{split}", rejected_by_split[split])

    write_json(output_dir / "dataset_info.json", build_dataset_info(args.dataset_name, all_fixed_records, list(split_records)))
    summary = {
        **summary_input,
        "output_splits": {
            split: len(read_json(output_dir / split))
            for split in split_records
            if (output_dir / split).exists()
        },
        "rejected_splits": {split: len(items) for split, items in rejected_by_split.items()},
        "actions": dict(stats),
        "issue_tags": dict(issue_stats.most_common()),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
