#!/usr/bin/env python3
"""Audit and repair grounded_action_v1 ShareGPT SFT data with an API judge.

This targets the latest compact grounded-action schema:
- answer_directly: next_action, supported_facts, inferred_facts, final_answer
- retrieve_more: next_action, follow_up_hypothesis
- abstain: next_action, final_answer

The API is called in batches, but records are still locally validated after
repair so long quotes and unsupported quote spans cannot silently pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
ALLOWED_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
WHITESPACE_RE = re.compile(r"\s+")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_for_match(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text or ""))


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]


def normalize_api_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("API response does not contain a JSON object")


def parse_assistant(record: dict[str, Any]) -> dict[str, Any] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    raw = str((conversations[-1] or {}).get("value") or "")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def user_prompt(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    return str((conversations[0] or {}).get("value") or "")


def rewrite_prompt_quote_rules(text: str) -> str:
    latest_rule = (
        "规则：只能使用证据；单条quote必须原文精确摘录，推荐20-60字，硬上限80字；"
        "每个fact最多2条quote且总长<=160字；supported_facts最多6条，所有quote总长最好<=400字；"
        "不要写无证据支持的事实；证据不足才retrieve_more。"
    )
    output = str(text or "")
    output = output.replace(
        "规则：只能使用证据；quote必须原文精确摘录且<=180字；不要写无证据支持的事实；证据不足才retrieve_more。",
        latest_rule,
    )
    output = output.replace(
        "规则：只能使用证据；quote必须原文精确摘录且<=80字；不要写无证据支持的事实；证据不足才retrieve_more。",
        latest_rule,
    )
    output = re.sub(
        r"quote必须原文精确摘录且<=\d+字",
        "单条quote必须原文精确摘录，推荐20-60字，硬上限80字",
        output,
    )
    return output


def clone_with_rewritten_prompt(record: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(record)
    conversations = cloned.get("conversations")
    if isinstance(conversations, list) and conversations:
        conversations[0]["value"] = rewrite_prompt_quote_rules(str(conversations[0].get("value") or ""))
    return cloned


def validate_payload(
    payload: dict[str, Any] | None,
    *,
    evidence_text: str,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_fact_quote_total_chars: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["assistant_json_invalid"]
    action = str(payload.get("next_action") or "").strip()
    if action not in ALLOWED_ACTIONS:
        return [f"invalid_action:{action or '<empty>'}"]
    if action == "retrieve_more":
        follow = payload.get("follow_up_hypothesis")
        if not isinstance(follow, dict):
            return ["retrieve_more_missing_follow_up_hypothesis"]
        if not (follow.get("entities") or follow.get("keywords")):
            return ["retrieve_more_follow_up_has_no_entities_or_keywords"]
        return []
    if action == "abstain":
        return [] if str(payload.get("final_answer") or "").strip() else ["abstain_missing_final_answer"]

    issues: list[str] = []
    supported = payload.get("supported_facts")
    if not isinstance(supported, list) or not supported:
        return ["answer_missing_supported_facts"]
    if max_supported_facts > 0 and len(supported) > max_supported_facts:
        issues.append(f"too_many_supported_facts:{len(supported)}>{max_supported_facts}")

    evidence_norm = normalize_for_match(evidence_text)
    all_quote_total = 0
    for fact_index, fact in enumerate(supported, start=1):
        if not isinstance(fact, dict):
            issues.append(f"fact_{fact_index}_not_object")
            continue
        if not str(fact.get("fact") or "").strip():
            issues.append(f"fact_{fact_index}_empty")
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            issues.append(f"fact_{fact_index}_missing_evidence_refs")
            continue
        if max_quotes_per_fact > 0 and len(refs) > max_quotes_per_fact:
            issues.append(f"fact_{fact_index}_too_many_quotes:{len(refs)}>{max_quotes_per_fact}")
        fact_quote_total = 0
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                issues.append(f"fact_{fact_index}_quote_{ref_index}_not_object")
                continue
            quote = str(ref.get("quote") or "").strip()
            if not quote:
                issues.append(f"fact_{fact_index}_quote_{ref_index}_empty")
                continue
            fact_quote_total += len(quote)
            all_quote_total += len(quote)
            if max_quote_chars > 0 and len(quote) > max_quote_chars:
                issues.append(f"fact_{fact_index}_quote_{ref_index}_over_{max_quote_chars}:{len(quote)}")
            if normalize_for_match(quote) not in evidence_norm:
                issues.append(f"fact_{fact_index}_quote_{ref_index}_not_in_evidence")
        if max_fact_quote_total_chars > 0 and fact_quote_total > max_fact_quote_total_chars:
            issues.append(f"fact_{fact_index}_quote_total_over_{max_fact_quote_total_chars}:{fact_quote_total}")
    if max_answer_quote_total_chars > 0 and all_quote_total > max_answer_quote_total_chars:
        issues.append(f"answer_quote_total_over_{max_answer_quote_total_chars}:{all_quote_total}")
    if not str(payload.get("final_answer") or "").strip():
        issues.append("answer_missing_final_answer")
    if not isinstance(payload.get("inferred_facts", []), list):
        issues.append("inferred_facts_not_list")
    return issues


def normalize_repaired_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("next_action") or "").strip()
    if action == "answer_directly":
        return {
            "next_action": "answer_directly",
            "supported_facts": payload.get("supported_facts") if isinstance(payload.get("supported_facts"), list) else [],
            "inferred_facts": payload.get("inferred_facts") if isinstance(payload.get("inferred_facts"), list) else [],
            "final_answer": str(payload.get("final_answer") or "").strip(),
        }
    if action == "retrieve_more":
        follow = payload.get("follow_up_hypothesis") if isinstance(payload.get("follow_up_hypothesis"), dict) else {}
        return {
            "next_action": "retrieve_more",
            "follow_up_hypothesis": {
                "question": str(follow.get("question") or "").strip(),
                "query_type": str(follow.get("query_type") or "").strip(),
                "entities": [str(item).strip() for item in follow.get("entities", []) if str(item).strip()][:12],
                "keywords": [str(item).strip() for item in follow.get("keywords", []) if str(item).strip()][:24],
                "expected_answer_type": str(follow.get("expected_answer_type") or "").strip(),
            },
        }
    if action == "abstain":
        return {"next_action": "abstain", "final_answer": str(payload.get("final_answer") or "现有证据不足以确认。").strip()}
    return payload


def coerce_repaired_assistant(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = extract_json_object(raw)
            except ValueError:
                return None
        return parsed if isinstance(parsed, dict) else None
    return None


SYSTEM_PROMPT = """你是 grounded_action_v1 SFT 数据审计修复器。
只能根据每条样本里的 user_prompt 中的证据判断，不要使用自己的明日方舟知识补事实。
输出必须是单个 JSON 对象：{"records":[...]}。

每条 records 元素：
- id: 原 id
- action: keep | fix | drop
- issue_tags: 字符串数组
- repaired_assistant: 修复后的 grounded_action_v1 JSON；keep 可返回原 assistant；drop 为 null
- reason: 简短中文原因

grounded_action_v1 最新格式：
- answer_directly: {"next_action":"answer_directly","supported_facts":[{"id":"","fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}
- retrieve_more: {"next_action":"retrieve_more","follow_up_hypothesis":{"question":"","query_type":"","entities":[],"keywords":[],"expected_answer_type":""}}
- abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}

硬规则：
1. 不要输出 missing_slots、clarify_user、confidence、current_round、decision。
2. answer_directly 的 quote 必须逐字来自 user_prompt 的证据，不得引用 minirag_hints_not_evidence。
3. 单条 quote 推荐 20-60 字，硬上限 80 字；禁止复制整段 evidence chunk。
4. 单个 supported_fact 最多 2 条 quote，quote 总长度最多 160 字。
5. supported_facts 最多 6 条；所有 quote 总长度最好不超过 400 字。
6. evidence_id 必须从 user_prompt 的“证据”部分逐字复制完整路径#chunk；不得虚构未出现的 chunk id。
7. quote 必须是对应 evidence_id 正文中的连续原文；不得改写标点，不得用省略号拼接非连续句。
8. 裁剪或替换 quote 后，必须同步删改 fact 和 final_answer；不得保留新 quote 不支持的细节。
9. 每个 fact 只写其 evidence_refs 能直接支持的内容；一个 fact 含多件事时，每件事都必须被该 fact 的 quote 支持。
10. fact 和 final_answer 不得引入 quote 不支持的新实体、新关系、新动机、新因果。
11. 证据能回答核心问题时应 answer_directly；只能回答部分时写“现有证据可确认...”并避免证据外补全。
12. 证据不足且还没到最大轮次时应 retrieve_more，并给出新的、具体的 follow_up_hypothesis。
13. 已到最大轮次且证据仍不足时 abstain。
"""


def build_batch_prompt(records: list[dict[str, Any]], *, max_user_chars: int, max_assistant_chars: int) -> str:
    compact: list[dict[str, Any]] = []
    for record in records:
        compact.append(
            {
                "id": str(record.get("id") or ""),
                "user_prompt": truncate_text(rewrite_prompt_quote_rules(user_prompt(record)), max_user_chars),
                "assistant": truncate_text(str((record.get("conversations") or [{}])[-1].get("value") or ""), max_assistant_chars),
            }
        )
    return "请审计并修复下面 grounded_action_v1 SFT 样本。返回 JSON，不要 markdown。\n" + json.dumps(
        {"records": compact},
        ensure_ascii=False,
        indent=2,
    )


def call_chat_api(messages: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key env: {args.api_key_env}")
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            req = Request(normalize_api_url(args.api_base_url), data=data, headers=headers, method="POST")
            with urlopen(req, timeout=args.timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            content = parsed["choices"][0]["message"]["content"]
            return extract_json_object(str(content))
        except (HTTPError, URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"API call failed: {last_error}")


def call_audit_api(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    extra_instruction: str = "",
) -> dict[str, Any]:
    prompt = build_batch_prompt(records, max_user_chars=args.max_user_chars, max_assistant_chars=args.max_assistant_chars)
    if extra_instruction:
        prompt = extra_instruction.strip() + "\n\n" + prompt
    return call_chat_api([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], args)


def call_audit_api_with_fallback(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    try:
        response = call_audit_api(records, args=args)
        response_records = response.get("records")
        if not isinstance(response_records, list):
            raise RuntimeError("API response missing records")
        stats[f"api_batch_success:{len(records)}"] += 1
        return [item for item in response_records if isinstance(item, dict)]
    except Exception as exc:  # noqa: BLE001 - API failures should not abort full dataset repair.
        stats[f"api_batch_error:{len(records)}"] += 1
        if len(records) <= 1:
            rid = str(records[0].get("id") or "")
            return [
                {
                    "id": rid,
                    "action": "drop",
                    "issue_tags": ["api_batch_failed"],
                    "reason": f"API调用失败：{type(exc).__name__}: {exc}",
                    "repaired_assistant": None,
                }
            ]
        midpoint = max(1, len(records) // 2)
        return call_audit_api_with_fallback(records[:midpoint], args=args, stats=stats) + call_audit_api_with_fallback(
            records[midpoint:],
            args=args,
            stats=stats,
        )


def dataset_info(dataset_name: str, splits: list[str]) -> dict[str, Any]:
    return {
        f"{dataset_name}_{split.removesuffix('.json')}": {
            "file_name": split,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }
        for split in splits
    }


def apply_audit(
    record: dict[str, Any],
    audit_item: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[str]]:
    action = str(audit_item.get("action") or "keep").strip()
    if action == "drop":
        return None, []
    repaired = clone_with_rewritten_prompt(record)
    payload = parse_assistant(record)
    if action == "fix":
        candidate = coerce_repaired_assistant(audit_item.get("repaired_assistant"))
        if not isinstance(candidate, dict):
            return None, ["api_fix_missing_repaired_assistant"]
        payload = normalize_repaired_payload(candidate)
    if payload is None:
        return None, ["assistant_json_invalid"]
    issues = validate_payload(
        payload,
        evidence_text=user_prompt(record),
        max_quote_chars=args.max_quote_chars,
        max_quotes_per_fact=args.max_quotes_per_fact,
        max_fact_quote_total_chars=args.max_fact_quote_total_chars,
        max_supported_facts=args.max_supported_facts,
        max_answer_quote_total_chars=args.max_answer_quote_total_chars,
    )
    if issues:
        return None, issues
    repaired["conversations"][-1]["value"] = compact_json(normalize_repaired_payload(payload))
    meta = repaired.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["api_grounded_action_qc"] = {
            "action": action,
            "issue_tags": audit_item.get("issue_tags") or [],
            "reason": str(audit_item.get("reason") or ""),
            "model": args.api_model,
        }
    return repaired, []


def load_records(input_dir: Path, splits: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    split_by_id: dict[str, str] = {}
    for split in splits:
        path = input_dir / split
        if not path.exists():
            continue
        for record in read_json(path):
            rid = str(record.get("id") or f"{split}:{len(records)}")
            record["id"] = rid
            records.append(record)
            split_by_id[rid] = split
    return records, split_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC/repair grounded_action_v1 SFT records with batched API calls.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="grounded_action_sft_api_repaired")
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse successful audit.jsonl rows in output-dir and continue remaining records.")
    parser.add_argument("--only-problematic", action="store_true", help="Only send locally problematic records to API; copy clean records.")
    parser.add_argument("--max-user-chars", type=int, default=18000)
    parser.add_argument("--max-assistant-chars", type=int, default=1800)
    parser.add_argument("--max-quote-chars", type=int, default=80)
    parser.add_argument("--max-quotes-per-fact", type=int, default=2)
    parser.add_argument("--max-fact-quote-total-chars", type=int, default=160)
    parser.add_argument("--max-supported-facts", type=int, default=6)
    parser.add_argument("--max-answer-quote-total-chars", type=int, default=400)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument(
        "--no-repair-invalid-once",
        action="store_true",
        help="Disable one extra single-record API repair when post-validation fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    records, split_by_id = load_records(input_dir, splits)
    rng = random.Random(args.seed)
    selected = list(records)
    rng.shuffle(selected)
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    dry_dir = output_dir / "dry_run"
    stats: Counter[str] = Counter()

    clean_records: dict[str, dict[str, Any]] = {}
    api_records: list[dict[str, Any]] = []
    for record in selected:
        issues = validate_payload(
            parse_assistant(record),
            evidence_text=user_prompt(record),
            max_quote_chars=args.max_quote_chars,
            max_quotes_per_fact=args.max_quotes_per_fact,
            max_fact_quote_total_chars=args.max_fact_quote_total_chars,
            max_supported_facts=args.max_supported_facts,
            max_answer_quote_total_chars=args.max_answer_quote_total_chars,
        )
        if args.only_problematic and not issues:
            clean_records[str(record["id"])] = clone_with_rewritten_prompt(record)
            stats["local_keep_clean"] += 1
        else:
            api_records.append(record)
            for issue in issues:
                stats[f"local_issue:{issue.split(':', 1)[0]}"] += 1

    if args.resume and audit_path.exists():
        original_by_id = {str(record["id"]): record for record in records}
        resume_success_ids: set[str] = set()
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                audit_row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(audit_row.get("id") or "")
            if not rid or audit_row.get("post_validate_issues"):
                continue
            original = original_by_id.get(rid)
            if original is None:
                continue
            repaired, issues = apply_audit(original, audit_row.get("api") or {}, args=args)
            if repaired is None or issues:
                continue
            clean_records[rid] = repaired
            resume_success_ids.add(rid)
        if resume_success_ids:
            api_records = [record for record in api_records if str(record["id"]) not in resume_success_ids]
            stats["resume_success_records"] = len(resume_success_ids)

    if args.dry_run:
        batch = api_records[: args.batch_size]
        prompt = build_batch_prompt(batch, max_user_chars=args.max_user_chars, max_assistant_chars=args.max_assistant_chars)
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
        (dry_dir / "user.txt").write_text(prompt, encoding="utf-8")
        write_json(output_dir / "summary.json", {"dry_run": True, "input_records": len(records), "selected": len(selected), "api_records": len(api_records), "stats": dict(stats)})
        print(json.dumps({"dry_run": True, "prompt": str(dry_dir / "user.txt"), "api_records": len(api_records)}, ensure_ascii=False, indent=2))
        return 0

    repaired_records: dict[str, dict[str, Any]] = dict(clean_records)
    for start in range(0, len(api_records), args.batch_size):
        batch = api_records[start : start + args.batch_size]
        response_records = call_audit_api_with_fallback(batch, args=args, stats=stats)
        by_id = {str(item.get("id") or ""): item for item in response_records}
        for record in batch:
            rid = str(record["id"])
            audit_item = by_id.get(rid) or {"id": rid, "action": "drop", "issue_tags": ["api_missing_record"], "reason": "API未返回该样本", "repaired_assistant": None}
            repaired, issues = apply_audit(record, audit_item, args=args)
            if repaired is None and issues and not args.no_repair_invalid_once:
                retry_instruction = (
                    "上一轮该样本修复后没有通过本地校验。"
                    f"校验错误：{issues}。"
                    "请只修复这一条：quote 必须从当前 user_prompt 的证据块中逐字复制连续原文，"
                    "evidence_id 必须是当前 user_prompt 中出现的完整路径#chunk；"
                    "不得引用未出现的 chunk，不得改写标点，不得拼接非连续句。"
                )
                try:
                    retry_response = call_audit_api([record], args=args, extra_instruction=retry_instruction)
                    retry_records = retry_response.get("records")
                except Exception as exc:  # noqa: BLE001 - keep processing remaining records.
                    retry_records = None
                    stats[f"api_retry_error:{type(exc).__name__}"] += 1
                if isinstance(retry_records, list) and retry_records:
                    retry_item = next(
                        (item for item in retry_records if isinstance(item, dict) and str(item.get("id") or "") == rid),
                        retry_records[0] if isinstance(retry_records[0], dict) else None,
                    )
                    if isinstance(retry_item, dict):
                        retry_repaired, retry_issues = apply_audit(record, retry_item, args=args)
                        if retry_repaired is not None:
                            audit_item = retry_item
                            repaired = retry_repaired
                            issues = []
                            stats["api_retry_fixed"] += 1
                        else:
                            issues = retry_issues
            audit_row = {"id": rid, "split": split_by_id.get(rid), "api": audit_item, "post_validate_issues": issues}
            append_jsonl(audit_path, audit_row)
            if repaired is None:
                append_jsonl(rejected_path, {"id": rid, "record": record, "audit": audit_item, "post_validate_issues": issues})
                stats["drop"] += 1
                continue
            repaired_records[rid] = repaired
            stats[f"api_action:{audit_item.get('action') or 'keep'}"] += 1
        print(f"[batch] {start + len(batch)}/{len(api_records)} repaired={len(repaired_records)} drops={stats['drop']}", flush=True)

    split_outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    for record in records:
        rid = str(record["id"])
        if args.limit is not None and rid not in {str(item["id"]) for item in selected}:
            continue
        repaired = repaired_records.get(rid)
        if repaired is None:
            continue
        split_outputs.setdefault(split_by_id.get(rid, "train.json"), []).append(repaired)
    for split, items in split_outputs.items():
        write_json(output_dir / split, items)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name, list(split_outputs)))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_records": len(records),
        "selected_records": len(selected),
        "api_records": len(api_records),
        "kept_records": sum(len(items) for items in split_outputs.values()),
        "stats": dict(stats),
        "quote_limits": {
            "max_quote_chars": args.max_quote_chars,
            "max_quotes_per_fact": args.max_quotes_per_fact,
            "max_fact_quote_total_chars": args.max_fact_quote_total_chars,
            "max_supported_facts": args.max_supported_facts,
            "max_answer_quote_total_chars": args.max_answer_quote_total_chars,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
