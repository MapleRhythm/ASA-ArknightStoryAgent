#!/usr/bin/env python3
"""Build evidence-grounded answer SFT data with an API judge/generator.

The script reads existing ShareGPT conclusion samples, asks an API model to
rewrite answer_directly outputs into a claim-grounded schema, validates quoted
evidence, and writes a new LLaMA-Factory SFT dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
HYPOTHESIS_RE = re.compile(r"(?s)^hypothesis:\s*(\{.*?\})\s*(?:\nround:|\nevidence_brief:|\Z)", re.MULTILINE)
EVIDENCE_BRIEF_RE = re.compile(
    r"(?s)^evidence_brief:\s*(.*?)(?:\nminirag_hints:|\noutput_schema:|\nfields:|\nnext_action_set:|\nfield_rules:|\Z)",
    re.MULTILINE,
)
MINIRAG_HINTS_RE = re.compile(r"(?s)^minirag_hints:\s*(.*?)(?:\noutput_schema:|\nfields:|\nnext_action_set:|\Z)", re.MULTILINE)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
CHINESE_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
IDENTITY_MARKERS = ("即", "就是", "同一人", "同一个人", "本名", "代号", "又名", "也叫", "名为")
GENERIC_TOKENS = {
    "现有证据",
    "证据",
    "确认",
    "无法确认",
    "部分",
    "关联",
    "关系",
    "身份",
    "同一人",
    "同一个人",
}


ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


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


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_text_for_match(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text or ""))


def parse_jsonish(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty JSON text")
    block = JSON_BLOCK_RE.search(raw)
    if block:
        raw = block.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError("cannot parse JSON object")


def extract_json_object(text: str) -> dict[str, Any]:
    parsed = parse_jsonish(text)
    if not isinstance(parsed, dict):
        raise ValueError("API response is not a JSON object")
    return parsed


def record_user_value(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    first = conversations[0]
    return str(first.get("value") or "") if isinstance(first, dict) else ""


def record_assistant_value(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    last = conversations[-1]
    return str(last.get("value") or "") if isinstance(last, dict) else ""


def extract_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def extract_prompt_parts(prompt: str) -> dict[str, str]:
    return {
        "question": extract_match(QUESTION_RE, prompt),
        "round": extract_match(ROUND_RE, prompt),
        "hypothesis": extract_match(HYPOTHESIS_RE, prompt),
        "evidence_brief": extract_match(EVIDENCE_BRIEF_RE, prompt),
        "minirag_hints": extract_match(MINIRAG_HINTS_RE, prompt),
    }


def assistant_action(record: dict[str, Any]) -> str:
    try:
        payload = parse_jsonish(record_assistant_value(record))
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("next_action") or "").strip()


def assistant_answer(record: dict[str, Any]) -> str:
    try:
        payload = parse_jsonish(record_assistant_value(record))
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("answer") or "").strip()


def build_training_prompt(source_prompt: str, *, max_evidence_chars: int) -> str:
    parts = extract_prompt_parts(source_prompt)
    evidence = truncate_middle(parts["evidence_brief"], max_evidence_chars)
    minirag_hints = truncate_middle(parts["minirag_hints"], 1600)
    user_lines = [
        "task: evidence_grounded_answer_generation",
        f"question: {parts['question']}",
    ]
    if parts["round"]:
        user_lines.append(f"round: {parts['round']}")
    if parts["hypothesis"]:
        user_lines.extend(["hypothesis:", parts["hypothesis"]])
    user_lines.extend(
        [
            "allowed_evidence:",
            evidence or "<empty>",
        ]
    )
    if minirag_hints:
        user_lines.extend(
            [
                "minirag_hints_not_evidence:",
                minirag_hints,
            ]
        )
    user_lines.extend(
        [
            "output_schema: evidence_grounded_answer_v1",
            'fields: supported_facts,inferred_facts,final_answer',
            "supported_facts item fields: id,fact,evidence_refs",
            "evidence_refs item fields: evidence_id,quote",
            "inferred_facts item fields: id,fact,premise_fact_ids,inference_type",
            "rules:",
            "1. 只输出 JSON，不要 markdown，不要思维过程。",
            "2. supported_facts 必须能由 allowed_evidence 中的原文 quote 直接支持。",
            "3. quote 必须从 allowed_evidence 原文复制，不要改写。",
            "4. inferred_facts 只能基于 supported_facts 的 premise_fact_ids 做最小必要推理，不得引入新实体、新动机、新因果。",
            "5. final_answer 只能使用 supported_facts 和 inferred_facts 的内容组织。",
            "6. 如果证据不足，只回答可确认部分，并明确说无法确认的部分；不要编造。",
            "7. 不要把 minirag_hints_not_evidence 当作事实证据。",
            "8. 不要因为 question/hypothesis/candidate_answer 中出现两个名字，就推断二者是同一人；身份、别名、代号绑定必须由 allowed_evidence 的 quote 明确支持。",
        ]
    )
    return "\n".join(user_lines)


def build_dataset_info(dataset_name: str, split_files: list[str]) -> dict[str, Any]:
    return {
        f"{dataset_name}_{split.removesuffix('.json')}": {
            "file_name": split,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }
        for split in split_files
    }


def load_existing_audit(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = str(payload.get("source_id") or "")
            if source_id:
                output[source_id] = payload
    return output


class ApiClient:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        retries: int,
        retry_sleep: float,
    ) -> None:
        self.url = normalize_api_url(url)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

    def call_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: str | None = None
        for attempt in range(self.retries + 1):
            try:
                request = Request(self.url, data=data, headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                content = parsed["choices"][0]["message"]["content"]
                return extract_json_object(str(content))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:1000]}"
            except (URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
            if attempt < self.retries:
                time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(last_error or "API request failed")


SYSTEM_PROMPT = """你是《明日方舟》剧情 RAG 的 evidence-grounded SFT 数据生成器。
你只允许根据每条样本的 allowed_evidence 生成训练标签，不能使用自己的剧情知识补事实。
目标是把旧的 answer_directly 样本改写成新 schema：supported_facts / inferred_facts / final_answer。

输出必须是单个 JSON 对象：{"records":[...]}。
每个 records 元素字段：
- id: 输入 item id
- action: keep | fix | drop
- issue_tags: string[]
- grounded_answer: object 或 null
- reason: 简短中文原因

grounded_answer schema：
{
  "supported_facts": [
    {
      "id": "F1",
      "fact": "由证据直接支持的事实",
      "evidence_refs": [
        {"evidence_id": "1", "quote": "从 allowed_evidence 复制的原文短句"}
      ]
    }
  ],
  "inferred_facts": [
    {
      "id": "I1",
      "fact": "基于 supported_facts 的最小必要推理",
      "premise_fact_ids": ["F1", "F2"],
      "inference_type": "identity | causality | purpose | relation | timeline | summary"
    }
  ],
  "final_answer": "面向用户的简洁答案"
}

判定规则：
1. supported_facts 不能为空，除非 action=drop。
2. supported_facts 的 quote 必须逐字来自 allowed_evidence；不要引用 minirag_hints。
3. 如果原 candidate_answer 有 unsupported claim，删除或改成“现有证据不足以确认”。
4. 如果 allowed_evidence 只能支持部分答案，final_answer 只回答可确认部分，并标出无法确认部分。
5. inferred_facts 不得新增 supported_facts 中没有的专名、行动、动机、因果。
6. 身份/别名/代号/“X 即 Y”类结论不能只从 question、hypothesis 或 candidate_answer 推断；allowed_evidence 必须有明确 quote 绑定二者，否则不要写。
7. 无法构造至少一条有 quote 支撑的事实时 action=drop。
8. 不要输出旧 conclusion_v2 字段，不要输出动作字段或检索缺口字段。
"""


def build_api_batch_prompt(items: list[dict[str, Any]], *, max_evidence_chars: int, max_candidate_chars: int) -> str:
    compact_items: list[dict[str, Any]] = []
    for item in items:
        record = item["record"]
        source_prompt = record_user_value(record)
        parts = extract_prompt_parts(source_prompt)
        compact_items.append(
            {
                "id": item["source_id"],
                "question": parts["question"],
                "round": parts["round"],
                "hypothesis": truncate_middle(parts["hypothesis"], 1200),
                "allowed_evidence": truncate_middle(parts["evidence_brief"], max_evidence_chars),
                "minirag_hints_not_evidence": truncate_middle(parts["minirag_hints"], 1200),
                "candidate_answer": truncate_middle(assistant_answer(record), max_candidate_chars),
            }
        )
    return (
        "请把下面每条旧 answer_directly 样本改写成 evidence_grounded_answer_v1 SFT 标签。"
        "返回 JSON，不要 markdown。\n\n"
        + json.dumps({"items": compact_items}, ensure_ascii=False, indent=2)
    )


def validate_grounded_answer(payload: Any, evidence_text: str, *, strict_quotes: bool) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return False, ["grounded_answer_not_object"]
    allowed_keys = {"supported_facts", "inferred_facts", "final_answer"}
    extra_keys = set(payload) - allowed_keys
    if extra_keys:
        issues.append("extra_fields:" + ",".join(sorted(extra_keys)))
    supported = payload.get("supported_facts")
    if not isinstance(supported, list) or not supported:
        issues.append("missing_supported_facts")
        supported = []
    evidence_compact = normalize_text_for_match(evidence_text)
    fact_ids: set[str] = set()
    for index, fact in enumerate(supported, start=1):
        if not isinstance(fact, dict):
            issues.append(f"supported_fact_{index}_not_object")
            continue
        fact_id = str(fact.get("id") or "").strip()
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_id:
            issues.append(f"supported_fact_{index}_missing_id")
        else:
            fact_ids.add(fact_id)
        if not fact_text:
            issues.append(f"supported_fact_{index}_missing_fact")
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            issues.append(f"supported_fact_{index}_missing_refs")
            continue
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                issues.append(f"supported_fact_{index}_ref_{ref_index}_not_object")
                continue
            quote = str(ref.get("quote") or "").strip()
            if not quote:
                issues.append(f"supported_fact_{index}_ref_{ref_index}_missing_quote")
                continue
            if strict_quotes and normalize_text_for_match(quote) not in evidence_compact:
                issues.append(f"supported_fact_{index}_ref_{ref_index}_quote_not_found")
    inferred = payload.get("inferred_facts")
    if inferred is None:
        payload["inferred_facts"] = []
        inferred = []
    if not isinstance(inferred, list):
        issues.append("inferred_facts_not_list")
        inferred = []
    for index, fact in enumerate(inferred, start=1):
        if not isinstance(fact, dict):
            issues.append(f"inferred_fact_{index}_not_object")
            continue
        if not str(fact.get("id") or "").strip():
            issues.append(f"inferred_fact_{index}_missing_id")
        if not str(fact.get("fact") or "").strip():
            issues.append(f"inferred_fact_{index}_missing_fact")
        premise_ids = fact.get("premise_fact_ids")
        if not isinstance(premise_ids, list) or not premise_ids:
            issues.append(f"inferred_fact_{index}_missing_premises")
            continue
        for premise_id in premise_ids:
            if str(premise_id) not in fact_ids:
                issues.append(f"inferred_fact_{index}_unknown_premise:{premise_id}")
        fact_text = str(fact.get("fact") or "").strip()
        inference_type = str(fact.get("inference_type") or "").strip()
        if inference_type == "identity" or any(marker in fact_text for marker in IDENTITY_MARKERS):
            tokens = [
                token
                for token in CHINESE_TOKEN_RE.findall(fact_text)
                if token not in GENERIC_TOKENS and not any(generic in token for generic in GENERIC_TOKENS)
            ]
            missing_tokens = [token for token in tokens if token not in evidence_compact]
            has_identity_marker_in_evidence = any(marker in evidence_compact for marker in IDENTITY_MARKERS)
            if missing_tokens or not has_identity_marker_in_evidence:
                issues.append(f"inferred_fact_{index}_identity_not_evidence_grounded")
    final_answer = str(payload.get("final_answer") or "").strip()
    if not final_answer:
        issues.append("missing_final_answer")
    return not any(
        issue.startswith("missing_supported_facts")
        or issue.endswith("_missing_quote")
        or issue.endswith("_quote_not_found")
        or issue.endswith("_identity_not_evidence_grounded")
        or issue == "missing_final_answer"
        for issue in issues
    ), issues


def make_sft_record(
    *,
    source_record: dict[str, Any],
    source_id: str,
    split: str,
    grounded_answer: dict[str, Any],
    output_prompt: str,
    audit: dict[str, Any],
    source_dataset_name: str,
) -> dict[str, Any]:
    meta = source_record.get("meta") if isinstance(source_record.get("meta"), dict) else {}
    return {
        "id": f"{source_id}__evidence_grounded_answer_sft",
        "task_type": "evidence_grounded_answer_generation",
        "bucket": "tool",
        "system": "你是《明日方舟》剧情 RAG 的证据约束回答模块。只输出 JSON。",
        "tools": [],
        "conversations": [
            {"from": "human", "value": output_prompt},
            {"from": "gpt", "value": compact_json(grounded_answer)},
        ],
        "meta": {
            "source_dataset": source_dataset_name,
            "source_split": split,
            "source_record_id": source_id,
            "source_task_type": source_record.get("task_type"),
            "source_kto_tag": source_record.get("kto_tag"),
            "source_prompt_key": meta.get("prompt_key"),
            "api_grounded_sft_v1": {
                "action": audit.get("action"),
                "issue_tags": audit.get("issue_tags") or [],
                "reason": audit.get("reason") or "",
                "validation_issues": audit.get("validation_issues") or [],
                "model": audit.get("model"),
            },
        },
    }


def select_source_records(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    input_dir = resolve_path(args.input_dir)
    source_by_id: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    allowed_actions = {item.strip() for item in args.actions.split(",") if item.strip()}
    split_files = [item.strip() for item in args.splits.split(",") if item.strip()]
    for split in split_files:
        path = input_dir / split
        if not path.exists():
            stats[f"missing_split:{split}"] += 1
            continue
        records = read_json(path)
        if not isinstance(records, list):
            raise SystemExit(f"Input split is not a list: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            stats["records_total"] += 1
            task_type = str(record.get("task_type") or "")
            if task_type != "conclusion_generation":
                stats[f"skip_task:{task_type or '<empty>'}"] += 1
                continue
            if args.positive_only and "kto_tag" in record and not bool(record.get("kto_tag")):
                stats["skip_negative_kto"] += 1
                continue
            action = assistant_action(record)
            if action not in allowed_actions:
                stats[f"skip_action:{action or '<parse_error>'}"] += 1
                continue
            source_prompt = record_user_value(record)
            parts = extract_prompt_parts(source_prompt)
            if not parts["question"] or not parts["evidence_brief"]:
                stats["skip_missing_question_or_evidence"] += 1
                continue
            answer = assistant_answer(record)
            if not answer and "answer_directly" in allowed_actions:
                stats["skip_empty_answer"] += 1
                continue
            source_id = str(record.get("id") or f"{split}:{index}")
            if source_id in source_by_id:
                source_id = f"{source_id}__dup{index}"
            item = {"source_id": source_id, "split": split, "record": record}
            source_by_id[source_id] = item
            selected.append(item)
            stats["records_selected"] += 1
            if args.limit is not None and len(selected) >= args.limit:
                return source_by_id, selected, stats
    return source_by_id, selected, stats


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def build_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def normalize_api_records(payload: dict[str, Any], expected_ids: set[str]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("API payload missing records list")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or "").strip()
        if source_id not in expected_ids or source_id in seen:
            continue
        seen.add(source_id)
        output.append(item)
    missing = expected_ids - seen
    if missing:
        raise ValueError(f"API payload missing record ids: {sorted(missing)[:5]}")
    return output


def run_api_generation(args: argparse.Namespace, selected: list[dict[str, Any]], audit_path: Path) -> dict[str, dict[str, Any]]:
    existing = load_existing_audit(audit_path)
    pending = [item for item in selected if item["source_id"] not in existing]
    if args.shuffle:
        random.Random(args.seed).shuffle(pending)
    if args.dry_run:
        preview_dir = resolve_path(args.output_dir) / "dry_run"
        preview_dir.mkdir(parents=True, exist_ok=True)
        batches = build_batches(pending[: args.batch_size * 3], args.batch_size)
        for index, batch in enumerate(batches, start=1):
            (preview_dir / f"batch_{index:04d}_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
            (preview_dir / f"batch_{index:04d}_user.txt").write_text(
                build_api_batch_prompt(
                    batch,
                    max_evidence_chars=args.max_api_evidence_chars,
                    max_candidate_chars=args.max_candidate_chars,
                ),
                encoding="utf-8",
            )
        print(f"[dry-run] selected={len(selected)} existing={len(existing)} pending={len(pending)} preview={preview_dir}")
        return existing

    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    client = ApiClient(
        url=args.api_base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    batches = build_batches(pending, args.batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        expected_ids = {str(item["source_id"]) for item in batch}
        user_prompt = build_api_batch_prompt(
            batch,
            max_evidence_chars=args.max_api_evidence_chars,
            max_candidate_chars=args.max_candidate_chars,
        )
        started = time.time()
        payload = client.call_json(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        records = normalize_api_records(payload, expected_ids)
        for api_record in records:
            source_id = str(api_record["id"])
            source_item = next(item for item in batch if item["source_id"] == source_id)
            evidence = extract_prompt_parts(record_user_value(source_item["record"]))["evidence_brief"]
            grounded_answer = api_record.get("grounded_answer")
            valid = False
            validation_issues: list[str] = []
            if str(api_record.get("action") or "").strip() != "drop":
                valid, validation_issues = validate_grounded_answer(
                    grounded_answer,
                    evidence,
                    strict_quotes=not args.no_strict_quotes,
                )
            else:
                validation_issues = []
            action = str(api_record.get("action") or "").strip()
            if action not in {"keep", "fix", "drop"}:
                action = "drop"
                validation_issues.append("invalid_action")
            if action != "drop" and not valid:
                action = "drop"
            audit = {
                "source_id": source_id,
                "split": source_item["split"],
                "ok": True,
                "action": action,
                "issue_tags": api_record.get("issue_tags") or [],
                "grounded_answer": grounded_answer if action != "drop" else None,
                "reason": str(api_record.get("reason") or ""),
                "validation_issues": validation_issues,
                "model": args.model,
                "batch_index": batch_index,
                "latency_sec": round(time.time() - started, 3),
                "created_at": int(time.time()),
            }
            append_jsonl(audit_path, audit)
            existing[source_id] = audit
        print(f"[api] batch={batch_index}/{len(batches)} records={len(batch)} elapsed={time.time() - started:.1f}s")
    return existing


def build_output_dataset(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    stats: Counter[str],
) -> dict[str, Any]:
    output_dir = resolve_path(args.output_dir)
    split_files = [item.strip() for item in args.splits.split(",") if item.strip()]
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in split_files}
    for item in selected:
        source_id = item["source_id"]
        audit = audits.get(source_id)
        if not audit:
            stats["skip_no_audit"] += 1
            continue
        action = str(audit.get("action") or "")
        stats[f"audit_action:{action or '<empty>'}"] += 1
        if action == "drop" or not audit.get("grounded_answer"):
            continue
        source_prompt = record_user_value(item["record"])
        output_prompt = build_training_prompt(source_prompt, max_evidence_chars=args.max_train_evidence_chars)
        record = make_sft_record(
            source_record=item["record"],
            source_id=source_id,
            split=item["split"],
            grounded_answer=audit["grounded_answer"],
            output_prompt=output_prompt,
            audit=audit,
            source_dataset_name=resolve_path(args.input_dir).name,
        )
        by_split.setdefault(item["split"], []).append(record)
        stats[f"output_split:{item['split']}"] += 1

    for split, records in by_split.items():
        write_json(output_dir / split, records)
    write_json(output_dir / "dataset_info.json", build_dataset_info(args.dataset_name, split_files))
    summary = {
        "input_dir": str(resolve_path(args.input_dir)),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "splits": {split: len(by_split.get(split, [])) for split in split_files},
        "actions": {key.removeprefix("audit_action:"): value for key, value in stats.items() if key.startswith("audit_action:")},
        "stats": dict(stats),
        "schema": "evidence_grounded_answer_v1",
        "strict_quotes": not args.no_strict_quotes,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence-grounded answer SFT data with an API model.")
    parser.add_argument("--input-dir", default="data/processed/llama_factory/conclusion_chosen_sft_v1_api_qc_v1")
    parser.add_argument("--output-dir", default="data/processed/llama_factory/evidence_grounded_answer_sft_v1_api")
    parser.add_argument("--dataset-name", default="evidence_grounded_answer_sft_v1_api")
    parser.add_argument("--splits", default="train.json,val.json", help="Comma-separated split files to read/write.")
    parser.add_argument("--actions", default="answer_directly", help="Comma-separated source conclusion actions to convert.")
    parser.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-strict-quotes", action="store_true", help="Do not drop outputs whose quotes are not exact substrings.")
    parser.add_argument("--max-api-evidence-chars", type=int, default=9000)
    parser.add_argument("--max-train-evidence-chars", type=int, default=12000)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if args.overwrite and output_dir.exists():
        for name in ("audit.jsonl", "train.json", "val.json", "dataset_info.json", "summary.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.jsonl"

    _source_by_id, selected, stats = select_source_records(args)
    print(f"[select] selected={len(selected)} stats={dict(stats)}")
    audits = run_api_generation(args, selected, audit_path)
    if args.dry_run:
        return
    summary = build_output_dataset(args, selected, audits, stats)
    print("[done] " + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
