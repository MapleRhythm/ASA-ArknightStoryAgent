#!/usr/bin/env python3
"""Merge policy SFT and grounded-answer SFT into a no-missing-slots action SFT.

Output schema:
{
  "question": "...",
  "next_action": "answer_directly | retrieve_more | abstain",
  "follow_up_hypothesis": {...} | null,
  "supported_facts": [...],
  "inferred_facts": [...],
  "final_answer": "..."
}

For retrieve_more, supported_facts/inferred_facts are empty and final_answer is
an empty string. No missing_slots field is emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
HYPOTHESIS_RE = re.compile(r"(?s)^hypothesis:\s*(\{.*?\})\s*(?:\nround:|\nevidence_brief:|\Z)", re.MULTILINE)
EVIDENCE_BRIEF_RE = re.compile(
    r"(?s)^evidence_brief:\s*(.*?)(?:\nminirag_hints:|\noutput_schema:|\nfields:|\nnext_action_set:|\nfield_rules:|\Z)",
    re.MULTILINE,
)
MINIRAG_HINTS_RE = re.compile(r"(?s)^minirag_hints:\s*(.*?)(?:\noutput_schema:|\nfields:|\nnext_action_set:|\Z)", re.MULTILINE)

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_jsonish(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def record_user_value(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    first = conversations[0]
    return str(first.get("value") or "") if isinstance(first, dict) else ""


def record_assistant_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    last = conversations[-1]
    return parse_jsonish(str(last.get("value") or "")) if isinstance(last, dict) else None


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


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]


def build_grounded_action_prompt(source_prompt: str, *, max_evidence_chars: int) -> str:
    parts = extract_prompt_parts(source_prompt)
    lines = [
        "task: grounded_action_generation",
        f"question: {parts['question']}",
    ]
    if parts["round"]:
        lines.append(f"round: {parts['round']}")
    if parts["hypothesis"]:
        lines.extend(["hypothesis:", parts["hypothesis"]])
    lines.extend(["allowed_evidence:", truncate_middle(parts["evidence_brief"], max_evidence_chars) or "<empty>"])
    if parts["minirag_hints"]:
        lines.extend(["minirag_hints_not_evidence:", truncate_middle(parts["minirag_hints"], 1600)])
    lines.extend(
        [
            "output_schema: grounded_action_v1",
            "fields: question,next_action,follow_up_hypothesis,supported_facts,inferred_facts,final_answer",
            "next_action_set: answer_directly,retrieve_more,abstain",
            "rules:",
            "1. 只输出 JSON，不要 markdown，不要思维过程。",
            "2. 不要输出旧版检索缺口列表字段。",
            "3. next_action=retrieve_more 时，follow_up_hypothesis 必须非空；supported_facts/inferred_facts 为空数组；final_answer 为空字符串。",
            "4. next_action=answer_directly 时，follow_up_hypothesis=null；supported_facts 必须引用 allowed_evidence 原文 quote；final_answer 只能使用 supported_facts 和 inferred_facts。",
            "5. quote 必须从 allowed_evidence 原文复制，不要改写。",
            "6. inferred_facts 只能基于 supported_facts 的 premise_fact_ids 做最小必要推理，不得引入新实体、新动机、新因果。",
            "7. 不要把 minirag_hints_not_evidence 当作事实证据。",
            "8. follow_up_hypothesis 仅包含 question,query_type,entities,keywords,expected_answer_type,dialogue_context。",
        ]
    )
    return "\n".join(lines)


def dataset_info(dataset_name: str, split_files: list[str]) -> dict[str, Any]:
    return {
        f"{dataset_name}_{split.removesuffix('.json')}": {
            "file_name": split,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }
        for split in split_files
    }


def load_grounded_answer_map(grounded_answer_dir: Path, split_files: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in split_files:
        path = grounded_answer_dir / split
        if not path.exists():
            continue
        records = read_json(path)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
            source_id = str(meta.get("source_record_id") or "")
            payload = record_assistant_payload(record)
            if source_id and isinstance(payload, dict):
                output[source_id] = payload
    return output


def clean_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def sanitize_follow_up_hypothesis(value: Any, *, question: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    entities = clean_string_list(value.get("entities"), limit=12)
    keywords = clean_string_list(value.get("keywords"), limit=24)
    payload = {
        "question": str(value.get("question") or question).strip(),
        "query_type": str(value.get("query_type") or "reasoning").strip(),
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": str(value.get("expected_answer_type") or "剧情问答").strip(),
        "dialogue_context": str(value.get("dialogue_context") or "").strip(),
    }
    if not payload["question"]:
        payload["question"] = question
    if not payload["entities"] and not payload["keywords"]:
        return None
    return payload


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(item_key == key or contains_key(item_value, key) for item_key, item_value in value.items())
    if isinstance(value, list):
        return any(contains_key(item, key) for item in value)
    return False


def make_grounded_action_payload(
    *,
    question: str,
    policy_payload: dict[str, Any],
    grounded_answer: dict[str, Any] | None,
) -> dict[str, Any] | None:
    action = str(policy_payload.get("next_action") or "").strip()
    if action == "answer_directly":
        if not grounded_answer:
            return None
        return {
            "question": question,
            "next_action": "answer_directly",
            "follow_up_hypothesis": None,
            "supported_facts": grounded_answer.get("supported_facts") or [],
            "inferred_facts": grounded_answer.get("inferred_facts") or [],
            "final_answer": str(grounded_answer.get("final_answer") or "").strip(),
        }
    if action == "retrieve_more":
        follow_up = sanitize_follow_up_hypothesis(policy_payload.get("follow_up_hypothesis"), question=question)
        if follow_up is None:
            return None
        return {
            "question": question,
            "next_action": "retrieve_more",
            "follow_up_hypothesis": follow_up,
            "supported_facts": [],
            "inferred_facts": [],
            "final_answer": "",
        }
    if action == "abstain":
        return {
            "question": question,
            "next_action": "abstain",
            "follow_up_hypothesis": None,
            "supported_facts": [],
            "inferred_facts": [],
            "final_answer": str(policy_payload.get("answer") or "现有证据不足以确认。").strip(),
        }
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-missing-slots grounded action SFT from policy and grounded-answer datasets.")
    parser.add_argument("--policy-sft-dir", required=True)
    parser.add_argument("--grounded-answer-sft-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--actions", default="answer_directly,retrieve_more")
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_dir = resolve_path(args.policy_sft_dir)
    grounded_dir = resolve_path(args.grounded_answer_sft_dir)
    output_dir = resolve_path(args.output_dir)
    split_files = [item.strip() for item in args.splits.split(",") if item.strip()]
    allowed_actions = {item.strip() for item in args.actions.split(",") if item.strip()}
    stats: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in [*split_files, "dataset_info.json", "summary.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    grounded_by_source_id = load_grounded_answer_map(grounded_dir, split_files)
    for split in split_files:
        path = policy_dir / split
        output_records: list[dict[str, Any]] = []
        if not path.exists():
            write_json(output_dir / split, output_records)
            stats[f"missing_split:{split}"] += 1
            continue
        records = read_json(path)
        if not isinstance(records, list):
            raise SystemExit(f"Policy split is not a list: {path}")
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("task_type") or "") != "conclusion_generation":
                stats["skip_non_conclusion"] += 1
                continue
            policy_payload = record_assistant_payload(record)
            if not isinstance(policy_payload, dict):
                stats["skip_parse_error"] += 1
                continue
            action = str(policy_payload.get("next_action") or "").strip()
            if action not in allowed_actions:
                stats[f"skip_action:{action or '<empty>'}"] += 1
                continue
            source_id = str(record.get("id") or "")
            prompt = record_user_value(record)
            question = extract_prompt_parts(prompt)["question"] or str(policy_payload.get("question") or "")
            grounded_payload = make_grounded_action_payload(
                question=question,
                policy_payload=policy_payload,
                grounded_answer=grounded_by_source_id.get(source_id),
            )
            if grounded_payload is None:
                stats[f"drop_no_payload:{action}"] += 1
                continue
            if contains_key(grounded_payload, "missing_slots"):
                raise RuntimeError("grounded_action payload unexpectedly contains legacy retrieval-gap field")
            output_records.append(
                {
                    "id": f"{source_id}__grounded_action_sft",
                    "task_type": "grounded_action_generation",
                    "bucket": "tool",
                    "system": "你是《明日方舟》剧情 RAG 的证据约束动作与回答模块。只输出 JSON。",
                    "tools": [],
                    "conversations": [
                        {"from": "human", "value": build_grounded_action_prompt(prompt, max_evidence_chars=args.max_evidence_chars)},
                        {"from": "gpt", "value": compact_json(grounded_payload)},
                    ],
                    "meta": {
                        "source_policy_sft_dir": policy_dir.name,
                        "source_grounded_answer_sft_dir": grounded_dir.name,
                        "source_record_id": source_id,
                        "source_split": split,
                    },
                }
            )
            stats[f"output_action:{action}"] += 1
            stats[f"output_split:{split}"] += 1
        write_json(output_dir / split, output_records)

    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name, split_files))
    summary = {
        "policy_sft_dir": str(policy_dir),
        "grounded_answer_sft_dir": str(grounded_dir),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "splits": {split: len(read_json(output_dir / split)) for split in split_files},
        "actions": sorted(allowed_actions),
        "schema": "grounded_action_v1",
        "stats": dict(stats),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
