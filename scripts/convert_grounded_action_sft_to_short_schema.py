#!/usr/bin/env python3
"""Convert grounded_action_v1 ShareGPT SFT data to compact-prompt variants."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}

QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
HYPOTHESIS_RE = re.compile(r"(?s)^hypothesis:\s*(\{.*?\})\s*(?:\nallowed_evidence:|\Z)", re.MULTILINE)
EVIDENCE_RE = re.compile(r"(?s)^allowed_evidence:\s*(.*?)(?:\nminirag_hints_not_evidence:|\noutput_schema:|\Z)", re.MULTILINE)
EVIDENCE_BLOCK_RE = re.compile(r"(?ms)^\[证据\s+(\d+)\]\n(.*?)(?=^\[证据\s+\d+\]\n|\Z)")


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def parse_jsonish(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON is not object")
    return payload


def clean_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def parse_evidence_blocks(evidence_text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for match in EVIDENCE_BLOCK_RE.finditer(evidence_text or ""):
        index = match.group(1).strip()
        body = match.group(2).strip()
        doc_id = ""
        clean_text = ""
        lines = body.splitlines()
        for line_index, line in enumerate(lines):
            if line.startswith("id:"):
                doc_id = line.split(":", 1)[1].strip()
            if line.strip() == "clean_text:":
                clean_text = "\n".join(lines[line_index + 1 :]).strip()
                break
        if not clean_text:
            clean_text = body
        blocks.append({"index": index, "id": doc_id, "text": clean_text})
    if blocks:
        return blocks
    raw = evidence_text.strip()
    return [{"index": "1", "id": "", "text": raw}] if raw else []


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def collect_quotes(payload: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    for fact in payload.get("supported_facts") or []:
        if not isinstance(fact, dict):
            continue
        for ref in fact.get("evidence_refs") or []:
            if isinstance(ref, dict) and str(ref.get("quote") or "").strip():
                quotes.append(str(ref["quote"]).strip())
    return quotes


def render_evidence(evidence_text: str, *, top_k: int | None = None, required_quotes: list[str] | None = None) -> str:
    blocks = parse_evidence_blocks(evidence_text)
    if top_k is not None and top_k > 0:
        required_quotes = required_quotes or []
        quote_norms = [normalize_for_match(quote) for quote in required_quotes if quote]
        required_indices = {
            index
            for index, block in enumerate(blocks)
            if quote_norms and any(quote and quote in normalize_for_match(block["text"]) for quote in quote_norms)
        }
        selected_indices = set(range(min(top_k, len(blocks)))) | required_indices
        while len(selected_indices) > top_k:
            removable = sorted((index for index in selected_indices if index not in required_indices), reverse=True)
            if not removable:
                selected_indices = set(sorted(selected_indices)[:top_k])
                break
            selected_indices.remove(removable[0])
        blocks = [block for index, block in enumerate(blocks) if index in selected_indices]
    return render_evidence_blocks(blocks)


def shorten_evidence(evidence_text: str, *, top_k: int | None = None, required_quotes: list[str] | None = None) -> str:
    blocks = parse_evidence_blocks(evidence_text)
    if top_k is not None and top_k > 0:
        required_quotes = required_quotes or []
        quote_norms = [normalize_for_match(quote) for quote in required_quotes if quote]
        required: list[dict[str, str]] = []
        optional: list[dict[str, str]] = []
        for block in blocks:
            block_norm = normalize_for_match(block["text"])
            if quote_norms and any(quote and quote in block_norm for quote in quote_norms):
                required.append(block)
            else:
                optional.append(block)
        selected: list[dict[str, str]] = []
        seen: set[str] = set()
        for block in [*required, *optional]:
            key = block["id"] or block["index"]
            if key in seen:
                continue
            seen.add(key)
            selected.append(block)
            if len(selected) >= top_k:
                break
        blocks = selected
    return render_evidence_blocks(blocks)


def render_evidence_blocks(blocks: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for block in blocks:
        header = f"E{block['index']}"
        if block["id"]:
            header += f" {block['id']}"
        rendered.append(header + "\n" + block["text"].strip())
    return "\n\n".join(rendered)


def short_hypothesis(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    return {
        "t": str(payload.get("query_type") or "").strip(),
        "e": clean_string_list(payload.get("entities"), limit=10),
        "k": clean_string_list(payload.get("keywords"), limit=16),
        "typ": str(payload.get("expected_answer_type") or "").strip(),
    }


def full_hypothesis(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    return {
        "query_type": str(payload.get("query_type") or "").strip(),
        "entities": clean_string_list(payload.get("entities"), limit=10),
        "keywords": clean_string_list(payload.get("keywords"), limit=16),
        "expected_answer_type": str(payload.get("expected_answer_type") or "").strip(),
    }


def build_compact_prompt(source_prompt: str, *, payload: dict[str, Any], top_k: int | None = None) -> str:
    question = extract_match(QUESTION_RE, source_prompt)
    round_id = extract_match(ROUND_RE, source_prompt)
    hypothesis = parse_jsonish(extract_match(HYPOTHESIS_RE, source_prompt) or "{}")
    evidence = extract_match(EVIDENCE_RE, source_prompt)
    lines = [
        "任务：根据证据决定RAG下一步动作，只输出JSON。",
        f"问题：{question}",
    ]
    if round_id:
        lines.append(f"轮次：{round_id}")
    lines.append("当前检索假设：" + compact_json(full_hypothesis(hypothesis)))
    lines.append("证据：")
    lines.append(render_evidence(evidence, top_k=top_k, required_quotes=collect_quotes(payload)) or "<empty>")
    lines.extend(
        [
            "输出格式：",
            'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}',
            'retrieve_more: {"next_action":"retrieve_more","follow_up_hypothesis":{"question":"","query_type":"","entities":[],"keywords":[],"expected_answer_type":""}}',
            'abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
            "规则：只能使用证据；单条quote必须原文精确摘录，推荐20-60字，硬上限80字；每个fact最多2条quote且总长<=160字；supported_facts最多6条，所有quote总长最好<=400字；不要写无证据支持的事实；证据不足才retrieve_more。",
        ]
    )
    return "\n".join(lines)


def build_short_prompt(source_prompt: str, *, payload: dict[str, Any], top_k: int | None = None) -> str:
    question = extract_match(QUESTION_RE, source_prompt)
    round_id = extract_match(ROUND_RE, source_prompt)
    hypothesis = parse_jsonish(extract_match(HYPOTHESIS_RE, source_prompt) or "{}")
    evidence = extract_match(EVIDENCE_RE, source_prompt)
    lines = [
        "task:rag_action",
        f"q:{question}",
    ]
    if round_id:
        lines.append(f"r:{round_id}")
    lines.append("h:" + compact_json(short_hypothesis(hypothesis)))
    lines.append("ev:")
    lines.append(shorten_evidence(evidence, top_k=top_k, required_quotes=collect_quotes(payload)) or "<empty>")
    lines.extend(
        [
            "schema:",
            'answer {"a":"answer","facts":[["quote","fact"]],"infer":[],"ans":""}',
            'retrieve {"a":"retrieve","q":"","t":"","e":[],"k":[],"typ":""}',
            'abstain {"a":"abstain","ans":""}',
            "rules: JSON only; use ev only; quote exact 20-60 chars, hard max 80; max 2 quotes/fact, max 6 facts; total quotes preferably <=400 chars; no unsupported facts; retrieve only if ev insufficient.",
        ]
    )
    return "\n".join(lines)


def full_assistant(payload: dict[str, Any]) -> dict[str, Any]:
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
                "question": str(follow.get("question") or payload.get("question") or "").strip(),
                "query_type": str(follow.get("query_type") or "reasoning").strip(),
                "entities": clean_string_list(follow.get("entities"), limit=10),
                "keywords": clean_string_list(follow.get("keywords"), limit=16),
                "expected_answer_type": str(follow.get("expected_answer_type") or "").strip(),
            },
        }
    if action == "abstain":
        return {
            "next_action": "abstain",
            "final_answer": str(payload.get("final_answer") or "现有证据不足以确认。").strip(),
        }
    raise ValueError(f"Unsupported action: {action}")


def short_assistant(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("next_action") or "").strip()
    if action == "answer_directly":
        facts: list[list[str]] = []
        for fact in payload.get("supported_facts") or []:
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get("fact") or "").strip()
            for ref in fact.get("evidence_refs") or []:
                if not isinstance(ref, dict):
                    continue
                quote = str(ref.get("quote") or "").strip()
                if quote and fact_text:
                    facts.append([quote, fact_text])
        infer = [
            str(item.get("fact") or "").strip()
            for item in payload.get("inferred_facts") or []
            if isinstance(item, dict) and str(item.get("fact") or "").strip()
        ]
        return {
            "a": "answer",
            "facts": facts,
            "infer": infer,
            "ans": str(payload.get("final_answer") or "").strip(),
        }
    if action == "retrieve_more":
        follow = payload.get("follow_up_hypothesis") if isinstance(payload.get("follow_up_hypothesis"), dict) else {}
        return {
            "a": "retrieve",
            "q": str(follow.get("question") or payload.get("question") or "").strip(),
            "t": str(follow.get("query_type") or "reasoning").strip(),
            "e": clean_string_list(follow.get("entities"), limit=10),
            "k": clean_string_list(follow.get("keywords"), limit=16),
            "typ": str(follow.get("expected_answer_type") or "").strip(),
        }
    if action == "abstain":
        return {"a": "abstain", "ans": str(payload.get("final_answer") or "现有证据不足以确认。").strip()}
    raise ValueError(f"Unsupported action: {action}")


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert grounded_action_v1 SFT data into compact prompt/assistant variants.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--mode",
        choices=["short_schema", "compact_prompt"],
        default="short_schema",
        help="short_schema shortens field names; compact_prompt keeps original output field names and only shortens prompt wording.",
    )
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--top-k-evidence", type=int, default=0, help="Optional evidence block cap; 0 keeps all blocks.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    top_k = args.top_k_evidence if args.top_k_evidence > 0 else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in [*splits, "dataset_info.json", "summary.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    stats: Counter[str] = Counter()
    for split in splits:
        records = read_json(input_dir / split)
        output_records: list[dict[str, Any]] = []
        for record in records:
            prompt = record["conversations"][0]["value"]
            payload = parse_jsonish(record["conversations"][-1]["value"])
            if args.mode == "compact_prompt":
                assistant_payload = full_assistant(payload)
                converted_prompt = build_compact_prompt(prompt, payload=payload, top_k=top_k)
                action = assistant_payload["next_action"]
            else:
                assistant_payload = short_assistant(payload)
                converted_prompt = build_short_prompt(prompt, payload=payload, top_k=top_k)
                action = assistant_payload["a"]
            output_records.append(
                {
                    **{k: v for k, v in record.items() if k not in {"conversations", "system"}},
                    "system": "你是明日方舟RAG动作模块。只输出JSON。",
                    "conversations": [
                        {"from": "human", "value": converted_prompt},
                        {"from": "gpt", "value": compact_json(assistant_payload)},
                    ],
                }
            )
            stats[f"action:{action}"] += 1
            stats[f"split:{split}"] += 1
        write_json(output_dir / split, output_records)

    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "mode": args.mode,
        "top_k_evidence": top_k,
        "splits": {split: len(read_json(output_dir / split)) for split in splits},
        "stats": dict(stats),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
