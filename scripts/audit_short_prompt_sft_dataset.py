#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_current_short_prompt_v1"

INTENTS = {
    "character_relation",
    "compare",
    "event_summary",
    "out_of_scope",
    "persona_chat",
    "plot_fact",
    "plot_reasoning",
    "timeline",
}
QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
GENERIC_VOICE_TAGS = (
    "信赖触摸",
    "信赖提升后交谈",
    "行动失败",
    "行动出发",
    "选中干员",
    "任命助理",
    "任命队长",
    "编入队伍",
    "闲置",
    "戳一下",
    "问候",
)
NOISY_TERMS = {
    "什么",
    "为什么",
    "怎么",
    "如何",
    "关系",
    "身份",
    "角色",
    "人物",
    "干员",
    "证据",
    "当前",
    "剧情",
    "问题",
    "具体",
}
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff·]{2,16}|[A-Za-z][A-Za-z0-9_.-]{1,31}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assistant_text(record: dict[str, Any]) -> str:
    for message in reversed(record.get("conversations") or []):
        if message.get("from") in {"gpt", "assistant"}:
            return str(message.get("value") or message.get("content") or "")
    return ""


def user_text(record: dict[str, Any]) -> str:
    for message in record.get("conversations") or []:
        if message.get("from") in {"human", "user"}:
            return str(message.get("value") or message.get("content") or "")
    return ""


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def schema_errors(task_type: str, payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return ["invalid_json"]
    errors: list[str] = []
    if task_type == "user_question_hypothesis_generation":
        required = {"question", "intent", "query_type", "entities", "keywords", "expected_answer_type", "dialogue_context"}
        missing = required - set(payload)
        if missing:
            errors.append("missing:" + ",".join(sorted(missing)))
        if payload.get("intent") not in INTENTS:
            errors.append("bad_intent")
        if payload.get("query_type") not in QUERY_TYPES:
            errors.append("bad_query_type")
        extra = set(payload) - required
        if extra:
            errors.append("extra:" + ",".join(sorted(extra)))
    elif task_type == "follow_up_hypothesis_generation":
        required = {"question", "query_type", "entities", "keywords", "expected_answer_type", "dialogue_context"}
        missing = required - set(payload)
        if missing:
            errors.append("missing:" + ",".join(sorted(missing)))
        if payload.get("query_type") not in QUERY_TYPES:
            errors.append("bad_query_type")
        extra = set(payload) - required
        if extra:
            errors.append("extra:" + ",".join(sorted(extra)))
    elif task_type == "conclusion_generation":
        required = {"question", "next_action", "answer", "missing_slots", "clarification_question", "follow_up_hypothesis"}
        missing = required - set(payload)
        if missing:
            errors.append("missing:" + ",".join(sorted(missing)))
        action = payload.get("next_action")
        if action not in RETRIEVAL_ACTIONS:
            errors.append("bad_action")
        follow_up = payload.get("follow_up_hypothesis")
        if action == "retrieve_more":
            if not isinstance(follow_up, dict):
                errors.append("missing_follow_up")
            elif follow_up.get("query_type") not in QUERY_TYPES:
                errors.append("follow_up_bad_query_type")
            if payload.get("answer"):
                errors.append("retrieve_more_has_answer")
        elif follow_up not in (None, {}):
            errors.append("unexpected_follow_up")
        extra = set(payload) - required
        if extra:
            errors.append("extra:" + ",".join(sorted(extra)))
    return errors


def evidence_brief_text(prompt: str) -> str:
    if "evidence_brief:" not in prompt:
        return ""
    tail = prompt.split("evidence_brief:", 1)[1]
    stop_markers = ("\nmissing_slots:", "\nminirag_hints:", "\noutput_schema:")
    end = len(tail)
    for marker in stop_markers:
        index = tail.find(marker)
        if index >= 0:
            end = min(end, index)
    return tail[:end].strip()


def tokens(text: str) -> list[str]:
    return [
        item
        for item in CJK_TOKEN_RE.findall(text)
        if len(item) >= 2 and item not in NOISY_TERMS and not any(noisy in item for noisy in ("什么", "为什么", "怎么"))
    ]


def generic_voice_evidence_only(evidence_text: str) -> bool:
    if not evidence_text:
        return False
    return any(tag in evidence_text for tag in GENERIC_VOICE_TAGS) and not any("#chunk-" in line for line in evidence_text.splitlines())


def quality_errors(task_type: str, prompt: str, payload: dict[str, Any] | None) -> list[str]:
    if task_type != "conclusion_generation" or not isinstance(payload, dict):
        return []
    errors: list[str] = []
    evidence = evidence_brief_text(prompt)
    question = str(payload.get("question") or "")
    action = str(payload.get("next_action") or "")
    question_terms = tokens(question)
    if evidence and question_terms and not any(term in evidence for term in question_terms[:6]):
        errors.append("question_anchor_not_in_evidence")
    if action == "answer_directly":
        if generic_voice_evidence_only(evidence) and any(term in question for term in ("身份", "关系", "为什么", "原因", "真相", "身世", "来历")):
            errors.append("voice_fragment_answer_directly")
        answer_terms = tokens(str(payload.get("answer") or ""))
        unsupported = [term for term in answer_terms[:30] if term not in evidence and term not in question]
        if len(unsupported) >= 8:
            errors.append("answer_many_terms_not_in_evidence")
    if action == "retrieve_more":
        slots = [str(item) for item in payload.get("missing_slots") or []]
        follow_up = payload.get("follow_up_hypothesis")
        follow_text = ""
        if isinstance(follow_up, dict):
            follow_text = " ".join(
                [
                    str(follow_up.get("question") or ""),
                    " ".join(str(item) for item in follow_up.get("entities") or []),
                    " ".join(str(item) for item in follow_up.get("keywords") or []),
                ]
            )
        if slots and follow_text:
            slot_terms = [term for slot in slots for term in tokens(slot)]
            if slot_terms and not any(term in follow_text for term in slot_terms[:12]):
                errors.append("missing_slots_not_aligned_with_follow_up")
    return errors


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, int(len(values) * pct))
    return values[index]


def audit_split(records: list[dict[str, Any]], *, sample_size: int) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    query_type_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    exact_counter: Counter[str] = Counter()
    minirag_hints = 0
    minirag_relation_hints = 0
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for record in records:
        task_type = str(record.get("task_type") or "")
        task_counts[task_type] += 1
        prompt = user_text(record)
        output = assistant_text(record)
        payload = parse_json(output)
        prompt_lengths.append(len(prompt))
        output_lengths.append(len(output))
        exact_counter[task_type + "\n" + prompt + "\n" + output] += 1
        if "minirag_hints:" in prompt and "minirag_hints: none" not in prompt:
            minirag_hints += 1
        if "relations=" in prompt:
            minirag_relation_hints += 1

        for error in schema_errors(task_type, payload):
            error_counts[error] += 1
            if len(examples[error]) < sample_size:
                examples[error].append({"prompt": prompt[:600], "output": output[:600]})
        for error in quality_errors(task_type, prompt, payload):
            quality_counts[error] += 1
            if len(examples[error]) < sample_size:
                examples[error].append({"prompt": prompt[:600], "output": output[:600]})

        if isinstance(payload, dict):
            if payload.get("next_action"):
                action_counts[str(payload["next_action"])] += 1
            if payload.get("query_type"):
                query_type_counts[str(payload["query_type"])] += 1
            if payload.get("intent"):
                intent_counts[str(payload["intent"])] += 1
            follow_up = payload.get("follow_up_hypothesis")
            if isinstance(follow_up, dict) and follow_up.get("query_type"):
                query_type_counts["follow_up:" + str(follow_up["query_type"])] += 1

    return {
        "records": len(records),
        "task_counts": dict(task_counts),
        "actions": dict(action_counts),
        "query_types": dict(query_type_counts),
        "intents": dict(intent_counts),
        "schema_errors": dict(error_counts),
        "quality_warnings": dict(quality_counts),
        "prompt_length": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "mean": round(mean(prompt_lengths), 1) if prompt_lengths else 0,
            "p50": int(median(prompt_lengths)) if prompt_lengths else 0,
            "p90": percentile(prompt_lengths, 0.9),
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
        "output_length": {
            "min": min(output_lengths) if output_lengths else 0,
            "mean": round(mean(output_lengths), 1) if output_lengths else 0,
            "p50": int(median(output_lengths)) if output_lengths else 0,
            "p90": percentile(output_lengths, 0.9),
            "max": max(output_lengths) if output_lengths else 0,
        },
        "minirag_hint_coverage": round(minirag_hints / len(records), 4) if records else 0.0,
        "minirag_relation_hint_coverage": round(minirag_relation_hints / len(records), 4) if records else 0.0,
        "duplicate_exact_groups": sum(1 for count in exact_counter.values() if count > 1),
        "max_duplicate_exact": max(exact_counter.values(), default=0),
        "error_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit short-prompt current-schema SFT data.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()
    random.seed(args.seed)

    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else PROJECT_ROOT / args.dataset_dir
    report: dict[str, Any] = {"dataset_dir": str(dataset_dir), "splits": {}}
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.json"
        if not path.exists():
            continue
        records = load_json(path)
        report["splits"][split] = audit_split(records, sample_size=args.sample_size)

    output = args.output
    if output is None:
        output = dataset_dir / "audit_report.json"
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
