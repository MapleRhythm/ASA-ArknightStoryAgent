#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llama_factory"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llama_factory"
    / "teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1_planner_rebalanced"
)

CONCLUSION_TASK_TYPE = "conclusion_generation"
PSEUDO_QUERY_TERMS = {
    "标题",
    "生日",
    "闲置",
    "戳一下",
    "行动出发",
    "任命队长",
    "任命助理",
    "编入队伍",
    "交谈1",
    "交谈2",
    "交谈3",
    "信赖触摸",
}
GENERIC_MISSING_SLOT_PATTERNS = (
    "更多信息",
    "相关背景",
    "详细资料",
    "完整剧情",
    "更直接的人物身份证据",
    "与主实体直接相关的桥接信息",
)


def load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def save_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def assistant_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if message.get("from") != "gpt":
            continue
        value = message.get("value")
        if not isinstance(value, str):
            return None
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def last_human_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if message.get("from") == "human":
            return str(message.get("value") or "")
    return ""


def retrieval_round(record: dict[str, Any]) -> int:
    match = re.search(r"当前检索轮次[:：]\s*第(\d+)轮\s*/\s*最多(\d+)轮", last_human_text(record))
    if not match:
        return 0
    return int(match.group(1))


def is_pseudo_identity_question(question: str) -> bool:
    normalized = question.strip()
    if not normalized:
        return True
    identity_like = any(token in normalized for token in ("是什么身份", "是谁", "是什么人", "是什么角色"))
    if not identity_like:
        return False
    return any(term in normalized for term in PSEUDO_QUERY_TERMS)


def is_generic_missing_slot(slot: Any) -> bool:
    if not isinstance(slot, str):
        return True
    slot = slot.strip()
    if not slot:
        return True
    return any(pattern in slot for pattern in GENERIC_MISSING_SLOT_PATTERNS)


def retrieve_more_quality_score(record: dict[str, Any], payload: dict[str, Any]) -> float:
    question = str(payload.get("question") or "")
    missing_slots = payload.get("missing_slots") or []
    follow_up = payload.get("follow_up_hypothesis") or {}
    score = 0.0

    if is_pseudo_identity_question(question):
        return -100.0
    if not isinstance(missing_slots, list) or not (1 <= len(missing_slots) <= 5):
        score -= 10.0
    else:
        score += min(len(missing_slots), 3)
        score -= sum(2.0 for slot in missing_slots if is_generic_missing_slot(slot))

    if isinstance(follow_up, dict):
        entities = follow_up.get("entities") or []
        keywords = follow_up.get("keywords") or []
        if isinstance(entities, list) and entities:
            score += min(len(entities), 4) * 0.7
        else:
            score -= 4.0
        if isinstance(keywords, list) and len(keywords) >= 3:
            score += 1.0
    else:
        score -= 8.0

    if any(token in question for token in ("为什么", "为何", "关系", "真相", "一事", "事件", "如何", "怎么")):
        score += 1.2
    if "是什么身份" in question or "是谁" in question:
        score -= 0.8

    round_no = retrieval_round(record)
    if round_no == 1:
        score += 0.3
    elif round_no == 2:
        score += 0.1
    elif round_no >= 3:
        score -= 2.0

    return score


def should_drop_record(record: dict[str, Any], payload: dict[str, Any] | None) -> tuple[bool, str]:
    if record.get("task_type") != CONCLUSION_TASK_TYPE:
        return False, ""
    if payload is None:
        return True, "invalid_assistant_json"
    question = str(payload.get("question") or "")
    if is_pseudo_identity_question(question):
        return True, "pseudo_identity_question"
    action = payload.get("next_action")
    if action not in {"answer_directly", "retrieve_more", "clarify_user", "abstain"}:
        return True, "invalid_next_action"
    return False, ""


def rebalance_split(
    records: list[dict[str, Any]],
    *,
    target_retrieve_more_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    kept_non_conclusion: list[dict[str, Any]] = []
    keep_conclusion: list[dict[str, Any]] = []
    retrieve_candidates: list[tuple[float, float, dict[str, Any]]] = []
    dropped = Counter()
    before_actions = Counter()

    for record in records:
        payload = assistant_payload(record)
        if record.get("task_type") == CONCLUSION_TASK_TYPE and payload is not None:
            before_actions[str(payload.get("next_action") or "")] += 1
        drop, reason = should_drop_record(record, payload)
        if drop:
            dropped[reason] += 1
            continue
        if record.get("task_type") != CONCLUSION_TASK_TYPE:
            kept_non_conclusion.append(record)
            continue

        assert payload is not None
        action = payload.get("next_action")
        if action == "retrieve_more":
            score = retrieve_more_quality_score(record, payload)
            if score < 0:
                dropped["low_quality_retrieve_more"] += 1
                continue
            retrieve_candidates.append((score, rng.random(), record))
        else:
            keep_conclusion.append(record)

    non_retrieve_conclusion = len(keep_conclusion)
    if target_retrieve_more_ratio <= 0:
        target_retrieve_count = 0
    elif target_retrieve_more_ratio >= 1:
        target_retrieve_count = len(retrieve_candidates)
    else:
        target_retrieve_count = round(
            (target_retrieve_more_ratio * non_retrieve_conclusion)
            / (1.0 - target_retrieve_more_ratio)
        )
    target_retrieve_count = min(target_retrieve_count, len(retrieve_candidates))
    retrieve_candidates.sort(key=lambda item: (-item[0], item[1]))
    selected_retrieve = [record for _, _, record in retrieve_candidates[:target_retrieve_count]]
    dropped["sampled_out_retrieve_more"] += max(0, len(retrieve_candidates) - len(selected_retrieve))

    output = kept_non_conclusion + keep_conclusion + selected_retrieve
    rng.shuffle(output)

    after_actions = Counter()
    task_types = Counter()
    for record in output:
        task_types[str(record.get("task_type") or "")] += 1
        if record.get("task_type") == CONCLUSION_TASK_TYPE:
            payload = assistant_payload(record)
            if payload is not None:
                after_actions[str(payload.get("next_action") or "")] += 1

    summary = {
        "input_records": len(records),
        "output_records": len(output),
        "task_types": dict(task_types),
        "before_conclusion_actions": dict(before_actions),
        "after_conclusion_actions": dict(after_actions),
        "retrieve_more_candidates_after_quality_filter": len(retrieve_candidates),
        "target_retrieve_more_ratio": target_retrieve_more_ratio,
        "dropped": dict(dropped),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a model-planner SFT copy with fewer bad retrieve_more examples. "
            "This keeps 4B in charge of retrieval planning; it only filters/rebalances training data."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-retrieve-more-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    for split, seed_offset in (("train", 0), ("val", 1)):
        input_path = args.input_dir / f"{split}.json"
        if not input_path.exists():
            continue
        records = load_json(input_path)
        output, summary = rebalance_split(
            records,
            target_retrieve_more_ratio=args.target_retrieve_more_ratio,
            seed=args.seed + seed_offset,
        )
        save_json(args.output_dir / f"{split}.json", output)
        summaries[split] = summary

    if not summaries:
        raise FileNotFoundError(f"No train.json/val.json found in {args.input_dir}")

    summary_path = args.output_dir / "rebalance_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
