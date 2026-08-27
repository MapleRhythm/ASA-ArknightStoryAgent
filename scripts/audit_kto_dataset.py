#!/usr/bin/env python3
"""Read-only structural and protocol audit for LLaMA-Factory KTO datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUESTION_PATTERNS = (
    re.compile(r"^question:\s*(.+)$", re.MULTILINE),
    re.compile(r"^(?:用户原问题|问题)[:：]\s*(.+)$", re.MULTILINE),
)
EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
LEGACY_KEYS = {"quote", "final_answer", "evidence_refs", "inferred_facts", "answer"}
EXX_TASK = "grounded_action_generation"
EXX_PROTOCOL = "grounded_action_exx_v1"
EXX_SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"


def load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def conversation_text(row: dict[str, Any], *, assistant: bool) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        index = -1 if assistant else 0
        return str(conversations[index].get("value") or "")
    return ""


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("？?。！!，,；;：:")


def extract_question(prompt: str) -> str:
    for pattern in QUESTION_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return normalize_question(match.group(1))
    return ""


def nested_keys(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            output.add(str(key))
            output.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            output.update(nested_keys(child))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-eval-json", action="append", type=Path, default=[])
    args = parser.parse_args()

    counts = Counter()
    problems = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_sets: dict[str, set[str]] = defaultdict(set)
    signature_labels: dict[str, set[bool]] = defaultdict(set)
    prompt_labels: dict[str, set[bool]] = defaultdict(set)
    row_records: list[dict[str, Any]] = []

    held_out_questions: set[str] = set()
    for path in args.sft_eval_json:
        for row in load(path):
            question = extract_question(conversation_text(row, assistant=False))
            if question:
                held_out_questions.add(question)

    existing_splits: set[str] = set()
    for split in ("train", "val", "test"):
        path = args.dataset_dir / f"{split}.json"
        if not path.exists():
            continue
        existing_splits.add(split)
        for index, row in enumerate(load(path)):
            counts[f"rows:{split}"] += 1
            prompt = conversation_text(row, assistant=False)
            assistant_text = conversation_text(row, assistant=True)
            question = extract_question(prompt)
            if question:
                question_sets[split].add(question)
            else:
                problems["missing_question"] += 1
            tag = row.get("kto_tag")
            if not isinstance(tag, bool):
                problems["kto_tag_not_bool"] += 1
            else:
                counts[f"tag:{split}:{str(tag).lower()}"] += 1
            try:
                payload = json.loads(assistant_text)
            except json.JSONDecodeError:
                payload = None
                problems["assistant_invalid_json"] += 1
            action = str(payload.get("next_action") or "") if isinstance(payload, dict) else ""
            if action:
                counts[f"action:{split}:{str(tag).lower()}:{action}"] += 1
            visible = set(EVIDENCE_RE.findall(prompt))
            row_problems: list[str] = []
            if row.get("task_type") == EXX_TASK:
                if row.get("system") != EXX_SYSTEM:
                    row_problems.append("non_canonical_exx_system")
                if f"output_schema: {EXX_PROTOCOL}" not in prompt:
                    row_problems.append("non_canonical_exx_protocol")
                if not visible:
                    row_problems.append("no_visible_evidence")
            if isinstance(payload, dict):
                legacy = LEGACY_KEYS.intersection(nested_keys(payload))
                if legacy:
                    row_problems.append("legacy_fields:" + ",".join(sorted(legacy)))
                if action == "answer_directly":
                    if set(payload) != {"next_action", "supported_facts"}:
                        row_problems.append("invalid_answer_top_schema")
                    facts = payload.get("supported_facts")
                    if isinstance(facts, list) and 1 <= len(facts) <= 8:
                        for fact in facts:
                            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                                row_problems.append("invalid_fact_schema")
                                continue
                            ids = fact.get("evidence_ids")
                            if not str(fact.get("fact") or "").strip():
                                row_problems.append("empty_fact")
                            if (
                                not isinstance(ids, list)
                                or not 1 <= len(ids) <= 2
                                or len(set(map(str, ids))) != len(ids)
                            ):
                                row_problems.append("invalid_evidence_ids")
                            elif any(str(item) not in visible for item in ids):
                                row_problems.append("unknown_evidence_id")
                    else:
                        row_problems.append("invalid_supported_facts")
                elif action == "retrieve_more":
                    if set(payload) != {"next_action", "follow_up_hypothesis"}:
                        row_problems.append("invalid_retrieve_top_schema")
                    follow_up = payload.get("follow_up_hypothesis")
                    if not isinstance(follow_up, dict) or not str(follow_up.get("question") or "").strip():
                        row_problems.append("invalid_follow_up_hypothesis")
                elif action == "abstain":
                    if set(payload) != {"next_action", "reason"}:
                        row_problems.append("invalid_abstain_top_schema")
                    if not str(payload.get("reason") or "").strip():
                        row_problems.append("invalid_abstain_reason")
                else:
                    row_problems.append("invalid_next_action")
            signature = question + "\n" + prompt + "\n" + assistant_text
            if isinstance(tag, bool):
                signature_labels[signature].add(tag)
                prompt_labels[str(row.get("system") or "") + "\n" + prompt].add(tag)
            if question in held_out_questions:
                row_problems.append("sft_eval_question_leakage")
            for problem in sorted(set(row_problems)):
                problems[problem] += 1
                if len(samples[problem]) < 20:
                    samples[problem].append({"split": split, "index": index, "id": row.get("id")})
            row_records.append(
                {
                    "split": split,
                    "index": index,
                    "id": row.get("id"),
                    "question": question,
                    "kto_tag": tag,
                    "action": action,
                    "problems": sorted(set(row_problems)),
                }
            )

    for labels in signature_labels.values():
        if len(labels) > 1:
            problems["identical_prompt_output_conflicting_tag"] += 1
    for labels in prompt_labels.values():
        if True not in labels:
            problems["preference_prompt_without_positive"] += 1
        if False not in labels:
            problems["preference_prompt_without_negative"] += 1
    overlap = {
        "train_val": len(question_sets["train"] & question_sets["val"]),
        "train_test": len(question_sets["train"] & question_sets["test"]),
        "val_test": len(question_sets["val"] & question_sets["test"]),
    }
    for split in existing_splits:
        positive = counts[f"tag:{split}:true"]
        negative = counts[f"tag:{split}:false"]
        if positive == 0:
            problems[f"missing_positive_rows:{split}"] += 1
        if negative == 0:
            problems[f"missing_negative_rows:{split}"] += 1
    report = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "counts": dict(sorted(counts.items())),
        "problems": dict(sorted(problems.items())),
        "problem_samples": dict(sorted(samples.items())),
        "question_overlap_counts": overlap,
        "held_out_sft_questions": len(held_out_questions),
        "rows": row_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"rows"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if problems or any(overlap.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
