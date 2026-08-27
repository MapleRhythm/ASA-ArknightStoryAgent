#!/usr/bin/env python3
"""Audit a frozen Exx SFT dataset before training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
QUESTION_RE = re.compile(r"^question:\s*(.+)$", re.MULTILINE)
PROTOCOL = "grounded_action_exx_v1"
SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("？?。！!，,；;：:")


def audit_grounded(row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return ["invalid_conversations"]
    prompt = str(conversations[0].get("value") or "")
    visible = set(EVIDENCE_RE.findall(prompt))
    if row.get("system") != SYSTEM:
        problems.append("non_canonical_system")
    if f"output_schema: {PROTOCOL}" not in prompt:
        problems.append("non_canonical_protocol")
    if not visible:
        problems.append("no_visible_evidence")
    try:
        payload = json.loads(str(conversations[-1].get("value") or ""))
    except json.JSONDecodeError:
        return [*problems, "invalid_assistant_json"]
    if not isinstance(payload, dict):
        return [*problems, "assistant_not_object"]
    legacy = FORBIDDEN.intersection(payload)
    if legacy:
        problems.append("legacy_fields:" + ",".join(sorted(legacy)))
    action = payload.get("next_action")
    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            problems.append("answer_top_schema")
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            problems.append("fact_count")
        else:
            for index, fact in enumerate(facts, start=1):
                if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                    problems.append(f"fact_{index}_schema")
                    continue
                ids = fact.get("evidence_ids")
                if not str(fact.get("fact") or "").strip():
                    problems.append(f"fact_{index}_empty")
                if (
                    not isinstance(ids, list)
                    or not 1 <= len(ids) <= 2
                    or len(set(map(str, ids))) != len(ids)
                    or any(str(item) not in visible for item in ids)
                ):
                    problems.append(f"fact_{index}_ids")
    elif action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            problems.append("retrieve_top_schema")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict) or not str(follow_up.get("question") or "").strip():
            problems.append("retrieve_follow_up")
    elif action == "abstain":
        if set(payload) != {"next_action", "reason"} or not str(payload.get("reason") or "").strip():
            problems.append("abstain_schema")
    else:
        problems.append("invalid_action")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_rows: dict[str, list[dict[str, Any]]] = {}
    counts = Counter()
    issues: list[dict[str, Any]] = []
    question_sets: dict[str, set[str]] = {}
    id_locations: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = args.dataset_dir / f"{split}.json"
        rows = read_rows(path) if path.exists() else []
        split_rows[split] = rows
        questions: set[str] = set()
        for index, row in enumerate(rows):
            row_id = str(row.get("id") or "")
            if not row_id:
                issues.append({"split": split, "index": index, "id": row_id, "problems": ["duplicate_or_missing_id"]})
            else:
                locations = id_locations.setdefault(row_id, set())
                # Different task types can intentionally share a base id, but
                # the same id must never repeat inside one task/split stream.
                location = f"{split}:{row.get('task_type')}"
                if location in locations:
                    issues.append({"split": split, "index": index, "id": row_id, "problems": ["duplicate_id_in_task"]})
                locations.add(location)
            if not isinstance(row.get("tools"), str) or not isinstance(row.get("meta"), str):
                issues.append({"split": split, "index": index, "id": row_id, "problems": ["non_string_tools_or_meta"]})
            task = str(row.get("task_type") or "")
            counts[f"{split}:task:{task}"] += 1
            conversations = row.get("conversations") or []
            prompt = str(conversations[0].get("value") or "") if conversations else ""
            match = QUESTION_RE.search(prompt)
            if match:
                questions.add(normalize_question(match.group(1)))
            if task == "grounded_action_generation":
                row_problems = audit_grounded(row)
                if row_problems:
                    issues.append({"split": split, "index": index, "id": row_id, "problems": row_problems})
                else:
                    payload = json.loads(str(conversations[-1]["value"]))
                    counts[f"{split}:action:{payload['next_action']}"] += 1
        question_sets[split] = questions

    overlaps = {
        "train_val": sorted(question_sets["train"] & question_sets["val"]),
        "train_test": sorted(question_sets["train"] & question_sets["test"]),
        "val_test": sorted(question_sets["val"] & question_sets["test"]),
    }
    report = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "counts": dict(sorted(counts.items())),
        "question_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "question_overlap_samples": {key: value[:20] for key, value in overlaps.items()},
        "problem_count": len(issues),
        "problems": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "problems"}, ensure_ascii=False, indent=2))
    return 1 if issues or any(overlaps.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
