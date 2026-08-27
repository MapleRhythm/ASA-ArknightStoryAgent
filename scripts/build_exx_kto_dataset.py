#!/usr/bin/env python3
"""Build a conservative Exx KTO dataset from teacher labels and legacy negatives.

The teacher's corrected label is always the desirable response.  A legacy
response is retained as undesirable only when it can be converted to the Exx
protocol from the same model-visible prompt and its original ``kto_tag`` is
explicitly false.  Rows whose polarity or evidence binding cannot be proven
are quarantined instead of being assigned a guessed preference label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELABEL_PATH = Path(__file__).with_name("relabel_conclusion_to_exx.py")
SPEC = importlib.util.spec_from_file_location("asa_exx_relabel_for_kto", RELABEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import relabel pipeline: {RELABEL_PATH}")
RELABEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RELABEL
SPEC.loader.exec_module(RELABEL)

PROTOCOL = RELABEL.PROTOCOL
ANSWER_TASK = "grounded_action_generation"
QUESTION_RE = re.compile(r"^question:\s*(.+)$", re.MULTILINE)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def dataset_info() -> dict[str, Any]:
    common = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
            "kto_tag": "kto_tag",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
            "function_tag": "function_call",
        },
    }
    return {
        f"exx_grounding_v1_kto_{split}": {"file_name": f"{split}.json", **common}
        for split in ("train", "val")
    }


def task_prompt(task: dict[str, Any]) -> str:
    evidence_text = "\n".join(
        f"[{item['label']}]\n{item['text']}" for item in task["evidence"]
    )
    return RELABEL.render_exx_user_prompt(
        question=str(task["question"]),
        hypothesis=task.get("hypothesis"),
        round_value=str(task.get("round") or "unknown"),
        evidence_text=evidence_text,
    )


def make_row(
    *,
    task: dict[str, Any],
    payload: dict[str, Any],
    desirable: bool,
    suffix: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"exx-kto-{task['task_id']}-{suffix}",
        "task_type": ANSWER_TASK,
        "bucket": "teacher_relabel_exx_kto",
        "system": RELABEL.SYSTEM_PROMPT,
        "tools": "[]",
        "kto_tag": desirable,
        "conversations": [
            {"from": "human", "value": task_prompt(task)},
            {"from": "gpt", "value": compact_json(payload)},
        ],
        "meta": compact_json(
            {
                "schema": PROTOCOL,
                "task_id": task["task_id"],
                "preference_polarity": "desirable" if desirable else "undesirable",
                **metadata,
            }
        ),
    }


def source_index(source_dirs: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for source_dir in source_dirs:
        for split in ("train", "val", "test"):
            path = source_dir / f"{split}.json"
            if not path.exists():
                continue
            for row in read_json(path):
                row_id = str(row.get("id") or "")
                if row_id:
                    indexed[(str(path.resolve()), row_id)] = row
    return indexed


def convert_legacy_negative(source: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    if source.get("kto_tag") is not False:
        raise ValueError("legacy_polarity_not_explicitly_negative")
    legacy = RELABEL.CONVERTER.parse_assistant(source)
    action = str(legacy.get("next_action") or "")
    if action == "answer_directly":
        raise ValueError("legacy_answer_has_no_deterministic_exx_binding")
    if action == "retrieve_more":
        follow_up = legacy.get("follow_up_hypothesis") or legacy.get("follow_up_query")
        if not isinstance(follow_up, dict) or not str(follow_up.get("question") or "").strip():
            raise ValueError("legacy_retrieve_missing_follow_up")
        candidate = {"next_action": "retrieve_more", "follow_up_hypothesis": follow_up}
    elif action == "abstain":
        candidate = {"next_action": "abstain", "reason": "现有证据不足以确认。"}
    else:
        raise ValueError("legacy_action_not_convertible")
    task_object = RELABEL.Task(
        task_id=str(task["task_id"]),
        split=str(task["split"]),
        question=str(task["question"]),
        hypothesis=str(task.get("hypothesis") or ""),
        round_value=str(task.get("round") or ""),
        evidence=[RELABEL.CONVERTER.Evidence(**item) for item in task["evidence"]],
        source_refs=list(task.get("source_refs") or []),
    )
    return RELABEL.validate_label(candidate, task_object)


def normalize_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt)
    return re.sub(r"\s+", "", match.group(1)).strip() if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--teacher-labels", action="append", type=Path, required=True)
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {output_dir}")
    tasks_path = args.tasks_json.resolve()
    tasks = {str(item["task_id"]): item for item in read_json(tasks_path)}
    sources = source_index([path.resolve() for path in args.source_dir])

    labels: dict[str, tuple[Path, dict[str, Any]]] = {}
    label_inputs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for path_arg in args.teacher_labels:
        path = path_arg.resolve()
        rows = load_jsonl(path)
        label_inputs.append({"path": str(path), "rows": len(rows), "sha256": sha256_file(path)})
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id not in tasks or not isinstance(row.get("label"), dict):
                stats["reject:teacher_unknown_or_invalid"] += 1
                continue
            if task_id in labels:
                stats["reject:teacher_duplicate_lower_precedence"] += 1
                continue
            labels[task_id] = (path, row)

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quarantine: list[dict[str, Any]] = []
    for task_id, (label_path, result) in sorted(labels.items()):
        task = tasks[task_id]
        split = str(task["split"])
        if split not in {"train", "val"}:
            stats["reject:teacher_test"] += 1
            continue
        task_object = RELABEL.Task(
            task_id=task_id,
            split=split,
            question=str(task["question"]),
            hypothesis=str(task.get("hypothesis") or ""),
            round_value=str(task.get("round") or ""),
            evidence=[RELABEL.CONVERTER.Evidence(**item) for item in task["evidence"]],
            source_refs=list(task.get("source_refs") or []),
        )
        positive = RELABEL.validate_label(result["label"], task_object)
        rows_by_split[split].append(
            make_row(
                task=task,
                payload=positive,
                desirable=True,
                suffix="teacher-positive",
                metadata={
                    "source": "semantic_teacher_final_label",
                    "teacher_labels_path": str(label_path),
                    "teacher_model": result.get("api", {}).get("model"),
                },
            )
        )
        stats[f"output:{split}:positive:{positive['next_action']}"] += 1

        negative_payloads: set[str] = set()
        for ref in task.get("source_refs") or []:
            source = sources.get((str(Path(ref["path"]).resolve()), str(ref["id"])))
            if source is None:
                stats["quarantine:source_ref_not_found"] += 1
                continue
            try:
                negative = convert_legacy_negative(source, task)
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                stats[f"quarantine:{reason}"] += 1
                quarantine.append(
                    {
                        "task_id": task_id,
                        "source_path": ref["path"],
                        "source_id": ref["id"],
                        "legacy_kto_tag": source.get("kto_tag"),
                        "reason": reason,
                    }
                )
                continue
            serialized = compact_json(negative)
            if serialized == compact_json(positive):
                stats["quarantine:negative_equals_positive"] += 1
                continue
            if serialized in negative_payloads:
                stats["quarantine:duplicate_negative_for_task"] += 1
                continue
            negative_payloads.add(serialized)
            rows_by_split[split].append(
                make_row(
                    task=task,
                    payload=negative,
                    desirable=False,
                    suffix=f"legacy-negative-{len(negative_payloads)}",
                    metadata={
                        "source": "explicit_legacy_kto_negative",
                        "source_path": ref["path"],
                        "source_id": ref["id"],
                    },
                )
            )
            stats[f"output:{split}:negative:{negative['next_action']}"] += 1

    # Keep every normalized question on only one side. Validation ownership wins.
    val_questions = {
        normalize_question(row["conversations"][0]["value"]) for row in rows_by_split["val"]
    }
    isolated_train: list[dict[str, Any]] = []
    for row in rows_by_split["train"]:
        if normalize_question(row["conversations"][0]["value"]) in val_questions:
            stats["quarantine:train_val_question_overlap"] += 1
            continue
        isolated_train.append(row)
    rows_by_split["train"] = isolated_train

    output_dir.mkdir(parents=True)
    for split in ("train", "val"):
        rows_by_split[split].sort(key=lambda item: item["id"])
        write_json(output_dir / f"{split}.json", rows_by_split[split])
    write_json(output_dir / "dataset_info.json", dataset_info())
    with (output_dir / "quarantine.jsonl").open("w", encoding="utf-8") as handle:
        for row in quarantine:
            handle.write(compact_json(row) + "\n")
    report = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "teacher_final_label_is_desirable": True,
            "legacy_tag_is_never_copied_to_teacher_label": True,
            "negative_requires_explicit_false_legacy_tag": True,
            "negative_requires_same_full_evidence_task": True,
            "unprovable_polarity_is_quarantined": True,
            "legacy_answer_directly_without_exx_binding_is_quarantined": True,
            "train_val_question_isolation": True,
        },
        "inputs": {
            "tasks": {"path": str(tasks_path), "rows": len(tasks), "sha256": sha256_file(tasks_path)},
            "teacher_labels": label_inputs,
            "source_dirs": [str(path.resolve()) for path in args.source_dir],
        },
        "stats": dict(sorted(stats.items())),
        "output_counts": {split: len(rows_by_split[split]) for split in ("train", "val")},
        "quarantine_rows": len(quarantine),
    }
    write_json(output_dir / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
