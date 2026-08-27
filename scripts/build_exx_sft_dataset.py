#!/usr/bin/env python3
"""Build a versioned Exx SFT dataset from deterministic and teacher labels.

All inputs are read-only.  Teacher rows are de-duplicated by task id with the
provider precedence specified on the command line.  The held-out test split is
copied only from the deterministic source; teacher-labelled test rows are not
accepted.  Non-answer planning tasks can be sampled deterministically so they
do not dilute the Exx grounded-action objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "grounded_action_exx_v1"
ANSWER_TASK = "grounded_action_generation"
PLANNING_TASKS = {"user_question_hypothesis_generation", "follow_up_hypothesis_generation"}
EVIDENCE_RE = re.compile(r"^\[E(\d+)\]\s*$", re.MULTILINE)
FORBIDDEN_OUTPUT_KEYS = {"quote", "final_answer", "inferred_facts", "evidence_refs"}
QUESTION_RE = re.compile(r"^(?:question|问题)[:：]\s*(.+)$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_record_columns(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with homogeneous Arrow-compatible metadata columns.

    Hugging Face Datasets/Arrow requires one physical type per column.  The
    deterministic source already stores tools as a JSON string, while teacher
    rows used a list.  The free-form ``meta`` objects also have provider-
    specific nested schemas.  Neither column is a model target, so serialize
    both as compact JSON strings while retaining their audit information.
    """
    normalized = dict(record)
    for key in ("tools", "meta"):
        value = normalized.get(key, "")
        if not isinstance(value, str):
            normalized[key] = compact_json(value)
    return normalized


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    value = record["conversations"][-1]["value"]
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("assistant output is not a JSON object")
    return payload


def question_key(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    prompt = str(conversations[0].get("value", ""))
    match = QUESTION_RE.search(prompt)
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(1)).strip()


def validate_grounded_record(record: dict[str, Any]) -> str:
    if record.get("task_type") != ANSWER_TASK:
        raise ValueError("not a grounded-action record")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("invalid conversations")
    prompt = str(conversations[0].get("value", ""))
    visible = {f"E{number}" for number in EVIDENCE_RE.findall(prompt)}
    if not visible:
        raise ValueError("prompt contains no visible evidence ids")
    payload = assistant_payload(record)
    serialized = compact_json(payload)
    if any(f'"{key}"' in serialized for key in FORBIDDEN_OUTPUT_KEYS):
        raise ValueError("legacy answer field remains")
    action = payload.get("next_action")
    if action not in {"answer_directly", "retrieve_more", "abstain"}:
        raise ValueError("invalid next_action")
    if action == "answer_directly":
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            raise ValueError("invalid supported_facts count")
        for fact in facts:
            ids = fact.get("evidence_ids") if isinstance(fact, dict) else None
            if (
                not str(fact.get("fact", "")).strip()
                or not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or any(str(item) not in visible for item in ids)
            ):
                raise ValueError("invalid fact evidence binding")
    return str(action)


def dataset_info() -> dict[str, Any]:
    common = {
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
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
        f"exx_grounding_v1_sft_{split}": {"file_name": f"{split}.json", **common}
        for split in ("train", "val", "test")
    }


def choose_planning_rows(
    rows: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_type"))].append(row)
    task_names = sorted(by_task)
    selected: list[dict[str, Any]] = []
    base, extra = divmod(limit, len(task_names))
    for index, task_name in enumerate(task_names):
        task_rows = sorted(by_task[task_name], key=lambda item: str(item.get("id", "")))
        rng = random.Random(seed + index)
        rng.shuffle(task_rows)
        selected.extend(task_rows[: min(len(task_rows), base + (index < extra))])
    return selected


def teacher_record(task: dict[str, Any], result: dict[str, Any], provider: str) -> dict[str, Any]:
    evidence_lines: list[str] = []
    for item in task["evidence"]:
        evidence_lines.extend((f"[{item['label']}]", str(item["text"])))
    prompt = "\n".join(
        (
            "task: grounded_action_generation",
            f"question: {task['question']}",
            f"hypothesis: {task.get('hypothesis') or '{}'}",
            f"round: {task.get('round') or 'unknown'}",
            "evidence:",
            *evidence_lines,
            f"output_schema: {PROTOCOL}",
            "rules: 只使用当前可见证据；回答时把每个可核验原子事实绑定到1至2个当前存在的E编号；"
            "不要复制引文，不要输出quote、final_answer或inferred_facts；证据不足时才retrieve_more。",
        )
    )
    model = result.get("api", {}).get("model")
    return {
        "id": f"relabel-{task['task_id']}",
        "task_type": ANSWER_TASK,
        "bucket": "teacher_relabel_exx",
        "system": "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。",
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": compact_json(result["label"])},
        ],
        "meta": {
            "schema": PROTOCOL,
            "teacher_provider": provider,
            "teacher_model": model,
            "conversion_method": "teacher_visible_full_evidence_only",
            "source_refs": task.get("source_refs", []),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-sft-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument(
        "--teacher-labels",
        action="append",
        nargs=2,
        metavar=("PROVIDER", "JSONL"),
        default=[],
        required=True,
        help="provider precedence is left to right; first label for a task id wins",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planning-train-max", type=int, default=256)
    parser.add_argument("--planning-val-max", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {output_dir}")
    deterministic_dir = args.deterministic_sft_dir.resolve()
    tasks_path = args.tasks_json.resolve()
    tasks = {str(row["task_id"]): row for row in read_json(tasks_path)}
    provider_inputs: list[dict[str, Any]] = []
    chosen: dict[str, tuple[str, dict[str, Any]]] = {}
    duplicate_stats: Counter[str] = Counter()
    for provider, raw_path in args.teacher_labels:
        path = Path(raw_path).resolve()
        rows = load_jsonl(path)
        provider_inputs.append(
            {"provider": provider, "path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
        )
        for row in rows:
            task_id = str(row.get("task_id", ""))
            if task_id not in tasks or not isinstance(row.get("label"), dict):
                duplicate_stats[f"reject:{provider}:unknown_or_invalid"] += 1
                continue
            if task_id in chosen:
                duplicate_stats[f"duplicate:{provider}:lost_to:{chosen[task_id][0]}"] += 1
                continue
            chosen[task_id] = (provider, row)

    teacher_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_stats: Counter[str] = Counter()
    for task_id, (provider, result) in sorted(chosen.items()):
        task = tasks[task_id]
        if task["split"] == "test":
            duplicate_stats[f"reject:{provider}:teacher_test"] += 1
            continue
        record = teacher_record(task, result, provider)
        action = validate_grounded_record(record)
        teacher_by_split[task["split"]].append(record)
        action_stats[f"teacher:{provider}:{task['split']}:{action}"] += 1

    output_rows: dict[str, list[dict[str, Any]]] = {}
    deterministic_inputs: list[dict[str, Any]] = []
    output_stats: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        source_path = deterministic_dir / f"{split}.json"
        source_rows = read_json(source_path) if source_path.exists() else []
        deterministic_inputs.append(
            {
                "split": split,
                "path": str(source_path),
                "rows": len(source_rows),
                "sha256": sha256_file(source_path) if source_path.exists() else None,
            }
        )
        grounded = [row for row in source_rows if row.get("task_type") == ANSWER_TASK]
        planning = [row for row in source_rows if row.get("task_type") in PLANNING_TASKS]
        for row in grounded:
            action_stats[f"deterministic:{split}:{validate_grounded_record(row)}"] += 1
        planning_limit = (
            args.planning_train_max if split == "train" else args.planning_val_max if split == "val" else 0
        )
        sampled_planning = choose_planning_rows(
            planning, limit=planning_limit, seed=args.seed + {"train": 0, "val": 100, "test": 200}[split]
        )
        merged = grounded + teacher_by_split.get(split, []) + sampled_planning
        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in sorted(merged, key=lambda item: str(item.get("id", ""))):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in seen_ids:
                output_stats[f"reject:{split}:duplicate_or_missing_id"] += 1
                continue
            seen_ids.add(row_id)
            unique.append(normalize_record_columns(row))
            output_stats[f"output:{split}:{row.get('task_type')}"] += 1
        output_rows[split] = unique

    # The deterministic and teacher sources were isolated independently.  Do
    # one final text-level audit across their union so no train row shares a
    # question with validation or test, even when ids came from different
    # source datasets.
    held_out_questions = {
        key
        for split in ("val", "test")
        for row in output_rows[split]
        if (key := question_key(row))
    }
    isolated_train: list[dict[str, Any]] = []
    for row in output_rows["train"]:
        key = question_key(row)
        if key and key in held_out_questions:
            output_stats["reject:train:held_out_question_overlap"] += 1
            continue
        isolated_train.append(row)
    output_rows["train"] = isolated_train

    # Recompute accepted-output statistics after the final cross-split
    # isolation. Rejection counters are retained, while pre-isolation output
    # counters must not leak into the frozen dataset audit.
    output_stats = Counter(
        {
            key: count
            for key, count in output_stats.items()
            if key.startswith("reject:")
        }
    )
    for split, rows in output_rows.items():
        for row in rows:
            output_stats[f"output:{split}:{row.get('task_type')}"] += 1

    output_dir.mkdir(parents=True)
    for split, rows in output_rows.items():
        write_json(output_dir / f"{split}.json", rows)
    write_json(output_dir / "dataset_info.json", dataset_info())
    audit = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "inputs_read_only": True,
            "teacher_provider_precedence": [provider for provider, _ in args.teacher_labels],
            "teacher_test_rows_allowed": False,
            "full_evidence_preserved": True,
            "planning_train_max": args.planning_train_max,
            "planning_val_max": args.planning_val_max,
            "seed": args.seed,
        },
        "inputs": {
            "deterministic": deterministic_inputs,
            "tasks": {"path": str(tasks_path), "rows": len(tasks), "sha256": sha256_file(tasks_path)},
            "teachers": provider_inputs,
        },
        "selected_unique_teacher_labels": len(chosen),
        "deduplication": dict(sorted(duplicate_stats.items())),
        "actions": dict(sorted(action_stats.items())),
        "outputs": dict(sorted(output_stats.items())),
        "output_counts": {split: len(rows) for split, rows in output_rows.items()},
    }
    write_json(output_dir / "audit.json", audit)
    # The manifest deliberately excludes itself and the checksum list to
    # avoid recursive hashes. Any post-build audit (for example an exact
    # tokenizer-length audit) must be written before calling this helper or
    # followed by regenerating these two metadata files.
    manifest_inputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "final_checksums.sha256"}
    )
    write_json(
        output_dir / "manifest.json",
        {
            "protocol": PROTOCOL,
            "created_at_utc": audit["created_at_utc"],
            "hash_policy": {
                "algorithm": "sha256",
                "manifest_excludes": ["manifest.json", "final_checksums.sha256"],
                "checksums_excludes": ["final_checksums.sha256"],
            },
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in manifest_inputs
            ],
        },
    )
    checksum_inputs = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "final_checksums.sha256"
    )
    with (output_dir / "final_checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in checksum_inputs:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
