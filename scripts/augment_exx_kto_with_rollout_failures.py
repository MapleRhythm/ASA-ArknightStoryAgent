#!/usr/bin/env python3
"""Augment a frozen Exx KTO dataset with objective SFT rollout failures.

Only failures whose polarity can be established without judging free-form
answer wording are retained: invalid protocol/JSON, generation truncation, or
an action that disagrees with the teacher action.  Each new negative keeps the
exact model-visible SFT prompt and is paired with that prompt's teacher output.
The existing KTO validation split is copied unchanged and its questions/story
families are excluded from augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
QUESTION_RE = re.compile(r"^question:\s*(.+)$", re.MULTILINE)
ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_key(row: dict[str, Any]) -> str:
    return str(row.get("system") or "") + "\n" + str(row["conversations"][0]["value"])


def question_from_prompt(prompt: str) -> str:
    match = QUESTION_RE.search(prompt)
    return (
        re.sub(r"\s+", "", match.group(1)).strip("？?。！!，,；;：:")
        if match
        else ""
    )


def normalize_story_family(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/").split("#", 1)[0]
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return ""
    if parts[0] == "activities" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] == "obt" and len(parts) > 2:
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def task_story_families(task: dict[str, Any]) -> set[str]:
    return {
        family
        for item in task.get("evidence") or []
        if (family := normalize_story_family(str(item.get("doc_id") or "")))
    }


def parse_payload(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def protocol_problems(text: str, prompt: str) -> list[str]:
    payload = parse_payload(text)
    if payload is None:
        return ["invalid_json"]
    problems: list[str] = []
    forbidden = FORBIDDEN.intersection(payload)
    if forbidden:
        problems.append("legacy_fields:" + ",".join(sorted(forbidden)))
    action = str(payload.get("next_action") or "")
    if action not in ACTIONS:
        return problems + ["invalid_action"]
    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            problems.append("invalid_answer_top_schema")
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return problems + ["invalid_fact_count"]
        visible = set(EVIDENCE_RE.findall(prompt))
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
            ):
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible for item in ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            problems.append("invalid_retrieve_top_schema")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict) or not str(follow_up.get("question") or "").strip():
            problems.append("invalid_follow_up_hypothesis")
    else:
        if set(payload) != {"next_action", "reason"}:
            problems.append("invalid_abstain_top_schema")
        if not str(payload.get("reason") or "").strip():
            problems.append("invalid_abstain_reason")
    return problems


def task_id_from_sft_id(row_id: str) -> str:
    return row_id[len("relabel-") :] if row_id.startswith("relabel-") else ""


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-kto-dir", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--rollout-predictions", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-combined-tokens", type=int, default=4800)
    parser.add_argument("--max-negatives-per-prompt", type=int, default=3)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {args.output_dir}")
    if args.max_combined_tokens < 1 or args.max_negatives_per_prompt < 1:
        parser.error("token and negative caps must be positive")

    base_train_all = read_json(args.base_kto_dir / "train.json")
    base_val = read_json(args.base_kto_dir / "val.json")
    sft_eval_path = args.sft_train.with_name("val.json")
    sft_eval_questions = {
        question_from_prompt(str(row["conversations"][0]["value"]))
        for row in read_json(sft_eval_path)
    }
    base_train = [
        row
        for row in base_train_all
        if question_from_prompt(str(row["conversations"][0]["value"])) not in sft_eval_questions
    ]
    sft_rows = {
        str(row.get("id")): row
        for row in read_json(args.sft_train)
        if row.get("task_type") == "grounded_action_generation"
    }
    rollout_rows = {str(row.get("id")): row for row in read_json(args.rollout_predictions)}
    tasks = {str(task["task_id"]): task for task in read_json(args.tasks_json)}

    val_questions = {
        question_from_prompt(str(row["conversations"][0]["value"])) for row in base_val
    }
    val_families: set[str] = set()
    for row in base_val:
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = json.loads(meta)
        val_families.update(str(item) for item in (meta or {}).get("story_families") or [])

    output_train = list(base_train)
    rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_train:
        rows_by_prompt[prompt_key(row)].append(row)
    stats: Counter[str] = Counter()
    added_records: list[dict[str, Any]] = []

    for row_id, gold_row in sorted(sft_rows.items()):
        rollout = rollout_rows.get(row_id)
        if rollout is None:
            stats["reject:missing_rollout"] += 1
            continue
        task_id = task_id_from_sft_id(row_id)
        task = tasks.get(task_id)
        if task is None:
            stats["reject:not_semantic_relabel_task"] += 1
            continue
        if task_story_families(task) & val_families:
            stats["reject:val_story_family_overlap"] += 1
            continue
        prompt = str(gold_row["conversations"][0]["value"])
        if question_from_prompt(prompt) in val_questions:
            stats["reject:val_question_overlap"] += 1
            continue

        gold_text = str(gold_row["conversations"][-1]["value"])
        negative_text = str(rollout.get("raw_output") or rollout["conversations"][-1]["value"])
        gold_payload = parse_payload(gold_text)
        if gold_payload is None or str(gold_payload.get("next_action") or "") not in ACTIONS:
            stats["reject:invalid_teacher_positive"] += 1
            continue
        negative_payload = parse_payload(negative_text)
        negative_action = str((negative_payload or {}).get("next_action") or "invalid")
        gold_action = str(gold_payload["next_action"])
        problems = protocol_problems(negative_text, prompt)
        finish_reason = str(rollout.get("finish_reason") or "")
        reasons = list(problems)
        if finish_reason == "length":
            reasons.append("generation_truncated")
        if negative_action != gold_action:
            reasons.append(f"action_mismatch:{gold_action}->{negative_action}")
        if not reasons:
            stats["reject:no_objective_failure"] += 1
            continue
        if negative_text.strip() == gold_text.strip():
            stats["reject:negative_equals_positive"] += 1
            continue
        combined_tokens = int(rollout.get("prompt_token_count") or 0) + int(
            rollout.get("generated_token_count") or 0
        )
        if combined_tokens > args.max_combined_tokens:
            stats["reject:combined_token_cap"] += 1
            continue

        key = prompt_key(gold_row)
        existing = rows_by_prompt[key]
        existing_negatives = [item for item in existing if item.get("kto_tag") is False]
        if len(existing_negatives) >= args.max_negatives_per_prompt:
            stats["reject:negative_cap"] += 1
            continue
        if any(str(item["conversations"][-1]["value"]).strip() == negative_text.strip() for item in existing):
            stats["reject:duplicate_output"] += 1
            continue

        families = sorted(task_story_families(task))
        if not existing:
            positive = {
                **{key_: value for key_, value in gold_row.items() if key_ not in {"id", "meta", "token_len"}},
                "id": f"exx-kto-{task_id}-rollout-teacher-positive",
                "bucket": "success559_rollout_failure_kto",
                "kto_tag": True,
                "meta": compact_json(
                    {
                        "schema": "grounded_action_exx_v1",
                        "task_id": task_id,
                        "story_families": families,
                        "preference_polarity": "desirable",
                        "source": "success559_sft_teacher_positive",
                        "source_row_id": row_id,
                    }
                ),
            }
            output_train.append(positive)
            rows_by_prompt[key].append(positive)
            stats[f"add:positive:{gold_action}"] += 1

        negative = {
            **{key_: value for key_, value in gold_row.items() if key_ not in {"id", "meta", "token_len"}},
            "id": f"exx-kto-{task_id}-success559-rollout-negative",
            "bucket": "success559_rollout_failure_kto",
            "kto_tag": False,
            "conversations": [gold_row["conversations"][0], {"from": "gpt", "value": negative_text}],
            "meta": compact_json(
                {
                    "schema": "grounded_action_exx_v1",
                    "task_id": task_id,
                    "story_families": families,
                    "preference_polarity": "undesirable",
                    "source": "success559_deterministic_rollout_objective_failure",
                    "source_row_id": row_id,
                    "failure_reasons": sorted(set(reasons)),
                    "gold_action": gold_action,
                    "negative_action": negative_action,
                    "finish_reason": finish_reason,
                    "prompt_token_count": rollout.get("prompt_token_count"),
                    "generated_token_count": rollout.get("generated_token_count"),
                }
            ),
        }
        output_train.append(negative)
        rows_by_prompt[key].append(negative)
        stats[f"add:negative:{negative_action}"] += 1
        for reason in sorted(set(reasons)):
            stats[f"failure:{reason}"] += 1
        added_records.append(
            {
                "id": negative["id"],
                "source_row_id": row_id,
                "gold_action": gold_action,
                "negative_action": negative_action,
                "failure_reasons": sorted(set(reasons)),
                "combined_tokens": combined_tokens,
            }
        )

    # Every retained prompt must have both polarities; this also protects the
    # original frozen rows if their format changes unexpectedly.
    labels_by_prompt: dict[str, set[bool]] = defaultdict(set)
    for row in output_train:
        labels_by_prompt[prompt_key(row)].add(bool(row.get("kto_tag")))
    if any(labels != {False, True} for labels in labels_by_prompt.values()):
        raise RuntimeError("augmentation produced an unpaired prompt")

    output_train.sort(key=lambda item: str(item.get("id") or ""))
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "train.json", output_train)
    write_json(args.output_dir / "val.json", base_val)
    write_json(args.output_dir / "dataset_info.json", dataset_info())
    report = {
        "protocol": "grounded_action_exx_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "objective_failures_only": True,
            "same_prompt_teacher_pair_required": True,
            "sft_eval_rows_excluded": True,
            "base_kto_val_unchanged": True,
            "val_question_and_story_family_isolation": True,
            "semantic_free_form_disagreement_not_used": True,
            "max_combined_tokens": args.max_combined_tokens,
            "max_negatives_per_prompt": args.max_negatives_per_prompt,
        },
        "inputs": {
            "base_kto_dir": str(args.base_kto_dir.resolve()),
            "sft_train": {"path": str(args.sft_train.resolve()), "sha256": sha256_file(args.sft_train)},
            "rollout_predictions": {
                "path": str(args.rollout_predictions.resolve()),
                "sha256": sha256_file(args.rollout_predictions),
            },
            "tasks": {"path": str(args.tasks_json.resolve()), "sha256": sha256_file(args.tasks_json)},
        },
        "base_counts": {
            "train_before_sft_eval_filter": len(base_train_all),
            "train_after_sft_eval_filter": len(base_train),
            "val": len(base_val),
        },
        "output_counts": {"train": len(output_train), "val": len(base_val)},
        "paired_prompt_counts": {"train": len(labels_by_prompt)},
        "stats": dict(sorted(stats.items())),
        "added_records": added_records,
    }
    write_json(args.output_dir / "audit.json", report)
    manifest_paths = [args.output_dir / name for name in ("train.json", "val.json", "dataset_info.json", "audit.json")]
    write_json(
        args.output_dir / "manifest.json",
        {
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in manifest_paths
            ]
        },
    )
    checksum_paths = manifest_paths + [args.output_dir / "manifest.json"]
    with (args.output_dir / "final_checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "added_records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
