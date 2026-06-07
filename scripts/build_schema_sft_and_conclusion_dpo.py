#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_CONCLUSION_KEYS = {
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
}
VALID_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
ABSTAIN_MARKERS = (
    "现有检索证据不足",
    "现有证据不足",
    "不足以确认",
    "无法确认",
    "无法判断",
    "不能确认",
    "没有足够",
    "未能找到",
    "无法回答",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stable_hash(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def dataset_info_sft(dataset_name: str) -> dict[str, Any]:
    entry = {
        "file_name": "",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
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
    train = copy.deepcopy(entry)
    val = copy.deepcopy(entry)
    train["file_name"] = "train.json"
    val["file_name"] = "val.json"
    return {f"{dataset_name}_train": train, f"{dataset_name}_val": val}


def dataset_info_dpo(dataset_name: str) -> dict[str, Any]:
    entry = {
        "file_name": "",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
            "tools": "tools",
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
    train = copy.deepcopy(entry)
    val = copy.deepcopy(entry)
    train["file_name"] = "train.json"
    val["file_name"] = "val.json"
    return {f"{dataset_name}_train": train, f"{dataset_name}_val": val}


def is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def conclusion_action(text: str) -> str:
    payload = parse_json_object(text)
    if not payload:
        return ""
    action = payload.get("next_action")
    return action if isinstance(action, str) else ""


def is_valid_conclusion_output(text: str) -> bool:
    payload = parse_json_object(text)
    if not payload:
        return False
    if set(payload) != REQUIRED_CONCLUSION_KEYS:
        return False
    if payload.get("next_action") not in VALID_ACTIONS:
        return False
    if not isinstance(payload.get("missing_slots"), list):
        return False
    return True


def is_abstain_like(text: str) -> bool:
    action = conclusion_action(text)
    return action == "abstain" or any(marker in text for marker in ABSTAIN_MARKERS)


def row_output(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    if not conversations:
        return ""
    return str((conversations[-1] or {}).get("value") or "")


def prompt_signature(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    payload = {
        "system": row.get("system") or "",
        "tools": row.get("tools") or "",
        "task_type": row.get("task_type") or "",
        "prompt": conversations[:-1],
    }
    return stable_hash(payload)


def sft_signature(row: dict[str, Any]) -> str:
    payload = {
        "system": row.get("system") or "",
        "tools": row.get("tools") or "",
        "conversations": row.get("conversations") or [],
    }
    return stable_hash(payload)


def dpo_signature(row: dict[str, Any]) -> str:
    payload = {
        "system": row.get("system") or "",
        "tools": row.get("tools") or "",
        "conversations": row.get("conversations") or [],
        "chosen": row.get("chosen") or {},
        "rejected": row.get("rejected") or {},
        "duplicate_index": (row.get("meta") or {}).get("duplicate_index", 0),
    }
    return stable_hash(payload)


def add_source_meta(row: dict[str, Any], source_name: str, source_split: str) -> dict[str, Any]:
    item = copy.deepcopy(row)
    meta = item.setdefault("meta", {})
    meta["schema_dpo_source_dataset"] = source_name
    meta["schema_dpo_source_split"] = source_split
    return item


def build_sft_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen: set[str] = set()
    summary: dict[str, Any] = {"sources": {}, "skipped": Counter()}

    base_sft_dir = Path(args.base_sft_dir)
    for split in ("train", "val"):
        path = base_sft_dir / f"{split}.json"
        if not path.exists():
            continue
        for row in load_json(path):
            sig = sft_signature(row)
            if sig in seen:
                summary["skipped"]["duplicate_base_sft"] += 1
                continue
            seen.add(sig)
            rows_by_split[split].append(add_source_meta(row, base_sft_dir.name, split))
            summary["sources"][(split, base_sft_dir.name)] = summary["sources"].get((split, base_sft_dir.name), 0) + 1

    # Add verifier-approved positive samples to reinforce the exact runtime prompts.
    kto_dir = Path(args.kto_mix_dir)
    for split in ("train", "val"):
        path = kto_dir / f"{split}.json"
        if not path.exists():
            continue
        for row in load_json(path):
            if row.get("kto_tag") is not True:
                continue
            output = row_output(row)
            if not is_json_object(output):
                summary["skipped"]["positive_not_json"] += 1
                continue
            sig = sft_signature(row)
            if sig in seen:
                summary["skipped"]["duplicate_kto_positive"] += 1
                continue
            seen.add(sig)
            rows_by_split[split].append(add_source_meta(row, kto_dir.name, split))
            summary["sources"][(split, kto_dir.name)] = summary["sources"].get((split, kto_dir.name), 0) + 1

    dataset_name = output_dir.name
    write_json(output_dir / "train.json", rows_by_split["train"])
    write_json(output_dir / "val.json", rows_by_split["val"])
    write_json(output_dir / "dataset_info.json", dataset_info_sft(dataset_name))
    materialized_summary = {
        "records": {split: len(rows) for split, rows in rows_by_split.items()},
        "task_type": {split: dict(Counter(row.get("task_type") for row in rows)) for split, rows in rows_by_split.items()},
        "sources": {str(key): value for key, value in summary["sources"].items()},
        "skipped": dict(summary["skipped"]),
        "dedupe_key": "sha1(system, tools, conversations)",
    }
    write_json(output_dir / "summary.json", materialized_summary)
    return materialized_summary


def make_dpo_pair(chosen: dict[str, Any], rejected: dict[str, Any], duplicate_index: int = 0) -> dict[str, Any]:
    prompt = chosen.get("conversations")[:-1]
    chosen_msg = copy.deepcopy(chosen["conversations"][-1])
    rejected_msg = copy.deepcopy(rejected["conversations"][-1])
    meta = {
        "chosen_id": chosen.get("id"),
        "rejected_id": rejected.get("id"),
        "chosen_meta": chosen.get("meta") or {},
        "rejected_meta": rejected.get("meta") or {},
        "chosen_action": conclusion_action(row_output(chosen)),
        "rejected_action": conclusion_action(row_output(rejected)),
        "duplicate_index": duplicate_index,
    }
    return {
        "id": f"{chosen.get('id', 'chosen')}__vs__{rejected.get('id', 'rejected')}__dup{duplicate_index}",
        "task_type": "conclusion_generation",
        "bucket": "schema_conclusion_dpo",
        "system": chosen.get("system") or "",
        "tools": chosen.get("tools") or "[]",
        "conversations": copy.deepcopy(prompt),
        "chosen": chosen_msg,
        "rejected": rejected_msg,
        "meta": meta,
    }


def pair_category(chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    chosen_action = conclusion_action(row_output(chosen))
    rejected_action = conclusion_action(row_output(rejected))
    if chosen_action == "answer_directly" and (rejected_action in {"retrieve_more", "abstain"} or is_abstain_like(row_output(rejected))):
        return "anti_over_abstain_or_over_retrieve"
    if chosen_action == "retrieve_more" and rejected_action == "answer_directly":
        return "anti_premature_answer"
    if chosen_action == "answer_directly" and rejected_action == "answer_directly":
        return "answer_quality"
    return f"{chosen_action}_vs_{rejected_action}"


def build_dpo_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.dpo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    summary: dict[str, Any] = {"skipped": Counter(), "pair_categories": Counter()}

    kto_dir = Path(args.kto_mix_dir)
    for split in ("train", "val"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in load_json(kto_dir / f"{split}.json"):
            if row.get("task_type") != "conclusion_generation":
                summary["skipped"][f"{split}:not_conclusion"] += 1
                continue
            groups[prompt_signature(row)].append(row)

        seen: set[str] = set()
        for group in groups.values():
            positives = [row for row in group if row.get("kto_tag") is True and is_valid_conclusion_output(row_output(row))]
            negatives = [row for row in group if row.get("kto_tag") is False]
            if not positives or not negatives:
                summary["skipped"][f"{split}:not_pairable"] += len(group)
                continue

            for chosen in positives:
                for rejected in negatives:
                    category = pair_category(chosen, rejected)
                    dup_count = 1
                    if split == "train" and category == "anti_over_abstain_or_over_retrieve":
                        dup_count = args.anti_over_abstain_dup

                    for duplicate_index in range(dup_count):
                        pair = make_dpo_pair(chosen, rejected, duplicate_index)
                        pair["meta"]["pair_category"] = category
                        sig = dpo_signature(pair)
                        if sig in seen:
                            summary["skipped"][f"{split}:duplicate_pair"] += 1
                            continue
                        seen.add(sig)
                        rows_by_split[split].append(pair)
                        summary["pair_categories"][(split, category)] += 1

    dataset_name = output_dir.name
    write_json(output_dir / "train.json", rows_by_split["train"])
    write_json(output_dir / "val.json", rows_by_split["val"])
    write_json(output_dir / "dataset_info.json", dataset_info_dpo(dataset_name))
    materialized_summary = {
        "records": {split: len(rows) for split, rows in rows_by_split.items()},
        "pair_categories": {str(key): value for key, value in summary["pair_categories"].items()},
        "skipped": dict(summary["skipped"]),
        "dedupe_key": "sha1(system, tools, prompt, chosen, rejected)",
    }
    write_json(output_dir / "summary.json", materialized_summary)
    return materialized_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build schema SFT patch and conclusion-only DPO datasets.")
    parser.add_argument(
        "--base-sft-dir",
        default="data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_plus_detail_conclusion_patch_v1_abstain_mini8_retrieval_trim_v1",
    )
    parser.add_argument(
        "--kto-mix-dir",
        default="data/processed/llama_factory/soda_teacher_scored_kto_mix_v1",
    )
    parser.add_argument(
        "--sft-output-dir",
        default="data/processed/llama_factory/schema_sft_patch_v1",
    )
    parser.add_argument(
        "--dpo-output-dir",
        default="data/processed/llama_factory/soda_conclusion_dpo_v1",
    )
    parser.add_argument("--anti-over-abstain-dup", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_summary = build_sft_dataset(args)
    dpo_summary = build_dpo_dataset(args)
    print(json.dumps({"sft": sft_summary, "dpo": dpo_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
