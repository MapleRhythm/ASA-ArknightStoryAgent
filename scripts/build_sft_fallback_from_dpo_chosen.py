#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
    base = {
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
    train = copy.deepcopy(base)
    val = copy.deepcopy(base)
    train["file_name"] = "train.json"
    val["file_name"] = "val.json"
    return {f"{dataset_name}_train": train, f"{dataset_name}_val": val}


def sft_signature(row: dict[str, Any]) -> str:
    return stable_hash(
        {
            "system": row.get("system") or "",
            "tools": row.get("tools") or "",
            "conversations": row.get("conversations") or [],
        }
    )


def convert_dpo_row(row: dict[str, Any]) -> dict[str, Any]:
    conversations = copy.deepcopy(row.get("conversations") or [])
    chosen = copy.deepcopy(row.get("chosen") or {})
    if not conversations or chosen.get("from") != "gpt":
        raise ValueError(f"bad DPO row: {row.get('id')}")
    conversations.append(chosen)
    item = {
        "id": f"{row.get('id', stable_hash(row))}__chosen_sft",
        "task_type": row.get("task_type") or "conclusion_generation",
        "bucket": "dpo_chosen_sft",
        "system": row.get("system") or "",
        "tools": row.get("tools") or "[]",
        "conversations": conversations,
        "meta": copy.deepcopy(row.get("meta") or {}),
    }
    item["meta"]["sft_fallback_source"] = "dpo_chosen"
    item["meta"]["source_dpo_id"] = row.get("id")
    return item


def build_chosen_sft(dpo_dir: Path, out_dir: Path, dedupe: bool) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen: set[str] = set()
    skipped = Counter()
    actions: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    pair_categories: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    for split in ("train", "val"):
        for row in load_json(dpo_dir / f"{split}.json"):
            try:
                item = convert_dpo_row(row)
            except Exception:
                skipped[f"{split}:bad_row"] += 1
                continue
            sig = sft_signature(item)
            if dedupe and sig in seen:
                skipped[f"{split}:duplicate"] += 1
                continue
            seen.add(sig)
            rows_by_split[split].append(item)
            meta = item.get("meta") or {}
            actions[split][str(meta.get("chosen_action") or "")] += 1
            pair_categories[split][str(meta.get("pair_category") or "")] += 1

    dataset_name = out_dir.name
    write_json(out_dir / "train.json", rows_by_split["train"])
    write_json(out_dir / "val.json", rows_by_split["val"])
    write_json(out_dir / "dataset_info.json", dataset_info_sft(dataset_name))
    summary = {
        "records": {split: len(rows) for split, rows in rows_by_split.items()},
        "chosen_actions": {split: dict(counter) for split, counter in actions.items()},
        "pair_categories": {split: dict(counter) for split, counter in pair_categories.items()},
        "skipped": dict(skipped),
        "dedupe": dedupe,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def build_merged_sft(schema_dir: Path, chosen_dir: Path, out_dir: Path, dedupe: bool) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen: set[str] = set()
    skipped = Counter()
    sources: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    for source_name, source_dir in (("schema_sft_patch", schema_dir), ("dpo_chosen_sft", chosen_dir)):
        for split in ("train", "val"):
            for row in load_json(source_dir / f"{split}.json"):
                item = copy.deepcopy(row)
                meta = item.setdefault("meta", {})
                meta["merged_sft_source"] = source_name
                sig = sft_signature(item)
                if dedupe and sig in seen:
                    skipped[f"{split}:duplicate:{source_name}"] += 1
                    continue
                seen.add(sig)
                rows_by_split[split].append(item)
                sources[split][source_name] += 1

    dataset_name = out_dir.name
    write_json(out_dir / "train.json", rows_by_split["train"])
    write_json(out_dir / "val.json", rows_by_split["val"])
    write_json(out_dir / "dataset_info.json", dataset_info_sft(dataset_name))
    summary = {
        "records": {split: len(rows) for split, rows in rows_by_split.items()},
        "sources": {split: dict(counter) for split, counter in sources.items()},
        "task_type": {split: dict(Counter(row.get("task_type") for row in rows)) for split, rows in rows_by_split.items()},
        "skipped": dict(skipped),
        "dedupe": dedupe,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-dir", default="data/processed/llama_factory/soda_conclusion_dpo_v1")
    parser.add_argument("--schema-dir", default="data/processed/llama_factory/schema_sft_patch_v1")
    parser.add_argument("--chosen-out", default="data/processed/llama_factory/conclusion_chosen_sft_v1")
    parser.add_argument(
        "--merged-out", default="data/processed/llama_factory/schema_sft_plus_conclusion_chosen_v1"
    )
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    dpo_dir = Path(args.dpo_dir)
    schema_dir = Path(args.schema_dir)
    chosen_out = Path(args.chosen_out)
    merged_out = Path(args.merged_out)

    chosen_summary = build_chosen_sft(dpo_dir, chosen_out, args.dedupe)
    merged_summary = build_merged_sft(schema_dir, chosen_out, merged_out, args.dedupe)
    print(json.dumps({"chosen_sft": chosen_summary, "merged_sft": merged_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
