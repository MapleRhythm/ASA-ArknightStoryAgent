#!/usr/bin/env python3
"""Quarantine disputed historical examples; never invent a new approval."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soda_dataset_integrity import question_split_key, split_records  # noqa: E402

ROLE_TAGS = {
    "role_tag": "from", "content_tag": "value", "user_tag": "human",
    "assistant_tag": "gpt", "observation_tag": "observation", "function_tag": "function_call",
}
DEFAULT_INPUT_DIR = Path(
    "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_clean_v1"
)

# These are review pointers from the historical cleaner, NOT adjudicated gold.
# The old hand-written replacements and verifier overwrites are intentionally
# removed. Keep the original records separately until fresh source review.
LEGACY_REVIEW_REQUIRED = {
    "9fa94eaa04a4a919": "historical father-coat claim dispute",
    "e115e5cf6a2fa71b": "historical political-causal phrasing dispute",
    "f2e058438aab8018": "historical infection-status claim dispute",
    "391172ae6f3fff4d": "historical leave-danger-area completion dispute",
}


def review_reason(record: dict[str, Any]) -> str | None:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    verifier = meta.get("api_verifier") if isinstance(meta.get("api_verifier"), dict) else {}
    if meta.get("clean_v1_patch_reason") or str(verifier.get("label_reason") or "").startswith("clean_v1:"):
        return "historical target/verifier rewrite requires fresh review"
    key = str(meta.get("prompt_key") or "")
    if (
        key in LEGACY_REVIEW_REQUIRED
        and record.get("task_type") == "conclusion_generation"
        and record.get("kto_tag") is True
        and meta.get("api_verifier_reason") == "verifier_chosen_answer_directly"
    ):
        return LEGACY_REVIEW_REQUIRED[key]
    return None


def partition_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    """Keep entire related question groups out of the candidate split.

    No answer, label, evidence, or verifier metadata is edited. This is not a
    semantic validator and makes no claim that the retained labels are correct.
    """
    records_by_id: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"].strip():
            raise ValueError("Training record must have a nonempty string id")
        rid = record["id"]
        if rid in records_by_id:
            if records_by_id[rid] != record:
                raise ValueError(f"Conflicting duplicate record id: {rid}")
            stats["identical_duplicates"] += 1
        else:
            records_by_id[rid] = record

    review_groups: dict[str, set[str]] = {}
    for record in records_by_id.values():
        reason = review_reason(record)
        if reason:
            review_groups.setdefault(question_split_key(record), set()).add(reason)

    kept, quarantine = [], []
    for record in records_by_id.values():
        group = question_split_key(record)
        if group in review_groups:
            quarantine.append({
                "record": deepcopy(record), "review_required": True,
                "question_group": group, "reasons": sorted(review_groups[group]),
            })
        else:
            kept.append(deepcopy(record))
    stats.update({
        "input_records": len(records), "unique_records": len(records_by_id),
        "retained_records": len(kept), "quarantined_records": len(quarantine),
        "quarantined_question_groups": len(review_groups), "answers_rewritten": 0,
        "verifier_judgments_modified": 0,
    })
    return kept, quarantine, stats


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept, _, stats = partition_records(records)
    return kept, stats


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools", "kto_tag": "kto_tag"},
            "tags": ROLE_TAGS,
        }
    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarantine disputed SODA records without rewriting targets or approvals.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    source, target = input_dir.resolve(), output_dir.resolve()
    if target == source or source in target.parents:
        raise ValueError("Output must be separate from the input directory")
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output_dir}")

    # Validate all required input and retained split before creating any output.
    inputs: dict[str, list[dict[str, Any]]] = {}
    input_hashes = {}
    for name in ("train.json", "val.json"):
        raw = (input_dir / name).read_bytes()
        inputs[name] = json.loads(raw)
        if not isinstance(inputs[name], list):
            raise ValueError(f"Input {name} must be an array")
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
    records, quarantine, stats = partition_records(inputs["train.json"] + inputs["val.json"])
    train, val = split_records(records, seed=args.seed, val_ratio=args.val_ratio)
    companions = [
        input_dir / name for name in ("api_verifier_records.jsonl", "teacher_full_chain.jsonl")
        if (input_dir / name).is_file()
    ]
    for path in companions:
        input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_name = args.dataset_name or output_dir.name
    summary = {
        "policy": "quarantine_without_target_or_judgment_rewrite_v1",
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "input_file_sha256": input_hashes, "records_total": len(records),
        "records_input": stats["input_records"], "records_unique": stats["unique_records"],
        "records_train": len(train), "records_val": len(val),
        "records_quarantined": len(quarantine),
        "answers_rewritten": 0, "verifier_judgments_modified": 0,
        "split_unit": "normalized_question_or_exact_input",
        "train_val_question_overlap": len(
            {question_split_key(r) for r in train} & {question_split_key(r) for r in val}
        ),
        "training_ready": False,
        "review_required": "Retained labels/contracts are not re-approved. Quarantined originals require fresh source review.",
        "legacy_review_prompt_keys": sorted(LEGACY_REVIEW_REQUIRED), "stats": dict(stats),
        "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
        "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(dataset_name))
    with (output_dir / "quarantine.jsonl").open("x", encoding="utf-8") as stream:
        for item in quarantine:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    for path in companions:
        # Preserve the complete original provenance, including quarantined rows.
        shutil.copyfile(path, output_dir / path.name)
        if hashlib.sha256((output_dir / path.name).read_bytes()).hexdigest() != input_hashes[path.name]:
            raise ValueError(f"Input changed during provenance copy: {path.name}")
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
