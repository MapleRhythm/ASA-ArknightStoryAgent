#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


DEFAULT_INPUT_DIR = Path(
    "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_clean_v1"
)


ANSWER_PATCHES: dict[str, dict[str, str]] = {
    "9fa94eaa04a4a919": {
        "answer": "彼得海姆中学内整合运动撤离后，学生之间爆发无差别混乱斗争，为争夺资源互相残杀并引发火灾，导致大量伤亡。卓娅到达时看到满地学生服、烧焦痕迹，以及并非整合运动直接造成的伤亡景象，并由现场景象联想到父亲而震惊。",
        "reason": "remove unsupported father-coat claim",
    },
    "e115e5cf6a2fa71b": {
        "answer": "暴力事件频发的原因包括：感染者社区被部分贵族视为眼中钉，围绕新政策的利益冲突持续存在；有人利用“感染者社区隐藏匪帮”等流言制造敌意；军用燃烧弹流入黑市；还有暴徒伪装成感染者制造事件；警备队也没有认真调查相关线索。",
        "reason": "narrow broad unsupported political-causal phrasing",
    },
    "f2e058438aab8018": {
        "answer": "天火出手是因为她目睹暴徒绑架感染者工人和市民，并看出有人用涂黑玻璃冒充源石结晶、伪装成感染者。她随后使用源石技艺制服暴徒，救下被绑架的人。",
        "reason": "change unconfirmed non-infected claim into evidence-supported disguise claim",
    },
    "391172ae6f3fff4d": {
        "answer": "夏栎先用源石技艺催生藤蔓突破暴徒防线，随后在苏茜即将杀死贝希曼时及时赶到，紧紧抱住苏茜并安抚她，阻止她继续攻击贝希曼。",
        "reason": "remove unsupported leave-danger-area completion",
    },
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def prompt_key(record: dict[str, Any], fallback: int) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    return str(meta.get("prompt_key") or record.get("id") or fallback)


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        by_prompt.setdefault(prompt_key(record, index), []).append(record)
    keys = list(by_prompt)
    random.Random(seed).shuffle(keys)
    target_val = max(1, int(round(len(records) * val_ratio))) if len(records) > 10 and val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for key in keys:
        bucket = by_prompt[key]
        if val_count < target_val:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(val)
    return train, val


def patch_record(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    output = json.loads(json.dumps(record, ensure_ascii=False))
    meta = output.get("meta") if isinstance(output.get("meta"), dict) else {}
    key = str(meta.get("prompt_key") or "")
    patch = ANSWER_PATCHES.get(key)
    if not patch:
        return output, None
    if output.get("task_type") != "conclusion_generation":
        return output, None
    if output.get("kto_tag") is not True:
        return output, None
    if meta.get("api_verifier_reason") != "verifier_chosen_answer_directly":
        return output, None

    conversations = output.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return output, None
    try:
        payload = json.loads(str(conversations[-1].get("value") or ""))
    except json.JSONDecodeError:
        return output, None
    if not isinstance(payload, dict) or payload.get("next_action") != "answer_directly":
        return output, None

    payload["answer"] = patch["answer"]
    payload["missing_slots"] = []
    payload["clarification_question"] = ""
    payload["follow_up_hypothesis"] = None
    conversations[-1]["value"] = compact_json(payload)

    verifier = meta.get("api_verifier") if isinstance(meta.get("api_verifier"), dict) else {}
    verifier["supported_answer"] = patch["answer"]
    verifier["label_reason"] = f"clean_v1: {patch['reason']}"
    verifier["teacher_answer_uses_prior_knowledge"] = False
    if verifier.get("teacher_action_error") == "unsupported_answer":
        verifier["teacher_action_error"] = "none"
    meta["api_verifier"] = verifier
    meta["label_reason"] = verifier["label_reason"]
    meta["clean_v1_patch_reason"] = patch["reason"]
    output["meta"] = meta
    return output, patch["reason"]


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    cleaned: list[dict[str, Any]] = []
    for record in records:
        output, reason = patch_record(record)
        if reason:
            stats[f"patched:{reason}"] += 1
            stats["patched_total"] += 1
        cleaned.append(output)
    return cleaned, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch confirmed weak accepts in SODA API-verifier KTO dataset.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    train_in = read_json(input_dir / "train.json", [])
    val_in = read_json(input_dir / "val.json", [])
    if not isinstance(train_in, list) or not isinstance(val_in, list):
        raise SystemExit(f"Invalid dataset: {input_dir}")
    records, stats = clean_records(train_in + val_in)
    train, val = split_records(records, seed=args.seed, val_ratio=max(0.0, min(0.5, args.val_ratio)))
    dataset_name = args.dataset_name or output_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(dataset_name))
    write_jsonl(output_dir / "api_verifier_records.jsonl", read_jsonl(input_dir / "api_verifier_records.jsonl"))
    teacher_full_chain = read_jsonl(input_dir / "teacher_full_chain.jsonl")
    if teacher_full_chain:
        write_jsonl(output_dir / "teacher_full_chain.jsonl", teacher_full_chain)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "records_total": len(records),
        "records_train": len(train),
        "records_val": len(val),
        "patch_prompt_keys": sorted(ANSWER_PATCHES),
        "stats": dict(stats),
        "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
        "task_counts": dict(Counter(str(record.get("task_type") or "") for record in records)),
        "api_verifier_reasons": dict(
            Counter(str((record.get("meta") or {}).get("api_verifier_reason") or "") for record in records)
        ),
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
