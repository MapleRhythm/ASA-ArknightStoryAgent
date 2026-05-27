#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_fix1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_answer_bias_fix1"
)

CONCLUSION_TASK = "conclusion_generation"
RETRIEVAL_ACTIONS = {"answer_directly", "retrieve_more", "clarify_user", "abstain"}
INTERNAL_EVIDENCE_META_RE = re.compile(
    r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]+\]\s*|\[E\d+\]\s*"
)
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff·]{2,16}|[A-Za-z][A-Za-z0-9_.-]{1,31}")
NOISY_TERMS = {
    "为什么",
    "为何",
    "怎么",
    "如何",
    "什么",
    "哪里",
    "哪儿",
    "是否",
    "有没有",
    "用户",
    "问题",
    "剧情",
    "证据",
    "当前",
    "回答",
    "关系",
    "身份",
    "原因",
    "具体",
    "哪些",
    "多少",
}
NO_EVIDENCE_MARKERS = (
    "未包含任何",
    "没有包含任何",
    "没有任何关于",
    "没有检索到",
    "未检索到",
    "无法回答",
)
PARTIAL_ANSWER_MARKERS = (
    "已检索到的证据能确认",
    "现有证据仅显示",
    "现有证据可以确认",
)
WEAK_PARTIAL_ANSWER_MARKERS = (
    "当前证据仅显示",
    "当前证据仅包含",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def user_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("value") or message.get("content") or "")
        for message in record.get("conversations") or []
        if message.get("from") in {"human", "user"}
    )


def assistant_message(record: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(record.get("conversations") or []):
        if message.get("from") in {"gpt", "assistant"}:
            return message
    return None


def assistant_text(record: dict[str, Any]) -> str:
    message = assistant_message(record)
    return str(message.get("value") or message.get("content") or "") if message else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def dump_compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def task_type(record: dict[str, Any]) -> str:
    value = str(record.get("task_type") or "").strip()
    if value:
        return value
    match = re.search(r"^task:\s*([^\n]+)", user_text(record), flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_round(prompt: str) -> tuple[int | None, int | None]:
    match = re.search(r"round:\s*(\d+)\s*/\s*(\d+)", prompt)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def extract_evidence(prompt: str) -> str:
    for label in ("当前证据:", "evidence_brief:"):
        if label not in prompt:
            continue
        tail = prompt.split(label, 1)[1]
        stops = [
            index
            for marker in (
                "\n输出要求:",
                "\noutput_schema:",
                "\nprevious_action:",
                "\nmissing_slots:",
                "\nfields:",
            )
            if (index := tail.find(marker)) >= 0
        ]
        if stops:
            tail = tail[: min(stops)]
        return tail.strip()
    return ""


def clean_internal_meta(text: str) -> str:
    cleaned = INTERNAL_EVIDENCE_META_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for term in CJK_TOKEN_RE.findall(question or ""):
        value = term.strip()
        if (
            not value
            or value in NOISY_TERMS
            or any(noisy in value for noisy in ("为什么", "为何", "怎么", "如何", "什么"))
        ):
            continue
        terms.append(value)
    return list(dict.fromkeys(terms))


def anchor_hit_count(question: str, evidence: str) -> int:
    compact_evidence = re.sub(r"\s+", "", evidence or "")
    return sum(1 for term in question_terms(question)[:12] if re.sub(r"\s+", "", term) in compact_evidence)


def is_no_evidence_answer(answer: str) -> bool:
    if any(marker in answer for marker in PARTIAL_ANSWER_MARKERS):
        return False
    return any(marker in answer for marker in NO_EVIDENCE_MARKERS)


def is_partial_answer(answer: str) -> bool:
    if is_no_evidence_answer(answer):
        return False
    if any(marker in answer for marker in WEAK_PARTIAL_ANSWER_MARKERS):
        return False
    return any(marker in answer for marker in PARTIAL_ANSWER_MARKERS)


def normalize_partial_answer(answer: str) -> str:
    answer = clean_internal_meta(answer)
    replacements = {
        "现有证据仅显示": "现有证据可以确认",
        "无法完整回答": "因此只能给出部分回答，未确认部分需要保留不确定。",
        "无法给出完整解释": "因此只能给出部分回答，未确认部分需要保留不确定。",
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    return answer.strip() or "现有证据只能确认部分事实，未确认部分需要保留不确定。"


def repair_conclusion_payload(
    payload: dict[str, Any],
    *,
    prompt: str,
    convert_partial_abstain: bool,
    convert_strong_anchor_abstain: bool,
) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(payload)
    reasons: list[str] = []
    action = str(repaired.get("next_action") or "")
    answer = str(repaired.get("answer") or "")

    if answer:
        cleaned_answer = clean_internal_meta(answer)
        if cleaned_answer != answer:
            repaired["answer"] = cleaned_answer
            answer = cleaned_answer
            reasons.append("clean_internal_evidence_meta")

    if action != "abstain":
        return repaired, reasons

    question = str(repaired.get("question") or "")
    evidence = extract_evidence(prompt)
    strong_anchor = anchor_hit_count(question, evidence) >= 2 and len(evidence) >= 200
    should_convert = False
    if convert_partial_abstain and is_partial_answer(answer):
        should_convert = True
        reasons.append("convert_partial_abstain_to_answer")
    elif convert_strong_anchor_abstain and strong_anchor and not is_no_evidence_answer(answer):
        should_convert = True
        reasons.append("convert_strong_anchor_abstain_to_answer")

    if should_convert:
        repaired["next_action"] = "answer_directly"
        repaired["answer"] = normalize_partial_answer(answer)
        repaired["missing_slots"] = []
        repaired["clarification_question"] = ""
        repaired["follow_up_hypothesis"] = None
    return repaired, reasons


def action_of(record: dict[str, Any]) -> str:
    if task_type(record) != CONCLUSION_TASK:
        return ""
    payload = parse_json_object(assistant_text(record))
    return str(payload.get("next_action") or "") if payload else "invalid_json"


def record_question(record: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    if payload and payload.get("question"):
        return str(payload["question"])
    meta_question = str((record.get("meta") or {}).get("source_question") or "").strip()
    if meta_question:
        return meta_question
    prompt = user_text(record)
    match = re.search(r"(?:question|用户原问题|用户问题)[:：]\s*([^\n]+)", prompt)
    return match.group(1).strip() if match else ""


def split_records_by_action(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_action[action_of(record)].append(record)
    return by_action


def downsample_conclusions(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_retrieve_per_question: int,
    retrieve_ratio: float,
    abstain_ratio: float,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    non_conclusion = [record for record in records if task_type(record) != CONCLUSION_TASK]
    conclusions = [record for record in records if task_type(record) == CONCLUSION_TASK]
    by_action = split_records_by_action(conclusions)
    answer_records = by_action.get("answer_directly", [])
    clarify_records = by_action.get("clarify_user", [])

    retrieve_records = by_action.get("retrieve_more", [])
    retrieve_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in retrieve_records:
        payload = parse_json_object(assistant_text(record))
        retrieve_by_question[record_question(record, payload)].append(record)

    capped_retrieve: list[dict[str, Any]] = []
    dropped_samples: list[dict[str, Any]] = []
    for question, grouped in retrieve_by_question.items():
        grouped = sorted(grouped, key=lambda item: item.get("id", ""))
        keep = grouped[:max_retrieve_per_question]
        drop = grouped[max_retrieve_per_question:]
        capped_retrieve.extend(keep)
        for record in drop:
            if len(dropped_samples) < 300:
                dropped_samples.append({"id": record.get("id"), "reason": "drop_extra_retrieve_more", "question": question})

    target_retrieve = max(1, int(round(len(answer_records) * retrieve_ratio))) if answer_records else len(capped_retrieve)
    if len(capped_retrieve) > target_retrieve:
        keep_ids = {id(record) for record in rng.sample(capped_retrieve, target_retrieve)}
        dropped = [record for record in capped_retrieve if id(record) not in keep_ids]
        capped_retrieve = [record for record in capped_retrieve if id(record) in keep_ids]
        for record in dropped[: max(0, 300 - len(dropped_samples))]:
            payload = parse_json_object(assistant_text(record))
            dropped_samples.append(
                {"id": record.get("id"), "reason": "downsample_retrieve_more", "question": record_question(record, payload)}
            )

    abstain_records = by_action.get("abstain", [])
    target_abstain = max(1, int(round(len(answer_records) * abstain_ratio))) if answer_records else len(abstain_records)
    if len(abstain_records) > target_abstain:
        keep_ids = {id(record) for record in rng.sample(abstain_records, target_abstain)}
        dropped = [record for record in abstain_records if id(record) not in keep_ids]
        abstain_records = [record for record in abstain_records if id(record) in keep_ids]
        for record in dropped[: max(0, 300 - len(dropped_samples))]:
            payload = parse_json_object(assistant_text(record))
            dropped_samples.append(
                {"id": record.get("id"), "reason": "downsample_abstain", "question": record_question(record, payload)}
            )

    keep_ids = {id(record) for record in non_conclusion + answer_records + capped_retrieve + abstain_records + clarify_records}
    output = [record for record in records if id(record) in keep_ids]
    stats = Counter(action_of(record) for record in output if task_type(record) == CONCLUSION_TASK)
    return output, stats, dropped_samples


def repair_split(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    convert_partial_abstain: bool,
    convert_strong_anchor_abstain: bool,
    max_retrieve_per_question: int,
    retrieve_ratio: float,
    abstain_ratio: float,
    disable_downsample: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    before_actions = Counter(action_of(record) for record in records if task_type(record) == CONCLUSION_TASK)
    repair_reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    repaired_records: list[dict[str, Any]] = []

    for record in records:
        new_record = copy.deepcopy(record)
        if task_type(new_record) == CONCLUSION_TASK:
            message = assistant_message(new_record)
            payload = parse_json_object(str(message.get("value") or "")) if message else None
            if payload is not None:
                prompt = user_text(new_record)
                repaired_payload, reasons = repair_conclusion_payload(
                    payload,
                    prompt=prompt,
                    convert_partial_abstain=convert_partial_abstain,
                    convert_strong_anchor_abstain=convert_strong_anchor_abstain,
                )
                for reason in reasons:
                    repair_reasons[reason] += 1
                if reasons and len(samples) < 300:
                    samples.append(
                        {
                            "id": new_record.get("id"),
                            "question": record_question(new_record, payload),
                            "before": payload,
                            "after": repaired_payload,
                            "reasons": reasons,
                            "evidence_preview": extract_evidence(prompt)[:800],
                        }
                    )
                if message is not None:
                    message["value"] = dump_compact_json(repaired_payload)
        repaired_records.append(new_record)

    if disable_downsample:
        after_records = repaired_records
        after_actions = Counter(action_of(record) for record in after_records if task_type(record) == CONCLUSION_TASK)
        dropped_samples: list[dict[str, Any]] = []
    else:
        after_records, after_actions, dropped_samples = downsample_conclusions(
            repaired_records,
            rng=rng,
            max_retrieve_per_question=max_retrieve_per_question,
            retrieve_ratio=retrieve_ratio,
            abstain_ratio=abstain_ratio,
        )
    samples.extend(dropped_samples)

    report = {
        "records_before": len(records),
        "records_after": len(after_records),
        "task_counts_before": dict(Counter(task_type(record) for record in records)),
        "task_counts_after": dict(Counter(task_type(record) for record in after_records)),
        "conclusion_actions_before": dict(before_actions),
        "conclusion_actions_after": dict(after_actions),
        "repair_reasons": dict(repair_reasons),
    }
    return after_records, report, samples


def make_dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
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

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
        f"{dataset_name}_test": entry("test.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair short-prompt SFT data with over-strong abstain bias.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-retrieve-per-question", type=int, default=1)
    parser.add_argument("--retrieve-ratio", type=float, default=0.75)
    parser.add_argument("--abstain-ratio", type=float, default=0.45)
    parser.add_argument("--disable-downsample", action="store_true")
    parser.add_argument("--no-convert-partial-abstain", dest="convert_partial_abstain", action="store_false")
    parser.add_argument("--convert-strong-anchor-abstain", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    rng = random.Random(args.seed)

    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = output_dir.name

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_name": dataset_name,
        "seed": args.seed,
        "settings": {
            "convert_partial_abstain": args.convert_partial_abstain,
            "convert_strong_anchor_abstain": args.convert_strong_anchor_abstain,
            "disable_downsample": args.disable_downsample,
            "max_retrieve_per_question": args.max_retrieve_per_question,
            "retrieve_ratio": args.retrieve_ratio,
            "abstain_ratio": args.abstain_ratio,
        },
        "splits": {},
    }
    all_records: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.json"
        if not input_path.exists():
            continue
        records = load_json(input_path)
        repaired_records, split_report, samples = repair_split(
            records,
            rng=rng,
            convert_partial_abstain=args.convert_partial_abstain,
            convert_strong_anchor_abstain=args.convert_strong_anchor_abstain,
            max_retrieve_per_question=args.max_retrieve_per_question,
            retrieve_ratio=args.retrieve_ratio,
            abstain_ratio=args.abstain_ratio,
            disable_downsample=args.disable_downsample,
        )
        write_json(output_dir / f"{split}.json", repaired_records)
        report["splits"][split] = split_report
        all_records.extend(repaired_records)
        for sample in samples:
            sample["split"] = split
        all_samples.extend(samples)

    write_json(output_dir / "dataset_info.json", make_dataset_info(dataset_name))
    write_jsonl(output_dir / "records.jsonl", all_records)
    write_jsonl(output_dir / "repair_samples.jsonl", all_samples)
    build_summary = {
        "records": len(all_records),
        "task_counts": dict(Counter(task_type(record) for record in all_records)),
        "conclusion_actions": dict(Counter(action_of(record) for record in all_records if task_type(record) == CONCLUSION_TASK)),
    }
    write_json(output_dir / "build_summary.json", build_summary)
    write_json(output_dir / "repair_report.json", report)
    print(json.dumps({"build_summary": build_summary, "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
