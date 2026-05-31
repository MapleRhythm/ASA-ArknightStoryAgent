#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_verifier_lite_v1"

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
EVIDENCE_START_RE = re.compile(r"(?m)^evidence_brief:\s*$")
EVIDENCE_END_RE = re.compile(
    r"(?m)^(?:minirag_hints:|output_schema:|fields:|next_action_set:|field_rules:|follow_up_hypothesis_fields:|forbidden_fields:)"
)
COMMON_TOKENS = {
    "这个",
    "那个",
    "一种",
    "因为",
    "为了",
    "所以",
    "因此",
    "但是",
    "以及",
    "最终",
    "现有",
    "证据",
    "显示",
    "确认",
    "无法",
    "部分",
    "问题",
    "回答",
    "相关",
    "具体",
    "原因",
    "目的",
    "结果",
    "过程",
    "情况",
    "剧情",
    "明日方舟",
    "根据",
    "提供",
    "说明",
    "认为",
    "可能",
    "不是",
    "没有",
    "需要",
    "通过",
    "进行",
    "成为",
    "作为",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def response_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[-1].get("value") or "").strip()
    return ""


def prompt_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[0].get("value") or "")
    return ""


def parse_jsonish(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_record(record: dict[str, Any], *, kto_tag: bool, reason: str, support: dict[str, Any] | None = None) -> dict[str, Any]:
    output = json.loads(json.dumps(record, ensure_ascii=False))
    output["kto_tag"] = bool(kto_tag)
    meta = output.setdefault("meta", {})
    meta["verifier_lite_reason"] = reason
    if support is not None:
        meta["verifier_lite_support"] = support
    return output


def extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_evidence(prompt: str) -> str:
    source = prompt or ""
    match = EVIDENCE_START_RE.search(source)
    if not match:
        return ""
    evidence = source[match.end() :]
    end_match = EVIDENCE_END_RE.search(evidence)
    if end_match:
        evidence = evidence[: end_match.start()]
    return evidence.strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text or ""):
        normalized = re.sub(r"\s+", "", token).strip("，。！？；：、（）()[]【】《》“”\"'")
        if (
            not normalized
            or normalized in COMMON_TOKENS
            or len(normalized) < 2
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tokens


def answer_support(answer: str, evidence: str, question: str) -> dict[str, Any]:
    compact_evidence = re.sub(r"\s+", "", evidence or "")
    compact_question = re.sub(r"\s+", "", question or "")
    tokens = tokenize(answer)
    if not answer.strip():
        return {"status": "none", "hit_rate": 0.0, "hit_count": 0, "token_count": 0, "missing": []}
    if not compact_evidence or not tokens:
        return {"status": "uncertain", "hit_rate": 0.0, "hit_count": 0, "token_count": len(tokens), "missing": tokens[:12]}

    hits = [token for token in tokens if token in compact_evidence]
    missing = [token for token in tokens if token not in compact_evidence and token not in compact_question]
    hit_rate = len(hits) / len(tokens) if tokens else 0.0
    if len(hits) >= 3 and hit_rate >= 0.35:
        status = "supported"
    elif len(tokens) >= 8 and len(hits) <= 1 and hit_rate < 0.18:
        status = "unsupported"
    elif len(tokens) >= 12 and len(missing) >= 8 and hit_rate < 0.25:
        status = "unsupported"
    else:
        status = "uncertain"
    return {
        "status": status,
        "hit_rate": round(hit_rate, 4),
        "hit_count": len(hits),
        "token_count": len(tokens),
        "hits": hits[:12],
        "missing": missing[:12],
    }


def action(payload: dict[str, Any] | None) -> str:
    return str((payload or {}).get("next_action") or "").strip()


def answer(payload: dict[str, Any] | None) -> str:
    return str((payload or {}).get("answer") or "").strip()


def is_conclusion(record: dict[str, Any]) -> bool:
    return str(record.get("task_type") or "") == "conclusion_generation"


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        by_prompt[key].append(record)
    keys = list(by_prompt)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = int(round(len(records) * val_ratio)) if val_ratio > 0 else 0
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
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


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


def process_group(
    records: list[dict[str, Any]],
    *,
    stats: Counter,
    drop_uncertain_teacher_answers: bool,
) -> list[dict[str, Any]]:
    teacher = next((record for record in records if record.get("kto_tag") is True), None)
    students = [record for record in records if record.get("kto_tag") is False]
    if teacher is None:
        stats["drop:no_teacher_positive"] += len(records)
        return []
    if not is_conclusion(teacher):
        stats["keep:non_conclusion"] += len(records)
        return records

    prompt = prompt_text(teacher)
    question = extract_question(prompt)
    evidence = extract_evidence(prompt)
    teacher_payload = parse_jsonish(response_text(teacher))
    teacher_action = action(teacher_payload)
    teacher_answer = answer(teacher_payload)
    teacher_support = answer_support(teacher_answer, evidence, question) if teacher_action == "answer_directly" else None
    output: list[dict[str, Any]] = []

    if teacher_action == "answer_directly":
        status = str((teacher_support or {}).get("status") or "none")
        stats[f"teacher_answer:{status}"] += 1
        if status == "unsupported" or (drop_uncertain_teacher_answers and status == "uncertain"):
            stats["drop:teacher_answer_not_supported"] += 1
            # If the student chose to retrieve, treat that as the safer positive
            # for this exact evidence state. Do not train the teacher's hidden-prior answer.
            promoted = False
            for student in students:
                student_payload = parse_jsonish(response_text(student))
                if action(student_payload) == "retrieve_more":
                    output.append(
                        normalize_record(
                            student,
                            kto_tag=True,
                            reason="promote_student_retrieve_more_when_teacher_answer_not_supported",
                            support=teacher_support,
                        )
                    )
                    promoted = True
                    stats["promote:student_retrieve_more"] += 1
            if not promoted:
                stats["drop:no_safe_positive_after_teacher_unsupported"] += len(records)
            return output
        output.append(normalize_record(teacher, kto_tag=True, reason="keep_teacher_answer", support=teacher_support))
        for student in students:
            student_payload = parse_jsonish(response_text(student))
            student_action = action(student_payload)
            student_support = (
                answer_support(answer(student_payload), evidence, question)
                if student_action == "answer_directly"
                else None
            )
            if student_action == "answer_directly" and student_support and student_support.get("status") in {"supported", "uncertain"}:
                stats["drop:student_answer_alt_not_rejected"] += 1
                continue
            output.append(
                normalize_record(
                    student,
                    kto_tag=False,
                    reason="reject_student_vs_supported_teacher_answer",
                    support=student_support,
                )
            )
        stats["keep:teacher_answer_group"] += len(output)
        return output

    # For retrieve_more / abstain / clarify_user teacher labels, keep the same-state
    # preference pair. This is the cleanest signal against premature answers.
    output.append(normalize_record(teacher, kto_tag=True, reason=f"keep_teacher_{teacher_action or 'unknown'}"))
    for student in students:
        output.append(normalize_record(student, kto_tag=False, reason=f"reject_student_vs_teacher_{teacher_action or 'unknown'}"))
    stats[f"keep:teacher_{teacher_action or 'unknown'}_group"] += len(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verifier-lite SODA/KTO dataset from existing student-state teacher replay records.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="soda_blackbox_deepseek_v1_550_parallel_verifier_lite_v1")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--drop-uncertain-teacher-answers", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    audit_path = input_dir / "audit_records.jsonl"
    if not audit_path.exists():
        raise SystemExit(f"Missing audit_records.jsonl: {audit_path}")

    records = read_jsonl(audit_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        prompt_key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        groups[prompt_key].append(record)

    stats: Counter = Counter()
    output_records: list[dict[str, Any]] = []
    decisions_path = output_dir / "verifier_lite_decisions.jsonl"
    if decisions_path.exists():
        decisions_path.unlink()
    for prompt_key, group in groups.items():
        before = len(output_records)
        processed = process_group(
            group,
            stats=stats,
            drop_uncertain_teacher_answers=args.drop_uncertain_teacher_answers,
        )
        output_records.extend(processed)
        append_jsonl(
            decisions_path,
            {
                "prompt_key": prompt_key,
                "task_type": group[0].get("task_type") if group else "",
                "input_records": len(group),
                "output_records": len(output_records) - before,
                "decision_reasons": [
                    record.get("meta", {}).get("verifier_lite_reason")
                    for record in processed
                ],
            },
        )

    train, val = split_records(output_records, seed=args.seed, val_ratio=args.val_ratio)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_records": len(records),
        "input_prompt_groups": len(groups),
        "output_records": len(output_records),
        "train_records": len(train),
        "val_records": len(val),
        "drop_uncertain_teacher_answers": args.drop_uncertain_teacher_answers,
        "stats": dict(sorted(stats.items())),
    }
    write_json(output_dir / "build_summary.json", summary)
    report_lines = [
        "# SODA Verifier-Lite Build Report",
        "",
        f"- input_records: {len(records)}",
        f"- input_prompt_groups: {len(groups)}",
        f"- output_records: {len(output_records)}",
        f"- train_records: {len(train)}",
        f"- val_records: {len(val)}",
        "",
        "## Stats",
        "",
    ]
    report_lines.extend(f"- {key}: {value}" for key, value in sorted(stats.items()))
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
