#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ABSTAIN_RE = re.compile(
    r"(证据不足|不足以确认|无法确认|不能确认|检索证据不足|无法判断|缺少足以|缺乏足以|不足以完整回答|未给出明确)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "_source_file": str(path),
                        "_line_no": line_no,
                        "_decode_error": str(exc),
                        "_raw": line[:500],
                    }
                )
                continue
            if isinstance(payload, dict):
                payload["_source_file"] = str(path)
                payload["_line_no"] = line_no
                records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_text(value: Any, limit: int = 900) -> str:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]


def parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def string_similarity(left: str, right: str) -> float:
    left = re.sub(r"\s+", "", left or "")
    right = re.sub(r"\s+", "", right or "")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def normalize_question(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    text = text.strip("\"'“”‘’")
    return re.sub(r"[?？。！!，,、：:；;\"'“”‘’（）()《》〈〉\[\]【】]", "", text)


def list_overlap(left: Any, right: Any) -> tuple[int, int, int]:
    left_set = {str(item).strip() for item in left or [] if str(item).strip()} if isinstance(left, list) else set()
    right_set = {str(item).strip() for item in right or [] if str(item).strip()} if isinstance(right, list) else set()
    return len(left_set & right_set), len(left_set), len(right_set)


def classify_pair(record: dict[str, Any]) -> tuple[str | None, str]:
    task_type = str(record.get("task_type") or "unknown")
    student = parse_json_object(record.get("student_output"))
    teacher = parse_json_object(record.get("teacher_output"))
    if record.get("_decode_error"):
        return "decode_error", record["_decode_error"]
    if not record.get("teacher_valid", True):
        return None, "teacher invalid, skip"
    if not record.get("student_valid", True):
        return "student_invalid", "student output is invalid while teacher is valid"
    if student is None or teacher is None:
        if compact_text(record.get("student_output")) != compact_text(record.get("teacher_output")):
            return "raw_text_divergence", "student and teacher raw outputs differ"
        return None, "same raw output"

    if task_type == "conclusion_generation":
        s_action = str(student.get("next_action") or "")
        t_action = str(teacher.get("next_action") or "")
        s_answer = str(student.get("answer") or "")
        t_answer = str(teacher.get("answer") or "")
        if s_action != t_action:
            if s_action in {"abstain", "retrieve_more"} and t_action == "answer_directly":
                return "over_abstain_or_retrieve", f"student={s_action}, teacher can answer"
            if s_action == "answer_directly" and t_action in {"abstain", "retrieve_more"}:
                return "premature_answer", f"student answers but teacher={t_action}"
            return "action_mismatch", f"student={s_action}, teacher={t_action}"
        if s_action == "answer_directly":
            sim = string_similarity(s_answer, t_answer)
            if sim < 0.42:
                return "answer_divergence", f"both answer_directly but similarity={sim:.3f}"
            if len(s_answer) < max(24, len(t_answer) * 0.35):
                return "under_answer", "student answer is much shorter than teacher answer"
        return None, "no strong negative signal"

    if task_type in {"user_question_hypothesis_generation", "follow_up_hypothesis_generation"}:
        s_query_type = str(student.get("query_type") or "")
        t_query_type = str(teacher.get("query_type") or "")
        if s_query_type and t_query_type and s_query_type != t_query_type:
            return "query_type_mismatch", f"student={s_query_type}, teacher={t_query_type}"
        entity_common, s_entities, t_entities = list_overlap(student.get("entities"), teacher.get("entities"))
        keyword_common, s_keywords, t_keywords = list_overlap(student.get("keywords"), teacher.get("keywords"))
        if t_entities and entity_common == 0:
            return "entity_set_mismatch", f"entity overlap=0/{s_entities},{t_entities}"
        if t_keywords >= 3 and keyword_common <= 1:
            return "keyword_set_mismatch", f"keyword overlap={keyword_common}/{s_keywords},{t_keywords}"
        return None, "no strong negative signal"

    if compact_text(record.get("student_output")) != compact_text(record.get("teacher_output")):
        return "raw_text_divergence", "unknown task outputs differ"
    return None, "same output"


def classify_final_answer(record: dict[str, Any]) -> tuple[str | None, str]:
    question = str(record.get("question") or "")
    answer = str(record.get("answer") or "")
    if record.get("error"):
        return "runtime_error", str(record.get("error"))
    if not answer.strip():
        return "empty_answer", "empty answer"
    if ABSTAIN_RE.search(answer):
        return "final_abstain", "answer abstains or says evidence is insufficient"
    if "当前证据只能确认以下片段事实" in answer or len(re.findall(r"(证据|片段)\s*\d", answer)) >= 2:
        return "evidence_dump_answer", "answer dumps evidence snippets instead of synthesizing"
    if "澄闪" in question and "夏栎" in answer:
        return "entity_mismatch", "question asks 澄闪 but answer maps 澄闪 to 夏栎"
    if "红龙" in question and "拉芙希妮" in answer and "爱布拉娜" not in answer:
        return "entity_mismatch", "question asks 红龙 but answer maps it to 拉芙希妮 without 爱布拉娜"
    if "炎景公主" in question and ("望" in answer or "司岁台" in answer) and "陈" not in answer:
        return "likely_hallucinated_event", "炎景公主 answer drifts to 望/司岁台 without 陈氏 evidence"
    if "博士" in question and "全舰防御" in question and not any(term in answer for term in ("PRTS", "Abyss", "核心", "后门", "数据")):
        return "likely_unsupported_cause", "博士关闭全舰防御 answer lacks PRTS/Abyss/core evidence anchor"
    if "特雷西斯" in question and "巴别塔" in answer:
        return "entity_or_plan_confusion", "特雷西斯 answer introduces 巴别塔 as actor/target"
    if "塔露拉" in question and "谋取利益" in answer:
        return "likely_unsupported_motive", "塔露拉 answer includes profit-seeking motive"
    if "西西里人" in question and "意大利" in answer:
        return "real_world_contamination", "叙拉古语境 question drifts to real-world Italy/Sicily explanation"
    return None, "no heuristic negative signal"


def build_teacher_answer_map(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("task_type") or "") != "conclusion_generation":
            continue
        if not record.get("teacher_valid", True):
            continue
        teacher = parse_json_object(record.get("teacher_output"))
        if not teacher:
            continue
        key = normalize_question(record.get("question") or teacher.get("question"))
        if not key:
            continue
        by_question[key].append(
            {
                "action": str(teacher.get("next_action") or ""),
                "answer": str(teacher.get("answer") or ""),
                "raw": compact_text(record.get("teacher_output"), 1600),
            }
        )
    return by_question


def select_teacher_reference(question: str, teacher_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = teacher_map.get(normalize_question(question), [])
    if not candidates:
        return None
    answerable = [item for item in candidates if item.get("action") == "answer_directly" and item.get("answer")]
    if answerable:
        return max(answerable, key=lambda item: len(str(item.get("answer") or "")))
    return max(candidates, key=lambda item: len(str(item.get("answer") or "")))


def classify_final_against_teacher(
    record: dict[str, Any],
    teacher_map: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, str, str]:
    answer = str(record.get("answer") or "")
    teacher = select_teacher_reference(str(record.get("question") or ""), teacher_map)
    if not teacher:
        return None, "no matched raw-pair teacher reference", ""
    teacher_action = str(teacher.get("action") or "")
    teacher_answer = str(teacher.get("answer") or "")
    teacher_raw = str(teacher.get("raw") or "")
    if teacher_action == "answer_directly" and teacher_answer:
        if ABSTAIN_RE.search(answer):
            return "final_over_abstain_vs_teacher", "matched teacher can answer but final answer abstains", teacher_raw
        sim = string_similarity(answer, teacher_answer)
        if sim < 0.28:
            return "final_teacher_divergence", f"matched teacher answer similarity={sim:.3f}", teacher_raw
        if len(answer) < max(24, len(teacher_answer) * 0.35):
            return "final_under_answer_vs_teacher", "final answer is much shorter than matched teacher answer", teacher_raw
    if teacher_action in {"retrieve_more", "abstain"} and answer and not ABSTAIN_RE.search(answer):
        return "final_premature_answer_vs_teacher", f"matched teacher action={teacher_action}", teacher_raw
    return None, "final answer aligns enough with matched teacher", teacher_raw


def build_pair_case(record: dict[str, Any], category: str, reason: str) -> dict[str, Any]:
    return {
        "source_type": "soda_raw_pair",
        "source_file": record.get("_source_file"),
        "line_no": record.get("_line_no"),
        "question": record.get("question"),
        "task_type": record.get("task_type"),
        "category": category,
        "reason": reason,
        "student_output": compact_text(record.get("student_output"), 1600),
        "teacher_output": compact_text(record.get("teacher_output"), 1600),
        "student_valid": record.get("student_valid"),
        "teacher_valid": record.get("teacher_valid"),
        "prompt_key": record.get("prompt_key"),
        "question_key": record.get("question_key"),
    }


def build_final_case(record: dict[str, Any], category: str, reason: str, teacher_output: str = "") -> dict[str, Any]:
    return {
        "source_type": "soda_final_eval",
        "source_file": record.get("_source_file"),
        "line_no": record.get("_line_no"),
        "question": record.get("question"),
        "task_type": "final_answer",
        "category": category,
        "reason": reason,
        "student_output": compact_text(record.get("answer"), 1800),
        "teacher_output": compact_text(teacher_output, 1800),
        "elapsed_sec": record.get("elapsed_sec"),
        "error": record.get("error"),
    }


def write_report(path: Path, cases: list[dict[str, Any]], all_pair_count: int, all_final_count: int) -> None:
    by_source = Counter(case["source_type"] for case in cases)
    by_task = Counter(str(case.get("task_type") or "") for case in cases)
    by_category = Counter(str(case.get("category") or "") for case in cases)
    category_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        category_by_task[str(case.get("task_type") or "")][str(case.get("category") or "")] += 1

    lines = [
        "# SODA Negative Case Mining Report",
        "",
        f"- raw_pair_records_scanned: {all_pair_count}",
        f"- final_eval_records_scanned: {all_final_count}",
        f"- negative_cases: {len(cases)}",
        "",
        "## By Source",
    ]
    for key, value in by_source.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Task"])
    for key, value in by_task.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Category"])
    for key, value in by_category.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Category By Task"])
    for task, counter in sorted(category_by_task.items()):
        joined = ", ".join(f"{key}={value}" for key, value in counter.most_common())
        lines.append(f"- {task}: {joined}")
    lines.extend(["", "## High Signal Samples"])
    priority = {
        "entity_mismatch",
        "likely_hallucinated_event",
        "likely_unsupported_cause",
        "real_world_contamination",
        "final_teacher_divergence",
        "final_premature_answer_vs_teacher",
        "final_over_abstain_vs_teacher",
        "over_abstain_or_retrieve",
        "premature_answer",
        "answer_divergence",
    }
    selected = [case for case in cases if case.get("category") in priority][:30]
    for index, case in enumerate(selected, 1):
        lines.append("")
        lines.append(f"### {index}. {case.get('category')} / {case.get('task_type')}")
        lines.append(f"- question: {case.get('question')}")
        lines.append(f"- reason: {case.get('reason')}")
        lines.append(f"- student: {compact_text(case.get('student_output'), 360)}")
        if case.get("teacher_output"):
            lines.append(f"- teacher: {compact_text(case.get('teacher_output'), 360)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine high-signal SODA negative cases from raw pairs and final eval outputs.")
    parser.add_argument(
        "--raw-pairs",
        type=Path,
        action="append",
        default=[],
        help="SODA raw_pairs.jsonl files containing student/teacher outputs.",
    )
    parser.add_argument(
        "--final-eval",
        type=Path,
        action="append",
        default=[],
        help="End-to-end eval jsonl files containing question/answer outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap after sorting; 0 means no cap.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_pair_paths = args.raw_pairs or [
        Path("data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel/raw_pairs.jsonl")
    ]
    final_eval_paths = args.final_eval or [
        Path("outputs/retrieval_eval/soda_blackbox_hard10_20260530.jsonl"),
        Path("outputs/retrieval_eval/soda_blackbox_extra10_20260530.jsonl"),
        Path("outputs/retrieval_eval/soda_blackbox_checkpoint300_focus4_20260530.jsonl"),
    ]
    pair_records: list[dict[str, Any]] = []
    for path in raw_pair_paths:
        pair_records.extend(read_jsonl(path if path.is_absolute() else PROJECT_ROOT / path))
    final_records: list[dict[str, Any]] = []
    for path in final_eval_paths:
        final_records.extend(read_jsonl(path if path.is_absolute() else PROJECT_ROOT / path))

    cases: list[dict[str, Any]] = []
    for record in pair_records:
        category, reason = classify_pair(record)
        if category:
            cases.append(build_pair_case(record, category, reason))
    teacher_map = build_teacher_answer_map(pair_records)
    for record in final_records:
        category, reason = classify_final_answer(record)
        teacher_output = ""
        if not category:
            category, reason, teacher_output = classify_final_against_teacher(record, teacher_map)
        if category:
            cases.append(build_final_case(record, category, reason, teacher_output))

    priority_order = {
        "entity_mismatch": 0,
        "likely_hallucinated_event": 0,
        "likely_unsupported_cause": 0,
        "real_world_contamination": 0,
        "final_teacher_divergence": 1,
        "final_premature_answer_vs_teacher": 1,
        "final_over_abstain_vs_teacher": 1,
        "premature_answer": 1,
        "over_abstain_or_retrieve": 1,
        "answer_divergence": 2,
        "action_mismatch": 3,
        "query_type_mismatch": 4,
    }
    cases.sort(key=lambda case: (priority_order.get(str(case.get("category")), 9), str(case.get("task_type")), str(case.get("question"))))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    write_jsonl(output_dir / "negative_cases.jsonl", cases)
    write_report(output_dir / "report.md", cases, len(pair_records), len(final_records))
    summary = {
        "raw_pair_records_scanned": len(pair_records),
        "final_eval_records_scanned": len(final_records),
        "negative_cases": len(cases),
        "by_category": Counter(str(case.get("category") or "") for case in cases),
        "by_task": Counter(str(case.get("task_type") or "") for case in cases),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
