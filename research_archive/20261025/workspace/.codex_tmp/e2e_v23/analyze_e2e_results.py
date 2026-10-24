#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


ABSTAIN_PREFIXES = (
    "现有证据不足",
    "现有检索证据不足",
    "无法根据当前证据",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            malformed.append(
                {
                    "line": line_number,
                    "error": f"{type(exc).__name__}: {exc}",
                    "prefix": line[:240],
                }
            )
            continue
        if not isinstance(value, dict):
            malformed.append(
                {
                    "line": line_number,
                    "error": f"top-level {type(value).__name__}, expected object",
                    "prefix": line[:240],
                }
            )
            continue
        rows.append(value)
    return rows, malformed


def load_questions(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_fact(text: str) -> str:
    text = re.sub(r"^(?:根据当前证据可确认[:：]\s*)", "", str(text or "").strip())
    return re.sub(r"[\s，。；;、！？!?：:“”\"'‘’（）()…\.]+", "", text)


def payload_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    for field in ("normalized_output", "raw_output"):
        text = str(diagnostics.get(field) or "").strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def candidate_answer(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    explicit = str(payload.get("final_answer") or payload.get("answer") or "").strip()
    if explicit:
        return explicit
    facts = [
        str(item.get("fact") or "").strip()
        for item in payload.get("supported_facts") or []
        if isinstance(item, dict) and str(item.get("fact") or "").strip()
    ]
    return "；".join(facts)


def answer_flags(question: str, answer: str) -> list[str]:
    flags: list[str] = []
    stripped = str(answer or "").strip()
    if not stripped:
        return ["empty_answer"]
    if stripped.startswith(ABSTAIN_PREFIXES):
        flags.append("abstain_answer")
    if re.search(r"[。！？!?；;][。！？!?；;]+$", stripped):
        flags.append("repeated_terminal_punctuation")
    if "。；" in stripped or "；。" in stripped:
        flags.append("mixed_fact_punctuation")
    if stripped.count("根据当前证据可确认") > 1:
        flags.append("repeated_boilerplate")
    compact_question = re.sub(r"\s+", "", question).rstrip("?？。.!")
    compact_answer = re.sub(r"\s+", "", stripped).rstrip("?？。.!")
    if compact_question and compact_answer == compact_question:
        flags.append("question_echo")
    pieces = [
        normalized_fact(piece)
        for piece in re.split(r"[；;\n]+", stripped)
        if normalized_fact(piece)
    ]
    if len(pieces) != len(set(pieces)):
        flags.append("duplicate_fact_exact")
    for left_index, left in enumerate(pieces):
        for right in pieces[left_index + 1 :]:
            shorter, longer = sorted((left, right), key=len)
            if len(shorter) >= 8 and shorter in longer:
                flags.append("duplicate_fact_containment")
                return sorted(set(flags))
    return sorted(set(flags))


def analyze(
    rows: list[dict[str, Any]],
    *,
    malformed: list[dict[str, Any]],
    expected_questions: list[str] | None,
    run_log: str,
) -> dict[str, Any]:
    all_actions: Counter[str] = Counter()
    final_actions: Counter[str] = Counter()
    grounding_statuses: Counter[str] = Counter()
    parse_statuses: Counter[str] = Counter()
    packing_statuses: Counter[str] = Counter()
    grounding_issues: Counter[str] = Counter()
    grounding_issue_families: Counter[str] = Counter()
    answer_flag_counts: Counter[str] = Counter()
    rounds: Counter[str] = Counter()
    final_outcomes: Counter[str] = Counter()
    elapsed: list[float] = []
    row_reports: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        trace = row.get("retrieval_trace") or []
        rounds[str(len(trace))] += 1
        final_conclusion = (trace[-1].get("conclusion") or {}) if trace else {}
        final_action = str(final_conclusion.get("next_action") or "missing")
        final_actions[final_action] += 1
        answer = str(row.get("answer") or "")
        flags = answer_flags(str(row.get("question") or ""), answer)
        answer_flag_counts.update(flags)
        elapsed_value = row.get("elapsed_sec")
        if isinstance(elapsed_value, (int, float)):
            elapsed.append(float(elapsed_value))

        round_reports: list[dict[str, Any]] = []
        for trace_row in trace:
            conclusion = trace_row.get("conclusion") or {}
            diagnostics = conclusion.get("generation_diagnostics") or {}
            action = str(conclusion.get("next_action") or "missing")
            grounding_status = str(diagnostics.get("grounding_status") or "missing")
            parse_status = str(diagnostics.get("parse_status") or "missing")
            all_actions[action] += 1
            grounding_statuses[grounding_status] += 1
            parse_statuses[parse_status] += 1
            issues = [
                str(issue)
                for issue in (
                    diagnostics.get("grounding_issues")
                    or diagnostics.get("grounding_warnings")
                    or conclusion.get("grounding_warnings")
                    or []
                )
            ]
            grounding_issues.update(issues)
            grounding_issue_families.update(issue.split(":", 1)[0] for issue in issues)
            packing = trace_row.get("evidence_packing") or []
            packing_statuses.update(str(item.get("status") or "missing") for item in packing)
            included = [item for item in packing if item.get("status") == "included"]
            payload = payload_from_diagnostics(diagnostics)
            round_reports.append(
                {
                    "round": trace_row.get("round"),
                    "planner_action": trace_row.get("planner_action"),
                    "action": action,
                    "parse_status": parse_status,
                    "grounding_status": grounding_status,
                    "grounding_issues": issues,
                    "packing": {
                        "input": len(packing),
                        "included": len(included),
                        "skipped": len(packing) - len(included),
                        "budget_used_chars": max(
                            [int(item.get("budget_used_chars") or 0) for item in packing] or [0]
                        ),
                    },
                    "rejected_candidate_answer": (
                        candidate_answer(payload)
                        if grounding_status.startswith("rejected_to_")
                        else ""
                    ),
                }
            )

        final_diagnostics = final_conclusion.get("generation_diagnostics") or {}
        final_grounding = str(final_diagnostics.get("grounding_status") or "missing")
        if final_action == "answer_directly":
            final_outcome = "accepted_answer"
        elif final_grounding == "rejected_to_abstain":
            final_outcome = "validator_rejected_candidate"
        elif final_action == "abstain":
            final_outcome = "model_abstain"
        else:
            final_outcome = final_action
        final_outcomes[final_outcome] += 1

        row_reports.append(
            {
                "index": row_index,
                "question": row.get("question"),
                "answer": answer,
                "error": row.get("error"),
                "elapsed_sec": elapsed_value,
                "round_count": len(trace),
                "final_action": final_action,
                "final_grounding_status": final_grounding,
                "final_outcome": final_outcome,
                "answer_flags": flags,
                "rounds": round_reports,
            }
        )

    actual_questions = [str(row.get("question") or "").strip() for row in rows]
    denominator: dict[str, Any] = {
        "result_rows": len(rows),
        "malformed_rows": len(malformed),
        "unique_result_questions": len(set(actual_questions)),
    }
    if expected_questions is not None:
        denominator.update(
            {
                "expected_rows": len(expected_questions),
                "exact_order_match": actual_questions == expected_questions[: len(actual_questions)],
                "complete": actual_questions == expected_questions,
                "missing_expected_questions": [
                    question for question in expected_questions if question not in set(actual_questions)
                ],
                "unexpected_result_questions": [
                    question for question in actual_questions if question not in set(expected_questions)
                ],
            }
        )

    return {
        "version": "asa_e2e_structural_analysis_v1",
        "boundaries": [
            "Structural diagnostics only; this is not semantic answer accuracy.",
            "Validator-rejected candidates still require claim-level review before being counted correct.",
            "Accepted grounding does not establish semantic correctness.",
        ],
        "denominator": denominator,
        "errors": {
            "rows_with_error": sum(bool(row.get("error")) for row in rows),
            "malformed_jsonl_rows": malformed,
        },
        "counts": {
            "answers_nonempty": sum(bool(str(row.get("answer") or "").strip()) for row in rows),
            "rounds": dict(sorted(rounds.items())),
            "all_actions": dict(all_actions.most_common()),
            "final_actions": dict(final_actions.most_common()),
            "final_outcomes": dict(final_outcomes.most_common()),
            "grounding_statuses": dict(grounding_statuses.most_common()),
            "parse_statuses": dict(parse_statuses.most_common()),
            "packing_statuses": dict(packing_statuses.most_common()),
            "grounding_issue_families": dict(grounding_issue_families.most_common()),
            "grounding_issues": dict(grounding_issues.most_common()),
            "answer_flags": dict(answer_flag_counts.most_common()),
        },
        "latency_sec": {
            "count": len(elapsed),
            "mean": statistics.mean(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "p90": percentile(elapsed, 0.90),
            "p95": percentile(elapsed, 0.95),
            "max": max(elapsed) if elapsed else None,
            "sum": sum(elapsed),
        },
        "log_signals": {
            "invalid_hypothesis_intent": len(
                re.findall(r"invalid hypothesis intent", run_log, flags=re.IGNORECASE)
            ),
            "hypothesis_fallback": len(
                re.findall(r"hypothesis.*fallback|fallback.*hypothesis", run_log, flags=re.IGNORECASE)
            ),
            "generation_failure": len(
                re.findall(r"generation.*(?:failed|failure|error)", run_log, flags=re.IGNORECASE)
            ),
            "context_truncation_mentions": len(
                re.findall(r"truncat|context length|max[_ ]model[_ ]len", run_log, flags=re.IGNORECASE)
            ),
            "fla_shape_warnings": len(
                re.findall(r"potential format mismatch", run_log, flags=re.IGNORECASE)
            ),
        },
        "rows": row_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows, malformed = load_rows(args.results)
    questions = load_questions(args.questions)
    run_log = (
        args.run_log.read_text(encoding="utf-8", errors="replace")
        if args.run_log is not None and args.run_log.exists()
        else ""
    )
    report = analyze(
        rows,
        malformed=malformed,
        expected_questions=questions,
        run_log=run_log,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
