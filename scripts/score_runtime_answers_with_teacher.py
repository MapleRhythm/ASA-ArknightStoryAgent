#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
from threading import Lock
import time
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import TeacherApiConfig, call_teacher_api  # noqa: E402


WRITE_LOCK = Lock()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_teacher_payload(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("teacher payload is not JSON object")
    return payload


def compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 14].rstrip() + "...[TRUNCATED]"
    return text


def final_action(row: dict[str, Any]) -> str:
    trace = row.get("retrieval_trace") or []
    if not trace:
        return ""
    return str((trace[-1] or {}).get("planner_action") or "")


def action_sequence(row: dict[str, Any]) -> str:
    trace = row.get("retrieval_trace") or []
    return ">".join(str((item or {}).get("planner_action") or "") for item in trace)


def compact_evidence_item(item: dict[str, Any], index: int, *, max_chars: int) -> dict[str, Any]:
    return {
        "rank": index + 1,
        "id": item.get("id"),
        "activity_name": item.get("activity_name"),
        "story_name": item.get("story_name"),
        "stage_code": item.get("stage_code"),
        "avg_tag": item.get("avg_tag"),
        "source_path": item.get("source_path"),
        "rerank_score": item.get("rerank_score"),
        "fusion_score": item.get("fusion_score"),
        "dense_score": item.get("dense_score"),
        "sparse_score": item.get("sparse_score"),
        "minirag_score": item.get("minirag_score"),
        "clean_text": compact_text(item.get("clean_text"), max_chars=max_chars),
    }


def compact_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step in row.get("retrieval_trace") or []:
        if not isinstance(step, dict):
            continue
        output.append(
            {
                "round": step.get("round"),
                "planner_action": step.get("planner_action"),
                "queries": step.get("queries"),
                "next_round_queries": step.get("next_round_queries"),
                "missing_slots": step.get("missing_slots"),
                "conclusion": step.get("conclusion"),
                "follow_up_hypothesis": step.get("follow_up_hypothesis"),
            }
        )
    return output


def compact_case_for_prompt(case: dict[str, Any]) -> dict[str, Any]:
    trace = []
    for step in case.get("retrieval_trace") or []:
        if not isinstance(step, dict):
            continue
        hypothesis = step.get("hypothesis") if isinstance(step.get("hypothesis"), dict) else {}
        follow = step.get("follow_up_hypothesis") if isinstance(step.get("follow_up_hypothesis"), dict) else {}
        trace.append(
            {
                "round": step.get("round"),
                "action": step.get("planner_action"),
                "entities": hypothesis.get("entities") or follow.get("entities"),
                "keywords": hypothesis.get("keywords") or follow.get("keywords"),
                "queries": step.get("queries"),
                "next_queries": step.get("next_round_queries"),
            }
        )
    evidence = []
    for item in case.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                "rank": item.get("rank"),
                "id": item.get("id"),
                "story": item.get("story_name"),
                "text": item.get("clean_text"),
            }
        )
    return {
        "case_id": case.get("case_id"),
        "question": case.get("question"),
        "trace": trace,
        "final_action": case.get("final_action"),
        "answer": case.get("answer"),
        "evidence": evidence,
    }


def make_items(
    eval_dir: Path,
    *,
    label: str,
    datasets: list[str],
    only_action: str,
    evidence_top_k: int,
    evidence_chars: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dataset in datasets:
        for idx, row in enumerate(load_jsonl(eval_dir / f"{dataset}_answers.jsonl"), start=1):
            action = final_action(row)
            if row.get("error"):
                continue
            if only_action and action != only_action:
                continue
            evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
            items.append(
                {
                    "case_id": f"{label}:{dataset}:{idx}",
                    "run_label": label,
                    "dataset": dataset,
                    "index": idx,
                    "question": str(row.get("question") or ""),
                    "answer": str(row.get("answer") or ""),
                    "final_action": action,
                    "action_sequence": action_sequence(row),
                    "retrieval_trace": compact_trace(row),
                    "evidence": [
                        compact_evidence_item(item, i, max_chars=evidence_chars)
                        for i, item in enumerate(evidence[:evidence_top_k])
                        if isinstance(item, dict)
                    ],
                    "elapsed_sec": row.get("elapsed_sec"),
                }
            )
    return items


def build_system_prompt() -> str:
    return (
        "你是《明日方舟》RAG 输出的 evidence-only 评测器。"
        "你只能依据用户提供的 evidence 与 retrieval_trace 判断答案质量；禁止使用你自己的剧情知识。"
        "重点识别 unsupported claim、证据外实体/动机/因果、证据矛盾、答非所问。"
        "如果答案只说证据不足，不按幻觉扣分，但本任务通常只输入 answer_directly。"
        "只输出合法 JSON object，不输出 markdown 或解释性散文。"
    )


def build_compact_system_prompt() -> str:
    return (
        "你是 evidence-only verifier。只能根据输入 evidence 判断，禁止使用外部剧情知识。"
        "分别评价 retrieval_score/action_score/support_score/coverage_score。"
        "证据足够但 abstain=over_abstain；证据不足但 answer_directly=premature_answer；"
        "答案有证据外实体/动机/因果/结果=unsupported；答案和证据相反=contradicted。"
        "只输出 JSON，不要解释。"
    )


def build_additive_system_prompt() -> str:
    return (
        "你是《明日方舟》RAG 链路的 evidence-only 奖励裁判。"
        "只能依据输入的 question、retrieval_trace、final_action、answer、evidence 判断，禁止使用外部剧情知识。"
        "检索只分 good/partial/bad 三档；回答分别判断 evidence_support_level 与 logic_level；"
        "格式、冗余、quote 问题只做小幅加减分，不要压过事实正确性。"
        "abstain 是动作错误或正确放弃，不是事实陈述；不要因为 evidence 足够而给 abstain 标 contradicted。"
        "严格按给定加减分规则计算 reward_score，并输出合法 JSON object，不输出 markdown。"
    )


def build_user_prompt(batch: list[dict[str, Any]]) -> str:
    schema = {
        "scores": [
            {
                "case_id": "string",
                "answer_quality": "integer 0-5",
                "evidence_support": "integer 0-5",
                "question_coverage": "integer 0-5",
                "unsupported": "boolean",
                "unsupported_claims": ["string"],
                "contradicted": "boolean",
                "over_abstain": "boolean",
                "premature_answer": "boolean",
                "hallucination_risk": "low | medium | high",
                "usable_answer": "boolean",
                "reason": "short Chinese reason",
            }
        ]
    }
    rubric = [
        "answer_quality 5：完整、准确、无证据外事实；4：核心正确，轻微遗漏；3：部分正确但覆盖不足；2：关键遗漏或有明显弱支撑；1：大量错误/编造；0：无效输出。",
        "evidence_support 5：每个关键实体/因果/动机都有 evidence 直接支持；4：核心事实支持充分，少量泛化；3：部分事实支持；2：关键事实支撑不足；1：多数事实无证据；0：无证据或矛盾。",
        "question_coverage 5：完整回答问题所有问点；3：只答部分；1：基本没答。",
        "unsupported=true：答案中任何关键实体、身份、动机、因果、结果无法从 evidence 找到支撑。",
        "premature_answer=true：当前 evidence 不足以回答却直接给出确定答案。",
        "usable_answer=true：answer_quality>=4 且 evidence_support>=4 且 unsupported=false 且 contradicted=false。",
    ]
    payload = {
        "schema": schema,
        "rubric": rubric,
        "cases": batch,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_compact_user_prompt(batch: list[dict[str, Any]]) -> str:
    payload = {
        "rules": {
            "scores": "0-5; 5最好, 3部分可用, 1明显错误, 0无效",
            "retrieval_score": "hypothesis/query是否能召回正确证据",
            "action_score": "当前evidence下next_action是否正确",
            "support_score": "answer关键事实是否全由evidence支持; 非answer填0或3",
            "coverage_score": "answer是否覆盖question; 非answer按动作合理性给0或3",
            "label": "positive:四分>=4且无严重errors; negative:任一分<=2或严重errors; 其余neutral",
            "errors": [
                "invalid_json",
                "wrong_entity",
                "wrong_intent",
                "weak_query",
                "query_pollution",
                "premature_answer",
                "over_abstain",
                "over_retrieve",
                "unsupported",
                "contradicted",
                "partial_answer",
                "irrelevant_answer",
            ],
        },
        "output_schema": {
            "scores": [
                {
                    "case_id": "",
                    "retrieval_score": 0,
                    "action_score": 0,
                    "support_score": 0,
                    "coverage_score": 0,
                    "errors": [],
                    "label": "positive|negative|neutral",
                    "reason": "",
                }
            ]
        },
        "cases": [compact_case_for_prompt(item) for item in batch],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_additive_user_prompt(batch: list[dict[str, Any]]) -> str:
    payload = {
        "task": "按加减分制评估 RAG 检索与最终动作/回答。只根据 evidence，不使用外部知识。",
        "output_schema": {
            "scores": [
                {
                    "case_id": "string",
                    "retrieval_level": "good|partial|bad",
                    "evidence_support_level": "full|mostly|weak|none",
                    "logic_level": "correct|mostly_correct|flawed|wrong",
                    "action_error": "none|over_abstain|premature_answer|over_retrieve",
                    "serious_errors": [
                        "unsupported",
                        "contradicted",
                        "wrong_entity",
                        "irrelevant_answer",
                        "query_pollution",
                    ],
                    "format_penalty": "integer 0..3",
                    "reward_score": "integer -10..10",
                    "label": "positive|neutral|negative",
                    "reason": "short Chinese reason",
                }
            ]
        },
        "scoring_rules": {
            "retrieval_base": {
                "good": "+2: evidence 命中问题核心，足以支持回答，或足以判断当前证据确实不足",
                "partial": "0: 部分相关但缺关键实体、事件、动机或因果",
                "bad": "-2: 章节/实体/事件明显歪，top evidence 基本无关",
            },
            "evidence_support_base": {
                "full": "+3: answer 的关键实体、事件、动机、因果都能从 evidence 直接支持",
                "mostly": "+2: 核心事实有证据，只有轻微概括或少量非关键缺失",
                "weak": "-1: 只有部分事实支持，关键原因/结论缺证据",
                "none": "-3: 大量证据外事实、证据矛盾，或 answer 无法由 evidence 支持",
            },
            "logic_base": {
                "correct": "+2: 推理链闭合，回答了 question 的核心",
                "mostly_correct": "+1: 主结论基本对，但有轻微遗漏或弱推断",
                "flawed": "-1: 因果跳跃、部分答非所问、覆盖不足",
                "wrong": "-3: 实体混淆、立场反转、结论相反、严重答非所问",
            },
            "action_adjustment": {
                "none": "0: final_action 在当前 evidence 下合理",
                "over_abstain": "-3: evidence 足够回答但 final_action=abstain",
                "premature_answer": "-3: evidence 不足但 final_action=answer_directly",
                "over_retrieve": "-1: evidence 已足够但仍继续检索",
                "correct_abstain_bonus": "+1: final_action=abstain 且 evidence 确实不足，不乱答",
            },
            "serious_error_penalty": {
                "unsupported": "-2: 答案含 evidence 外关键实体、身份、动机、因果或结果",
                "contradicted": "-4: 仅当 final_action=answer_directly 且答案中的具体事实陈述与 evidence 相反时使用；abstain 本身不能标 contradicted",
                "wrong_entity": "-4: 问题实体和答案/证据实体混淆",
                "irrelevant_answer": "-3: 答案基本没回答问题",
                "query_pollution": "-2: 检索 query 混入 JSON 字段、校验错误、quote_not_found 等非问题语义",
            },
            "format_penalty": {
                "0": "格式干净",
                "1": "轻微重复、quote 太长/缺 quote、少量废话",
                "2": "明显重复、粘贴大段 evidence、结构混乱但仍可读",
                "3": "格式严重损坏或几乎不可读",
            },
            "label_rule": {
                "positive": "reward_score >= 4，且 action_error=none，且 serious_errors 为空",
                "negative": "reward_score <= -2，或 action_error in [over_abstain,premature_answer]，或 serious_errors 含 contradicted/wrong_entity/irrelevant_answer",
                "neutral": "其他情况",
            },
            "important_distinctions": [
                "final_action=abstain 且 evidence 足够回答：action_error=over_abstain；serious_errors 通常为空；不要标 contradicted。",
                "final_action=abstain 且 evidence 不足：action_error=none，并加 correct_abstain_bonus。",
                "final_action=answer_directly 且 evidence 不足：action_error=premature_answer；若答案还有证据外关键事实，再加 unsupported。",
                "contradicted 只用于答案中明确说 A，但 evidence 明确说非 A 或相反立场。",
            ],
        },
        "calculation": (
            "reward_score = retrieval_base + evidence_support_base + logic_base "
            "+ action_adjustment + correct_abstain_bonus_if_applicable "
            "+ serious_error_penalties - format_penalty。"
            "如果 final_action=abstain 且 evidence 不足，evidence_support_level 可填 weak，logic_level 可填 mostly_correct，"
            "action_error=none，并加 correct_abstain_bonus。"
        ),
        "cases": [compact_case_for_prompt(item) for item in batch],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def call_teacher_json(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    retries: int,
    retry_sleep: float,
    raw_output: Path,
    request_meta: dict[str, Any],
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, retries + 2):
        raw_text = None
        raw_response = None
        started = time.time()
        try:
            raw_text, raw_response = call_teacher_api(api_config, system_prompt=system_prompt, user_prompt=user_prompt)
            payload = parse_teacher_payload(raw_text)
            append_jsonl(
                raw_output,
                {
                    **request_meta,
                    "attempt": attempt,
                    "ok": True,
                    "latency": round(time.time() - started, 3),
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                },
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            append_jsonl(
                raw_output,
                {
                    **request_meta,
                    "attempt": attempt,
                    "ok": False,
                    "latency": round(time.time() - started, 3),
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "error": last_error,
                },
            )
            if attempt <= retries:
                time.sleep(retry_sleep)
    raise RuntimeError(last_error or "teacher call failed")


def normalize_additive_score(score: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if "reward_score" not in score:
        return score

    normalized = dict(score)
    raw_errors = normalized.get("serious_errors")
    if isinstance(raw_errors, list):
        errors = [str(item) for item in raw_errors if str(item)]
    else:
        errors = []

    # Abstain is an action decision, not a factual claim. Do not let the API
    # turn over-abstain into contradicted unless it also produced a concrete answer.
    if str(case.get("final_action") or "") == "abstain" and "contradicted" in errors:
        normalized.setdefault("raw_serious_errors", errors)
        errors = [item for item in errors if item != "contradicted"]
        normalized["serious_errors"] = errors

    try:
        reward_score = int(round(float(normalized.get("reward_score", 0))))
    except Exception:  # noqa: BLE001
        reward_score = 0
    normalized["reward_score"] = reward_score

    action_error = str(normalized.get("action_error") or "none")
    severe_errors = {"contradicted", "wrong_entity", "irrelevant_answer"}
    if reward_score <= -2 or action_error in {"over_abstain", "premature_answer"} or severe_errors.intersection(errors):
        label = "negative"
    elif reward_score >= 4 and action_error == "none" and not errors:
        label = "positive"
    else:
        label = "neutral"

    raw_label = str(normalized.get("label") or "")
    if raw_label != label:
        normalized.setdefault("raw_label", raw_label)
        normalized["label_normalized"] = True
    normalized["label"] = label
    return normalized


def score_batch(
    batch: list[dict[str, Any]],
    *,
    api_config: TeacherApiConfig,
    output_dir: Path,
    retries: int,
    retry_sleep: float,
    prompt_style: str,
) -> list[dict[str, Any]]:
    if prompt_style == "compact":
        system_prompt = build_compact_system_prompt()
        user_prompt = build_compact_user_prompt(batch)
    elif prompt_style == "additive":
        system_prompt = build_additive_system_prompt()
        user_prompt = build_additive_user_prompt(batch)
    else:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(batch)
    payload = call_teacher_json(
        api_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retries=retries,
        retry_sleep=retry_sleep,
        raw_output=output_dir / "raw_teacher_scores.jsonl",
        request_meta={"case_ids": [item["case_id"] for item in batch], "batch_size": len(batch)},
    )
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise ValueError("teacher payload missing scores list")
    by_id = {str(item.get("case_id")): item for item in batch}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score in scores:
        if not isinstance(score, dict):
            continue
        case_id = str(score.get("case_id") or "")
        if case_id not in by_id:
            continue
        score = normalize_additive_score(score, by_id[case_id])
        seen.add(case_id)
        output.append({"case": by_id[case_id], "score": score})
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"teacher omitted {len(missing)} case scores: {missing[:5]}")
    return output


def batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def score_value(score: dict[str, Any], key: str) -> float:
    try:
        return float(score.get(key))
    except Exception:  # noqa: BLE001
        return 0.0


def score_errors(score: dict[str, Any]) -> list[str]:
    additive_errors = score.get("serious_errors")
    output = []
    if isinstance(additive_errors, list):
        output.extend(str(item) for item in additive_errors if str(item))
    action_error = str(score.get("action_error") or "")
    if action_error and action_error != "none":
        output.append(action_error)
    if output:
        return output

    errors = score.get("errors")
    if isinstance(errors, list):
        return [str(item) for item in errors if str(item)]
    output = []
    if score.get("unsupported"):
        output.append("unsupported")
    if score.get("contradicted"):
        output.append("contradicted")
    error_type = str(score.get("error_type") or "")
    if error_type and error_type != "none":
        output.append(error_type)
    return output


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        run = str(record.get("case", {}).get("run_label") or "")
        by_run.setdefault(run, []).append(record)
    summary: dict[str, Any] = {"total_scored": len(records), "runs": {}}
    for run, rows in sorted(by_run.items()):
        compact_mode = any("retrieval_score" in (r.get("score") or {}) for r in rows)
        additive_mode = any("reward_score" in (r.get("score") or {}) for r in rows)
        if additive_mode:
            labels = Counter(str(r.get("score", {}).get("label") or "") for r in rows)
            errors = Counter(error for r in rows for error in score_errors(r.get("score", {})))
            summary["runs"][run] = {
                "count": len(rows),
                "datasets": dict(Counter(str(r.get("case", {}).get("dataset") or "") for r in rows)),
                "labels": dict(labels),
                "errors": dict(errors),
                "retrieval_levels": dict(Counter(str(r.get("score", {}).get("retrieval_level") or "") for r in rows)),
                "evidence_support_levels": dict(
                    Counter(str(r.get("score", {}).get("evidence_support_level") or "") for r in rows)
                ),
                "logic_levels": dict(Counter(str(r.get("score", {}).get("logic_level") or "") for r in rows)),
                "action_errors": dict(Counter(str(r.get("score", {}).get("action_error") or "") for r in rows)),
                "avg_reward_score": round(sum(score_value(r["score"], "reward_score") for r in rows) / len(rows), 3)
                if rows
                else 0.0,
            }
            continue

        if compact_mode:
            labels = Counter(str(r.get("score", {}).get("label") or "") for r in rows)
            errors = Counter(error for r in rows for error in score_errors(r.get("score", {})))
            summary["runs"][run] = {
                "count": len(rows),
                "datasets": dict(Counter(str(r.get("case", {}).get("dataset") or "") for r in rows)),
                "labels": dict(labels),
                "errors": dict(errors),
                "avg_retrieval_score": round(sum(score_value(r["score"], "retrieval_score") for r in rows) / len(rows), 3)
                if rows
                else 0.0,
                "avg_action_score": round(sum(score_value(r["score"], "action_score") for r in rows) / len(rows), 3)
                if rows
                else 0.0,
                "avg_support_score": round(sum(score_value(r["score"], "support_score") for r in rows) / len(rows), 3)
                if rows
                else 0.0,
                "avg_coverage_score": round(sum(score_value(r["score"], "coverage_score") for r in rows) / len(rows), 3)
                if rows
                else 0.0,
            }
            continue

        unsupported = [r for r in rows if bool(r.get("score", {}).get("unsupported"))]
        usable = [r for r in rows if bool(r.get("score", {}).get("usable_answer"))]
        risks = Counter(str(r.get("score", {}).get("hallucination_risk") or "") for r in rows)
        datasets = Counter(str(r.get("case", {}).get("dataset") or "") for r in rows)
        summary["runs"][run] = {
            "count": len(rows),
            "datasets": dict(datasets),
            "unsupported": len(unsupported),
            "unsupported_rate": round(len(unsupported) / len(rows), 4) if rows else 0.0,
            "usable_answer": len(usable),
            "usable_rate": round(len(usable) / len(rows), 4) if rows else 0.0,
            "hallucination_risk": dict(risks),
            "avg_answer_quality": round(sum(score_value(r["score"], "answer_quality") for r in rows) / len(rows), 3)
            if rows
            else 0.0,
            "avg_evidence_support": round(sum(score_value(r["score"], "evidence_support") for r in rows) / len(rows), 3)
            if rows
            else 0.0,
            "avg_question_coverage": round(sum(score_value(r["score"], "question_coverage") for r in rows) / len(rows), 3)
            if rows
            else 0.0,
        }
    return summary


def write_markdown(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = ["# Runtime Answer Teacher Score", ""]
    for run, payload in summary.get("runs", {}).items():
        lines.append(f"## {run}")
        for key, value in payload.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.append("## Unsupported Cases")
    for record in records:
        score = record.get("score", {})
        errors = score_errors(score)
        if not score.get("unsupported") and "unsupported" not in errors and "contradicted" not in errors:
            continue
        case = record.get("case", {})
        claims = score.get("unsupported_claims") or errors
        lines.append(f"- `{case.get('case_id')}` {case.get('question')}")
        lines.append(f"  - answer: {compact_text(case.get('answer'), max_chars=300)}")
        lines.append(f"  - claims: {compact_text(claims, max_chars=300)}")
        lines.append(f"  - reason: {compact_text(score.get('reason'), max_chars=300)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score runtime RAG answers with an evidence-only teacher.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--datasets", default="eval50,hard10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-action", default="answer_directly")
    parser.add_argument("--include-all-actions", action="store_true")
    parser.add_argument("--prompt-style", choices=("full", "compact", "additive"), default="full")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evidence-top-k", type=int, default=8)
    parser.add_argument("--evidence-chars", type=int, default=900)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "openai_responses"), default="chat_completions")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        raise SystemExit(f"Missing API key env: {args.api_key_env}")
    api_config = TeacherApiConfig(
        api_type=args.api_type,
        base_url=args.api_base,
        model=args.api_model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for label, raw_dir in args.run:
        items.extend(
            make_items(
                resolve_path(Path(raw_dir)),
                label=label,
                datasets=datasets,
                only_action="" if args.include_all_actions else args.only_action,
                evidence_top_k=args.evidence_top_k,
                evidence_chars=args.evidence_chars,
            )
        )
    if args.max_items > 0:
        items = items[: args.max_items]

    write_json(output_dir / "score_input_summary.json", {"items": len(items), "runs": args.run, "datasets": datasets})
    batches = batched(items, args.batch_size)
    scored_path = output_dir / "teacher_answer_scores.jsonl"
    if scored_path.exists():
        scored_path.unlink()

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                score_batch,
                batch,
                api_config=api_config,
                output_dir=output_dir,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
                prompt_style=args.prompt_style,
            )
            for batch in batches
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="score-runtime-answers"):
            batch_records = future.result()
            for record in batch_records:
                append_jsonl(scored_path, record)
            records.extend(batch_records)

    summary = summarize(records)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
