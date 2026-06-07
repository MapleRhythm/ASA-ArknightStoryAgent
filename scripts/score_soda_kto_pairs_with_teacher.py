#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sys
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


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored"

QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
EVIDENCE_BRIEF_RE = re.compile(
    r"(?s)^evidence_brief:\s*(.*?)(?:\nminirag_hints:|\noutput_schema:|\nfields:|\nrules:|\Z)",
    re.MULTILINE,
)
MINIRAG_HINTS_RE = re.compile(r"(?s)^minirag_hints:\s*(.*?)(?:\noutput_schema:|\nfields:|\nrules:|\Z)", re.MULTILINE)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_text(text: Any, *, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars] + "...[TRUNCATED]"
    return value


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
        raise ValueError("teacher payload is not a JSON object")
    return payload


def extract_first_user_prompt(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        if item.get("from") in {"human", "user"}:
            return str(item.get("value") or "")
    return ""


def extract_first_assistant_output(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        if item.get("from") in {"gpt", "assistant"}:
            return str(item.get("value") or "")
    return ""


def extract_question(prompt: str, record: dict[str, Any] | None = None) -> str:
    match = QUESTION_RE.search(prompt or "")
    if match:
        return match.group(1).strip()
    meta = record.get("meta", {}) if isinstance(record, dict) else {}
    source = meta.get("source") if isinstance(meta, dict) else {}
    if isinstance(source, dict) and source.get("question"):
        return str(source["question"]).strip()
    return ""


def extract_round(prompt: str) -> str:
    match = ROUND_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_evidence(prompt: str, *, max_chars: int) -> str:
    evidence_match = EVIDENCE_BRIEF_RE.search(prompt or "")
    if evidence_match:
        return compact_text(evidence_match.group(1), max_chars=max_chars)
    hints_match = MINIRAG_HINTS_RE.search(prompt or "")
    if hints_match:
        return compact_text(hints_match.group(1), max_chars=max_chars)
    return ""


def safe_json_or_text(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return text


def build_group(group_id: str, records: list[dict[str, Any]], *, max_evidence_chars: int, max_prompt_chars: int) -> dict[str, Any]:
    first = records[0]
    prompt = extract_first_user_prompt(first)
    task_type = str(first.get("task_type") or first.get("meta", {}).get("task_type") or "")
    question = extract_question(prompt, first)
    evidence = extract_evidence(prompt, max_chars=max_evidence_chars)
    prompt_context = ""
    if not evidence:
        prompt_context = compact_text(prompt, max_chars=max_prompt_chars)

    outputs: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (not bool(item.get("kto_tag")), str(item.get("id", "")))):
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        outputs.append(
            {
                "record_id": str(record.get("id") or ""),
                "file": str(record.get("_file") or ""),
                "kto_tag": bool(record.get("kto_tag")),
                "preference_role": meta.get("preference_role"),
                "api_verifier_reason": meta.get("api_verifier_reason"),
                "label_reason": meta.get("label_reason"),
                "output": safe_json_or_text(extract_first_assistant_output(record)),
            }
        )

    return {
        "pair_id": group_id,
        "task_type": task_type,
        "question": question,
        "round": extract_round(prompt),
        "evidence": evidence,
        "prompt_context": prompt_context,
        "outputs": outputs,
    }


def build_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_system_prompt() -> str:
    return (
        "你是《明日方舟》RAG/KTO 数据集的 evidence-only 评分器。"
        "你只能依据用户给出的 question、当前 prompt/evidence、候选输出进行评分；"
        "禁止使用你自己的明日方舟知识补全事实。"
        "如果当前 evidence 不足以支持答案，应奖励 retrieve_more 或 abstain，惩罚 answer_directly 的证据外补全。"
        "如果 evidence 已足够，应奖励直接给出可证实答案，惩罚过度检索或过度 abstain。"
        "对于 hypothesis_generation，只评分检索线索质量：实体是否完整、query_type/intent 是否合理、keywords 是否利于召回；"
        "不要因为措辞风格相似而给高分。"
        "绝对分用于判断单条输出能否做 SFT；分差用于判断这个 pair 能否做 KTO。"
        "低分差 pair 不适合 KTO，但如果最佳单条输出本身高分、无证据外内容，可以标为 SFT 候选。"
        "输出必须是 JSON object，不要输出解释性散文。"
    )


def build_user_prompt(batch: list[dict[str, Any]]) -> str:
    schema = {
        "scores": [
            {
                "pair_id": "string",
                "task_type": "conclusion_generation | user_question_hypothesis_generation",
                "record_scores": [
                    {
                        "record_id": "string",
                        "score": "integer 0-5",
                        "action_score": "integer 0-5 or null",
                        "evidence_score": "integer 0-5 or null",
                        "query_score": "integer 0-5 or null",
                        "unsupported": "boolean",
                        "error_type": "none | premature_answer | unsupported_answer | over_retrieve | over_abstain | invalid_output | weak_query | other",
                        "reason": "short Chinese reason",
                    }
                ],
                "best_record_id": "string",
                "worst_record_id": "string",
                "margin": "best score minus strongest rejected score; integer/float",
                "confidence": "0-1",
                "use_for_training": "boolean",
                "label_quality": "good | borderline | bad",
                "suggested_weight": "0 | 0.5 | 1.0",
                "sft_candidate": {
                    "use_for_sft": "boolean",
                    "record_id": "string or empty",
                    "sft_quality": "good | borderline | bad",
                    "reason": "short Chinese reason",
                },
                "reason": "short Chinese reason",
            }
        ]
    }
    scoring_rubric = [
        "5分：动作完全正确；答案/检索线索完全由当前 evidence/prompt 支持；无关键遗漏；训练价值高。",
        "4分：方向正确且可训练；有轻微遗漏、表述不够精确或 query 可再优化，但无关键 unsupported。",
        "3分：部分可用；核心方向大致正确，但证据使用不充分、缺少关键实体、missing_slots/query 有明显瑕疵。",
        "2分：不适合作为 chosen；早答、过度检索、弱 query、答案有重要 unsupported，或只能提供很弱监督。",
        "1分：明显错误；动作与 evidence 状态相反，或答案大段编造，或 query 严重偏题。",
        "0分：无效输出；不是要求的 JSON/schema，空输出，或无法用于训练。",
        "KTO 使用标准：best_score >= 4，margin >= 1.5，confidence >= 0.65，且 best 明显优于 rejected。",
        "SFT 使用标准：best_score >= 4，unsupported=false，动作/答案本身正确；即使 margin < 1.5，也可以 use_for_sft=true。",
        "低分差处理：如果两个输出都 4 分以上且差异只是措辞/轻微 query 粒度，use_for_training=false，但选更好的一个做 SFT 候选。",
        "如果 best 只是 retrieve_more，但 follow_up_hypothesis 断词严重、重复严重或 missing_slots 不清楚，SFT 只能 borderline 或不用。",
    ]
    examples = [
        {
            "case": "证据不足但候选直接回答且夹带先验",
            "score": 1,
            "error_type": "premature_answer/unsupported_answer",
            "decision": "KTO 可把 retrieve_more 作为 chosen；直接回答不做 SFT。",
        },
        {
            "case": "证据充分，候选准确 answer_directly，无证据外细节",
            "score": 5,
            "error_type": "none",
            "decision": "可做 KTO chosen；也可做 SFT。",
        },
        {
            "case": "证据充分但候选 retrieve_more",
            "score": 2,
            "error_type": "over_retrieve",
            "decision": "作为 rejected；不做 SFT。",
        },
        {
            "case": "hypothesis 两个输出都能召回，正样本实体略完整，负样本也不差",
            "score": "4 vs 3.5",
            "error_type": "weak_query",
            "decision": "margin 低，KTO 不用或降权；较好一条可做 SFT。",
        },
        {
            "case": "答案只回答 evidence 可确认部分，并明确不补未知部分",
            "score": 4,
            "error_type": "none",
            "decision": "可做 SFT；若 rejected 有编造，可做 KTO。",
        },
    ]
    requirements = [
        "每个 pair_id 都必须返回一个评分对象，数量要与输入 pairs 一致。",
        "严格按下方评分标准给 0-5 分，允许 0.5 小数。",
        "conclusion_generation 必须检查 next_action 是否正确、answer 是否被 evidence 支持、missing_slots/follow_up_hypothesis 是否合理。",
        "answer 中任何关键事实没有 evidence 支持，unsupported=true，并扣 evidence_score。",
        "hypothesis_generation 的 query_score 主要看能否帮助召回关键章节/角色/事件；正负差异很小时 margin 应低，label_quality=borderline。",
        "use_for_training 表示是否适合 KTO/偏好训练；use_for_sft 表示是否适合抽成 SFT 单条训练。",
        "use_for_training=true 仅当 best 比 rejected 有清楚优势，且 confidence >= 0.65；低分差 pair 不要强行标 KTO。",
        "sft_candidate 只能选择一个最佳 record；若没有单条达到 SFT 标准，record_id 置空且 use_for_sft=false。",
        "如果原 kto_tag=True 的输出不是 best，也照实打分并给 bad/borderline。",
        "不要使用外部知识判断答案事实真伪，只判断当前证据能否支持。",
    ]
    return (
        "请给下面这一批 KTO pair/group 打分。一次有多个 pair，用同一个 JSON 返回。\n\n"
        "评分要求：\n"
        + "\n".join(f"- {item}" for item in requirements)
        + "\n\n评分标准：\n"
        + "\n".join(f"- {item}" for item in scoring_rubric)
        + "\n\n评分示例：\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
        + "\n\n输出 JSON schema 示例：\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n输入 pairs：\n"
        + json.dumps(batch, ensure_ascii=False, indent=2)
    )


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
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        raw_text = None
        raw_response = None
        started = time.time()
        try:
            raw_text, raw_response = call_teacher_api(
                api_config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
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
                    "created_at": int(time.time()),
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
                    "created_at": int(time.time()),
                },
            )
            if attempt <= retries:
                time.sleep(retry_sleep)
    raise RuntimeError(last_error or "teacher api failed")


def load_existing_scores(path: Path) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return scored
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id") or "")
            if pair_id:
                scored[pair_id] = payload
    return scored


def normalize_score_payload(payload: dict[str, Any], batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError("teacher payload missing scores list")
    expected = {str(item["pair_id"]) for item in batch}
    output: list[dict[str, Any]] = []
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id") or "")
        if pair_id not in expected:
            continue
        item["pair_id"] = pair_id
        record_scores = item.get("record_scores")
        if not isinstance(record_scores, list):
            item["record_scores"] = []
        output.append(item)
    seen = {str(item.get("pair_id")) for item in output}
    missing = expected - seen
    if missing:
        raise ValueError(f"teacher payload missing pair ids: {sorted(missing)[:5]}")
    return output


def score_batches(
    groups: list[dict[str, Any]],
    *,
    api_config: TeacherApiConfig,
    batch_size: int,
    output_dir: Path,
    retries: int,
    retry_sleep: float,
    dry_run: bool,
    limit_batches: int | None,
) -> dict[str, dict[str, Any]]:
    score_jsonl = output_dir / "teacher_pair_scores.jsonl"
    raw_jsonl = output_dir / "teacher_pair_score_raw.jsonl"
    existing = load_existing_scores(score_jsonl)
    pending = [item for item in groups if str(item["pair_id"]) not in existing]
    batches = build_batches(pending, batch_size)
    if limit_batches is not None:
        batches = batches[:limit_batches]

    if dry_run:
        preview_dir = output_dir / "dry_run"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for index, batch in enumerate(batches[:3], start=1):
            (preview_dir / f"batch_{index:04d}_system.txt").write_text(build_system_prompt(), encoding="utf-8")
            (preview_dir / f"batch_{index:04d}_user.txt").write_text(build_user_prompt(batch), encoding="utf-8")
        print(f"[dry-run] groups_total={len(groups)} existing={len(existing)} pending={len(pending)} batches={len(batches)}")
        print(f"[dry-run] wrote prompt previews to {preview_dir}")
        return existing

    system_prompt = build_system_prompt()
    for batch_index, batch in enumerate(tqdm(batches, desc="teacher scoring batches"), start=1):
        pair_ids = [str(item["pair_id"]) for item in batch]
        payload = call_teacher_json(
            api_config,
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(batch),
            retries=retries,
            retry_sleep=retry_sleep,
            raw_output=raw_jsonl,
            request_meta={
                "batch_index": batch_index,
                "pair_ids": pair_ids,
                "api_type": api_config.api_type,
                "model": api_config.model,
                "base_url": api_config.base_url,
            },
        )
        normalized = normalize_score_payload(payload, batch)
        for item in normalized:
            item["_batch_index"] = batch_index
            item["_model"] = api_config.model
            append_jsonl(score_jsonl, item)
            existing[str(item["pair_id"])] = item
    return existing


def get_record_score(score_payload: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    record_scores = score_payload.get("record_scores")
    if not isinstance(record_scores, list):
        return None
    for item in record_scores:
        if isinstance(item, dict) and str(item.get("record_id") or "") == record_id:
            return item
    return None


def score_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def annotate_and_filter(
    records_by_file: dict[str, list[dict[str, Any]]],
    scores: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    dataset_name: str,
    min_margin: float,
    min_confidence: float,
    min_best_score: float,
    min_sft_score: float,
) -> dict[str, Any]:
    scored_dir = output_dir / "scored_dataset"
    filtered_dir = output_dir / "filtered_dataset"
    sft_dir = output_dir / "sft_candidate_dataset"
    scored_by_file: dict[str, list[dict[str, Any]]] = {}
    filtered_by_file: dict[str, list[dict[str, Any]]] = {}
    sft_by_file: dict[str, list[dict[str, Any]]] = {}
    stats: Counter[str] = Counter()

    for file_name, records in records_by_file.items():
        scored_records: list[dict[str, Any]] = []
        filtered_records: list[dict[str, Any]] = []
        sft_records: list[dict[str, Any]] = []
        for record in records:
            record = json.loads(json.dumps(record, ensure_ascii=False))
            record.pop("_file", None)
            meta = record.setdefault("meta", {})
            pair_id = str(meta.get("prompt_key") or record.get("id") or "")
            score_payload = scores.get(pair_id)
            if not score_payload:
                stats["unscored_records"] += 1
                scored_records.append(record)
                continue

            rec_score = get_record_score(score_payload, str(record.get("id") or ""))
            teacher_scoring = {
                "pair_id": pair_id,
                "record_score": rec_score,
                "pair_margin": score_payload.get("margin"),
                "pair_confidence": score_payload.get("confidence"),
                "pair_use_for_training": score_payload.get("use_for_training"),
                "pair_label_quality": score_payload.get("label_quality"),
                "pair_suggested_weight": score_payload.get("suggested_weight"),
                "sft_candidate": score_payload.get("sft_candidate"),
                "best_record_id": score_payload.get("best_record_id"),
                "worst_record_id": score_payload.get("worst_record_id"),
                "pair_reason": score_payload.get("reason"),
            }
            meta["teacher_scoring"] = teacher_scoring
            scored_records.append(record)

            margin = score_number(score_payload.get("margin"), default=-999.0)
            confidence = score_number(score_payload.get("confidence"), default=0.0)
            best_id = str(score_payload.get("best_record_id") or "")
            rec_score_value = score_number(rec_score.get("score") if isinstance(rec_score, dict) else None)
            best_score = 0.0
            for item in score_payload.get("record_scores") or []:
                if isinstance(item, dict) and str(item.get("record_id") or "") == best_id:
                    best_score = score_number(item.get("score"))
                    break
            use_pair = (
                bool(score_payload.get("use_for_training"))
                and margin >= min_margin
                and confidence >= min_confidence
                and best_score >= min_best_score
            )
            if use_pair and rec_score is not None:
                if bool(record.get("kto_tag")):
                    if str(record.get("id") or "") == best_id and rec_score_value >= min_best_score:
                        filtered_records.append(record)
                        stats["kept_positive_records"] += 1
                    else:
                        stats["dropped_positive_not_best"] += 1
                else:
                    filtered_records.append(record)
                    stats["kept_negative_records"] += 1
            else:
                stats["dropped_by_pair_filter"] += 1

            sft_candidate = score_payload.get("sft_candidate")
            if isinstance(sft_candidate, dict) and bool(sft_candidate.get("use_for_sft")):
                sft_record_id = str(sft_candidate.get("record_id") or "")
                if str(record.get("id") or "") == sft_record_id and rec_score is not None:
                    sft_score = score_number(rec_score.get("score"))
                    unsupported = bool(rec_score.get("unsupported"))
                    if sft_score >= min_sft_score and not unsupported:
                        sft_record = json.loads(json.dumps(record, ensure_ascii=False))
                        sft_record.pop("_file", None)
                        sft_record.pop("kto_tag", None)
                        sft_record.setdefault("meta", {})["teacher_scoring"] = teacher_scoring
                        sft_records.append(sft_record)
                        stats["kept_sft_candidate_records"] += 1
                    else:
                        stats["dropped_sft_candidate_by_record_score"] += 1

        scored_by_file[file_name] = scored_records
        filtered_by_file[file_name] = filtered_records
        sft_by_file[file_name] = sft_records
        write_json(scored_dir / file_name, scored_records)
        write_json(filtered_dir / file_name, filtered_records)
        write_json(sft_dir / file_name, sft_records)

    kto_dataset_info = {
        f"{dataset_name}_teacher_scored_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools", "kto_tag": "kto_tag"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        },
        f"{dataset_name}_teacher_scored_val": {
            "file_name": "val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools", "kto_tag": "kto_tag"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        },
    }
    sft_dataset_info = {
        f"{dataset_name}_sft_candidates_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        },
        f"{dataset_name}_sft_candidates_val": {
            "file_name": "val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        },
    }
    write_json(scored_dir / "dataset_info.json", kto_dataset_info)
    write_json(filtered_dir / "dataset_info.json", kto_dataset_info)
    write_json(sft_dir / "dataset_info.json", sft_dataset_info)
    stats_payload = {
        "scored_records": {name: len(items) for name, items in scored_by_file.items()},
        "filtered_records": {name: len(items) for name, items in filtered_by_file.items()},
        "sft_candidate_records": {name: len(items) for name, items in sft_by_file.items()},
        "stats": dict(stats),
        "filters": {
            "min_margin": min_margin,
            "min_confidence": min_confidence,
            "min_best_score": min_best_score,
            "min_sft_score": min_sft_score,
        },
    }
    write_json(output_dir / "summary.json", stats_payload)
    return stats_payload


def collect_groups(
    input_dir: Path,
    *,
    max_evidence_chars: int,
    max_prompt_chars: int,
    shuffle_seed: int,
    limit_groups: int | None,
    task_type: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records_by_file: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for file_name in ("train.json", "val.json"):
        path = input_dir / file_name
        records = load_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} is not a list")
        for record in records:
            if isinstance(record, dict):
                record["_file"] = file_name
        records_by_file[file_name] = records
        all_records.extend(record for record in records if isinstance(record, dict))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        group_id = str(meta.get("prompt_key") or record.get("id") or "")
        grouped[group_id].append(record)

    groups: list[dict[str, Any]] = []
    for group_id, records in grouped.items():
        if task_type != "all":
            first_task = str(records[0].get("task_type") or records[0].get("meta", {}).get("task_type") or "")
            if first_task != task_type:
                continue
        if not any(bool(item.get("kto_tag")) for item in records):
            continue
        if not any(not bool(item.get("kto_tag")) for item in records):
            continue
        groups.append(
            build_group(
                group_id,
                records,
                max_evidence_chars=max_evidence_chars,
                max_prompt_chars=max_prompt_chars,
            )
        )

    rng = random.Random(shuffle_seed)
    rng.shuffle(groups)
    if limit_groups is not None:
        groups = groups[:limit_groups]
    return records_by_file, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score SODA/KTO pairs with an evidence-only teacher verifier.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-evidence-chars", type=int, default=3000)
    parser.add_argument("--max-prompt-chars", type=int, default=1400)
    parser.add_argument("--task-type", choices=("all", "conclusion_generation", "user_question_hypothesis_generation"), default="all")
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260601)
    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "responses"), default="chat_completions")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--min-margin", type=float, default=1.5)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-best-score", type=float, default=3.5)
    parser.add_argument("--min-sft-score", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_file, groups = collect_groups(
        input_dir,
        max_evidence_chars=args.max_evidence_chars,
        max_prompt_chars=args.max_prompt_chars,
        shuffle_seed=args.shuffle_seed,
        limit_groups=args.limit_groups,
        task_type=args.task_type,
    )
    write_json(output_dir / "score_input_summary.json", {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "groups": len(groups),
        "batch_size": args.batch_size,
        "task_type": args.task_type,
        "max_evidence_chars": args.max_evidence_chars,
        "max_prompt_chars": args.max_prompt_chars,
    })
    print(f"[groups] {len(groups)} pair/groups from {input_dir}")

    api_config = TeacherApiConfig(
        api_type=args.api_type,
        base_url=args.api_base,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        json_mode=True,
    )
    if not args.dry_run and not os.environ.get(args.api_key_env):
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    scores = score_batches(
        groups,
        api_config=api_config,
        batch_size=args.batch_size,
        output_dir=output_dir,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        dry_run=args.dry_run,
        limit_batches=args.limit_batches,
    )
    if args.dry_run:
        return

    summary = annotate_and_filter(
        records_by_file,
        scores,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        min_margin=args.min_margin,
        min_confidence=args.min_confidence,
        min_best_score=args.min_best_score,
        min_sft_score=args.min_sft_score,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
