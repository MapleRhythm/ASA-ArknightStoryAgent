#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
        raise ValueError("teacher payload is not an object")
    return payload


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
                    "api_type": api_config.api_type,
                    "model": api_config.model,
                    "base_url": api_config.base_url,
                    "latency": round(time.time() - started, 3),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
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
                    "api_type": api_config.api_type,
                    "model": api_config.model,
                    "base_url": api_config.base_url,
                    "latency": round(time.time() - started, 3),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "error": last_error,
                    "created_at": int(time.time()),
                },
            )
            if attempt <= retries:
                time.sleep(retry_sleep)
    raise RuntimeError(last_error or "teacher api failed")


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/opd_teacher_scores/deepseek_v4_flash_v1"
WRITE_LOCK = Lock()
LOG_LOCK = Lock()


def log(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def extract_candidate_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    conversations = raw.get("conversations") if isinstance(raw.get("conversations"), list) else []
    prompt = ""
    candidate = ""
    if conversations:
        for item in conversations:
            if not isinstance(item, dict):
                continue
            role = item.get("from") or item.get("role")
            value = str(item.get("value") or item.get("content") or "")
            if role in {"human", "user"} and not prompt:
                prompt = value
            elif role in {"gpt", "assistant"} and not candidate:
                candidate = value
    question = (
        first_string(raw, ("question", "source_question"))
        or first_string(meta, ("source_question", "question"))
    )
    task_type = (
        first_string(raw, ("task_type", "task"))
        or first_string(meta, ("task_family", "task_type"))
        or infer_task_type(prompt, candidate)
    )
    candidate_value = first_value(raw, ("candidate", "output", "assistant_payload", "response"))
    if candidate_value is not None:
        candidate = candidate_value if isinstance(candidate_value, str) else compact_json(candidate_value)
    prompt = first_string(raw, ("prompt", "student_prompt", "input")) or prompt
    dialogue_context = first_string(raw, ("dialogue_context",)) or first_string(meta, ("source_dialogue_context",))
    evidence = first_value(raw, ("evidence", "evidence_brief", "evidence_chain", "retrieved_evidence"))
    retrieval_metrics = first_value(raw, ("retrieval_metrics", "metrics", "trace", "retrieval_trace"))
    gold = first_value(raw, ("gold", "gold_answer", "gold_evidence", "expected_answer", "reference"))
    candidate_id = (
        first_string(raw, ("candidate_id", "id", "record_id"))
        or f"candidate-{index:08d}"
    )
    return {
        "candidate_id": candidate_id,
        "source_index": index,
        "task_type": task_type,
        "question": question,
        "dialogue_context": dialogue_context,
        "prompt": prompt,
        "candidate": candidate,
        "evidence": evidence,
        "retrieval_metrics": retrieval_metrics,
        "gold": gold,
        "raw": raw,
    }


def infer_task_type(prompt: str, candidate: str) -> str:
    text = f"{prompt}\n{candidate}"
    if "follow_up_hypothesis" in text:
        return "follow_up_hypothesis_generation"
    if "conclusion" in text or "next_action" in text:
        return "conclusion_generation"
    if "hypothesis" in text or "entities" in text or "keywords" in text:
        return "user_question_hypothesis_generation"
    return "unknown"


def build_teacher_system() -> str:
    return (
        "你是《明日方舟》RAG/OPD 训练数据的教师评分器。"
        "你只负责评价候选输出是否适合继续训练 4B 学生模型。"
        "只输出一个合法 JSON 对象，不输出 markdown、解释或思维过程。"
    )


def render_optional_json(value: Any, *, max_chars: int) -> str:
    if value is None or value == "":
        return "无"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    return text[:max_chars]


def compact_full_chain_candidate(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        return candidate
    trace = candidate.get("retrieval_trace") if isinstance(candidate.get("retrieval_trace"), list) else []
    compact_trace: list[dict[str, Any]] = []
    for step in trace[:3]:
        if not isinstance(step, dict):
            continue
        conclusion = step.get("conclusion") if isinstance(step.get("conclusion"), dict) else {}
        hypothesis = step.get("hypothesis") if isinstance(step.get("hypothesis"), dict) else {}
        follow = step.get("follow_up_hypothesis") if isinstance(step.get("follow_up_hypothesis"), dict) else None
        compact_step: dict[str, Any] = {
            "round": step.get("round"),
            "planner_action": step.get("planner_action") or conclusion.get("next_action") or "",
            "queries": [str(item)[:180] for item in (step.get("queries") or [])[:2]],
            "hypothesis": {
                "query_type": hypothesis.get("query_type"),
                "entities": (hypothesis.get("entities") or [])[:8],
                "keywords": (hypothesis.get("keywords") or [])[:8],
            },
            "missing_slots": (step.get("missing_slots") or conclusion.get("missing_slots") or [])[:4],
            "conclusion": {
                "next_action": conclusion.get("next_action") or "",
                "answer_preview": str(conclusion.get("answer") or "")[:260],
            },
        }
        if follow:
            compact_step["follow_up_hypothesis"] = {
                "query_type": follow.get("query_type"),
                "entities": (follow.get("entities") or [])[:8],
                "keywords": (follow.get("keywords") or [])[:8],
            }
        compact_trace.append(compact_step)
    final_hypothesis = candidate.get("final_hypothesis") if isinstance(candidate.get("final_hypothesis"), dict) else {}
    return {
        "question": candidate.get("question"),
        "answer": str(candidate.get("answer") or "")[:800],
        "intent": candidate.get("intent"),
        "final_hypothesis": {
            "query_type": final_hypothesis.get("query_type"),
            "entities": (final_hypothesis.get("entities") or [])[:10],
            "keywords": (final_hypothesis.get("keywords") or [])[:10],
        },
        "retrieval_trace": compact_trace,
    }


def compact_full_chain_evidence(evidence: Any) -> Any:
    if not isinstance(evidence, list):
        return evidence
    compact: list[dict[str, Any]] = []
    for item in evidence[:4]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "doc_id": item.get("doc_id") or item.get("id") or "",
                "story_name": item.get("story_name") or "",
                "stage_code": item.get("stage_code") or "",
                "chain_roles": item.get("evidence_chain_roles"),
                "text": (
                    str(item.get("clean_text") or item.get("evidence_chain_text") or "")
                    .replace("[CHAIN_LEN=", "[META_REMOVED=")[:500]
                ),
            }
        )
    return compact


def render_candidate_for_prompt(item: dict[str, Any]) -> str:
    candidate = item["candidate"]
    max_chars = 2600
    if item.get("task_type") == "full_chain_generation":
        candidate = compact_full_chain_candidate(candidate)
        return json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    return render_optional_json(candidate, max_chars=max_chars)


def render_evidence_for_prompt(item: dict[str, Any]) -> str:
    evidence = item["evidence"]
    max_chars = 3600
    if item.get("task_type") == "full_chain_generation":
        evidence = compact_full_chain_evidence(evidence)
        return json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    return render_optional_json(evidence, max_chars=max_chars)


def has_retrieval_metrics(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    text = str(value).strip()
    return bool(text and text != "无")


def build_score_prompt(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            "任务: 对一个 4B 学生模型候选输出做 OPD 评分。",
            "目标: 选择最适合继续训练当前 RAG 流程的候选；首要保护 R@1/R@5/MRR，其次才是 R@50。",
            "不要偏好写得漂亮但会伤害首轮精确检索、无故多轮检索或 grounding 的输出。",
            "",
            "任务级 schema:",
            "- user_question_hypothesis_generation: 必须输出 question,intent,query_type,entities,keywords,expected_answer_type,dialogue_context。",
            "- follow_up_hypothesis_generation: 必须输出 question,query_type,entities,keywords,expected_answer_type,dialogue_context；不需要 intent，输出 intent 不加分。",
            "- conclusion_generation: 必须输出 question,next_action,answer,missing_slots,clarification_question,follow_up_hypothesis。",
            "- full_chain_generation: candidate 是完整 runtime 链，包含 answer、final_hypothesis、retrieval_trace；评分重点是整条链是否值得作为 OPD 优化样本。",
            "- query_type 只能是 fact/relation/causality/reasoning/reveal/mystery/answerability。",
            "- intent 只能是 plot_fact/plot_reasoning/timeline/character_relation/event_summary/compare/persona_chat/out_of_scope。",
            "",
            f"task_type: {item['task_type']}",
            f"question: {item['question'] or '无'}",
            f"dialogue_context: {item['dialogue_context'] or '无'}",
            "student_prompt:",
            render_optional_json(item["prompt"], max_chars=2600),
            "candidate_output:",
            render_candidate_for_prompt(item),
            "retrieval_metrics:",
            render_optional_json(item["retrieval_metrics"], max_chars=1800),
            "evidence_or_evidence_brief:",
            render_evidence_for_prompt(item),
            "gold_or_reference:",
            render_optional_json(item["gold"], max_chars=1800),
            "",
            "输出字段严格为:",
            "candidate_id, task_type, accept, total_score, dimension_scores, hard_failures, reasons, suggested_fix",
            "",
            "dimension_scores 字段严格为:",
            "schema_validity, retrieval_gain, bridge_terms, grounding, decision_quality, anti_repetition, alias_hygiene, prompt_fit",
            "",
            "评分范围:",
            "- 每个维度 0-5 分，total_score 0-100 分。",
            "- accept 只有 true/false。",
            "",
            "关键评分规则:",
            "1. schema_validity: 输出必须是目标 task 对应的合法 JSON；字段缺失、字段多余、枚举非法、数组类型错误要扣分。完全无法解析 JSON 时 schema_validity=0 且 accept=false。",
            "2. retrieval_gain: follow_up_hypothesis 必须能提升真实召回方向；如果 retrieval_metrics 显示 rank 改善、miss->hit、hit 进入 top20/top10/top5，应高分。只新增泛词、把首轮可命中的问题拖到后轮、或没有实际增益应低分。",
            "2.1 follow_up_hypothesis 没有 retrieval_metrics 时，不能确认真实召回增益，retrieval_gain 最高只能 2，accept 必须为 false。",
            "2.2 如果 retrieval_metrics 显示 follow-up 后 rank 没变差但仍未命中，最多给 2；只有真实 rank 改善或 miss->hit 才能给 3 分以上。",
            "2.3 full_chain_generation 使用 final_hit/hit_at_k/rounds_run/final_action 判断检索效果；优先奖励 final_action=answer_directly 且 final_hit 进入 top20/top10/top5 的链。",
            "2.4 full_chain_generation 如果 final_hit 已进入 top5/top10 却继续多轮 retrieve_more 后 abstain，应视为 over_abstain/unnecessary_followup，不应 accept。",
            "3. bridge_terms: entities/keywords/missing_slots 应包含具体桥接词，如人物、自然称谓、组织、地点、章节、事件、物品、关键台词。禁止只写 原因/背景/真相/线索/关系/影响/情况/剧情。",
            "4. grounding: conclusion 的 answer 只能基于 evidence_or_evidence_brief；证据不足时应 retrieve_more 或 abstain。不得把证据外剧情、图谱关系、百科背景当成确定答案。",
            "4.1 full_chain_generation 的最终 answer 也必须由 evidence_or_evidence_brief 或 retrieval_trace 中的证据支撑；不得输出 [CHAIN_LEN]、[EVIDENCE_TYPES]、[E1] 等内部证据链元标签作为自然答案。",
            "5. decision_quality: conclusion 要正确判断 answer_directly/retrieve_more/abstain。证据充分时必须 answer_directly；证据只部分命中时才 retrieve_more；最后一轮仍不足才 abstain。",
            "5.1 full_chain_generation 要评价整条 planner 行为：首轮证据足够时应直接回答；只有缺少关键桥接实体/事件/时间线时才继续检索；最终命中证据后仍 abstain 是硬失败。",
            "6. anti_repetition: 严重复读、关键词刷屏、同一实体反复出现、输出被截断、生成长 alias 串，必须低分。",
            "7. alias_hygiene: 不要输出内部 alias、异常别名串、图谱关系串。凯尔希相关自然称谓可以是 凯尔希/凯尔希医生/凯尔希所长；不要强制输出 凯尔希·思衡托。",
            "8. prompt_fit: 候选必须贴合短 prompt 和当前 runtime 输入，不依赖 teacher-only 信息，不凭空引用未给出的 MiniRAG 图谱线索。",
            "",
            "硬失败规则:",
            "- invalid_json: 非 JSON 或无法解析。",
            "- schema_error: 字段/枚举/类型不符合任务。",
            "- hallucinated_answer: answer 明显超出证据。",
            "- no_retrieval_gain: follow-up 没有新增有效检索方向，或 metrics 显示无增益。",
            "- keyword_repetition: 关键词/实体明显复读刷屏。",
            "- alias_pollution: 内部别名或异常 alias 串污染。",
            "- wrong_action: conclusion 动作与证据状态相反。",
            "- bad_full_chain: full_chain 检索未命中且仍给确定答案，或链路动作明显错误。",
            "- over_abstain: 已命中可回答证据，尤其 final_hit 在 top20/top10/top5 内，但最终 abstain。",
            "- unnecessary_followup: 首轮或当前证据已经足够回答，却继续 retrieve_more 或生成泛化 follow-up。",
            "- evidence_tag_leak: 最终 answer 泄漏内部证据链标签/元信息。",
            "",
            "accept 判断:",
            "- total_score >= 75 且 hard_failures 为空，accept=true。",
            "- 如果是 follow_up_hypothesis，必须 retrieval_gain>=3 且 bridge_terms>=3 才能 accept=true。",
            "- 如果是 conclusion_generation，必须 grounding>=3 且 decision_quality>=3 才能 accept=true。",
            "- 如果是 user_question_hypothesis_generation，必须 schema_validity>=4、bridge_terms>=3、anti_repetition>=4 才能 accept=true。",
            "- 如果是 full_chain_generation，必须 final_action=answer_directly、final_hit 进入 top20 或更高、retrieval_gain>=3、grounding>=3、decision_quality>=3、anti_repetition>=4 才能 accept=true。",
            "- full_chain_generation 中 hit_abstain 不作为正例，即使答案保守也不能 accept；miss_answer 必须 reject。",
            "",
            f"candidate_id: {item['candidate_id']}",
        ]
    )


def normalize_score_payload(payload: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    dims = payload.get("dimension_scores")
    if not isinstance(dims, dict):
        dims = {}
    required_dims = [
        "schema_validity",
        "retrieval_gain",
        "bridge_terms",
        "grounding",
        "decision_quality",
        "anti_repetition",
        "alias_hygiene",
        "prompt_fit",
    ]
    norm_dims: dict[str, int] = {}
    for key in required_dims:
        value = dims.get(key, 0)
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            score = 0
        norm_dims[key] = max(0, min(5, score))
    hard_failures = payload.get("hard_failures")
    if not isinstance(hard_failures, list):
        hard_failures = []
    hard_failures = [str(item).strip() for item in hard_failures if str(item).strip()]
    reasons = payload.get("reasons")
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []
    reasons = [str(item).strip() for item in reasons if str(item).strip()][:8]
    try:
        total_score = int(round(float(payload.get("total_score", 0))))
    except (TypeError, ValueError):
        total_score = int(round(sum(norm_dims.values()) / (len(norm_dims) * 5) * 100))
    total_score = max(0, min(100, total_score))
    accept = bool(payload.get("accept")) and total_score >= 75 and not hard_failures
    task_type = str(payload.get("task_type") or item["task_type"])
    if task_type == "follow_up_hypothesis_generation":
        accept = accept and norm_dims["retrieval_gain"] >= 3 and norm_dims["bridge_terms"] >= 3
        if not has_retrieval_metrics(item.get("retrieval_metrics")):
            norm_dims["retrieval_gain"] = min(norm_dims["retrieval_gain"], 2)
            accept = False
            if "missing_retrieval_metrics" not in hard_failures:
                hard_failures.append("missing_retrieval_metrics")
    elif task_type == "conclusion_generation":
        accept = accept and norm_dims["grounding"] >= 3 and norm_dims["decision_quality"] >= 3
        candidate_text = str(item.get("candidate") or "")
        if "follow_up_hypothesis" in candidate_text and not has_retrieval_metrics(item.get("retrieval_metrics")):
            accept = False
            if "missing_retrieval_metrics" not in hard_failures:
                hard_failures.append("missing_retrieval_metrics")
    elif task_type == "user_question_hypothesis_generation":
        accept = accept and norm_dims["schema_validity"] >= 4 and norm_dims["bridge_terms"] >= 3 and norm_dims["anti_repetition"] >= 4
    elif task_type == "full_chain_generation":
        metrics = item.get("retrieval_metrics") if isinstance(item.get("retrieval_metrics"), dict) else {}
        candidate_text = str(item.get("candidate") or "")
        has_final_hit = bool(metrics.get("final_hit"))
        answered = bool(metrics.get("answered"))
        final_action = str(metrics.get("final_action") or "")
        try:
            final_rank = int(metrics.get("final_rank_value") or 10**9)
        except (TypeError, ValueError):
            final_rank = 10**9
        if not has_retrieval_metrics(metrics):
            norm_dims["retrieval_gain"] = min(norm_dims["retrieval_gain"], 2)
            accept = False
            if "missing_retrieval_metrics" not in hard_failures:
                hard_failures.append("missing_retrieval_metrics")
        if not has_final_hit:
            norm_dims["retrieval_gain"] = min(norm_dims["retrieval_gain"], 2)
            if answered and final_action != "abstain":
                accept = False
                if "bad_full_chain" not in hard_failures:
                    hard_failures.append("bad_full_chain")
        if has_final_hit and final_action == "abstain":
            accept = False
            norm_dims["decision_quality"] = min(norm_dims["decision_quality"], 2)
            if "over_abstain" not in hard_failures:
                hard_failures.append("over_abstain")
        if has_final_hit and final_action == "retrieve_more":
            accept = False
            norm_dims["decision_quality"] = min(norm_dims["decision_quality"], 2)
            if "unnecessary_followup" not in hard_failures:
                hard_failures.append("unnecessary_followup")
        if has_final_hit and final_action == "answer_directly" and final_rank > 20:
            accept = False
            norm_dims["retrieval_gain"] = min(norm_dims["retrieval_gain"], 2)
            if "low_rank_hit" not in hard_failures:
                hard_failures.append("low_rank_hit")
        if any(tag in candidate_text for tag in ("[CHAIN_LEN=", "[EVIDENCE_TYPES=", "[CAUSAL_ORDER=", "[E1]")):
            accept = False
            norm_dims["grounding"] = min(norm_dims["grounding"], 2)
            if "evidence_tag_leak" not in hard_failures:
                hard_failures.append("evidence_tag_leak")
        accept = (
            accept
            and has_final_hit
            and final_action == "answer_directly"
            and final_rank <= 20
            and norm_dims["retrieval_gain"] >= 3
            and norm_dims["grounding"] >= 3
            and norm_dims["decision_quality"] >= 3
            and norm_dims["anti_repetition"] >= 4
        )
    return {
        "candidate_id": str(payload.get("candidate_id") or item["candidate_id"]),
        "source_index": item["source_index"],
        "task_type": task_type,
        "accept": accept,
        "total_score": total_score,
        "dimension_scores": norm_dims,
        "hard_failures": hard_failures,
        "reasons": reasons,
        "suggested_fix": str(payload.get("suggested_fix") or "").strip(),
    }


def load_teacher_api_config(args: argparse.Namespace) -> TeacherApiConfig:
    teacher_cfg: dict[str, Any] = {}
    if args.teacher_config:
        loaded_cfg = load_json(resolve_path(args.teacher_config))
        teacher_cfg = loaded_cfg.get("teacher_api", loaded_cfg) if isinstance(loaded_cfg, dict) else {}
    api_type = str(args.api_type or teacher_cfg.get("api_type") or "chat_completions")
    api_base = str(args.api_base or teacher_cfg.get("base_url") or "")
    api_key_env = str(args.api_key_env or teacher_cfg.get("api_key_env") or "TEACHER_API_KEY")
    auth_header = str(args.auth_header or teacher_cfg.get("auth_header") or "bearer")
    model = str(args.model or teacher_cfg.get("model") or "")
    timeout = int(args.timeout if args.timeout is not None else teacher_cfg.get("timeout_seconds") or 300)
    temperature = float(args.temperature if args.temperature is not None else 0.0)
    max_output_tokens = int(args.max_output_tokens if args.max_output_tokens is not None else 2048)
    json_mode = bool(teacher_cfg.get("json_mode", True)) and not args.no_json_mode
    extra_headers = teacher_cfg.get("extra_headers") if isinstance(teacher_cfg.get("extra_headers"), dict) else None
    if not api_base or not model:
        raise SystemExit("--api-base and --model are required.")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise SystemExit(
            f"{api_key_env} contains non-latin-1 characters. "
            "Set it to the real API key, not a Chinese placeholder such as '你的key'."
        ) from exc
    return TeacherApiConfig(
        api_type=api_type,
        base_url=api_base,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        extra_headers=extra_headers,
        auth_header=auth_header,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OPD candidate outputs with an API teacher for 4B RAG optimization.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL candidates to score.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--teacher-config", type=Path, default=None)
    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "responses"), default=None)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--auth-header", choices=("bearer", "x-api-key", "both"), default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument("--no-json-mode", action="store_true")
    return parser.parse_args()


def score_one(
    item: dict[str, Any],
    *,
    api_config: TeacherApiConfig,
    raw_output: Path,
    api_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    payload = call_teacher_json(
        api_config,
        system_prompt=build_teacher_system(),
        user_prompt=build_score_prompt(item),
        retries=api_retries,
        retry_sleep=retry_sleep,
        raw_output=raw_output,
        request_meta={
            "question_key": item["candidate_id"],
            "candidate_id": item["candidate_id"],
            "task_type": "opd_teacher_score",
            "source_task_type": item["task_type"],
            "round": 0,
        },
    )
    score = normalize_score_payload(payload, item)
    return {
        **score,
        "question": item["question"],
        "candidate_preview": render_optional_json(item["candidate"], max_chars=500),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.jsonl"
    raw_output = output_dir / "raw_teacher_scores.jsonl"
    failed_path = output_dir / "failed_scores.jsonl"
    if not input_path.exists():
        raise SystemExit(f"Input candidates file does not exist: {input_path}")
    records = load_jsonl(input_path)
    if not records:
        raise SystemExit(f"Input candidates file is empty: {input_path}")
    if args.limit is not None:
        records = records[: args.limit]
    items = [extract_candidate_record(record, index) for index, record in enumerate(records)]
    done = {
        str(record.get("candidate_id") or "")
        for record in load_jsonl(scores_path)
    } if args.resume else set()
    if done:
        items = [item for item in items if item["candidate_id"] not in done]
    log(f"[items] pending={len(items)} resume_done={len(done)} input={input_path}")
    api_config = load_teacher_api_config(args)
    stats: Counter[str] = Counter()

    def run(item: dict[str, Any]) -> dict[str, Any]:
        return score_one(
            item,
            api_config=api_config,
            raw_output=raw_output,
            api_retries=args.api_retries,
            retry_sleep=args.retry_sleep,
        )

    parallel = max(1, args.parallel)
    if parallel == 1:
        iterator = tqdm(items, desc="opd teacher scoring", unit="candidate")
        for item in iterator:
            try:
                score = run(item)
                with WRITE_LOCK:
                    append_jsonl(scores_path, score)
                stats["completed"] += 1
                stats[f"task:{score['task_type']}"] += 1
                stats[f"accept:{score['accept']}"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                with WRITE_LOCK:
                    append_jsonl(failed_path, {"candidate_id": item["candidate_id"], "error": str(exc), "item": item, "created_at": int(time.time())})
            iterator.set_postfix({"ok": stats["completed"], "failed": stats["failed"]})
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_item = {executor.submit(run, item): item for item in items}
            progress = tqdm(as_completed(future_to_item), total=len(future_to_item), desc="opd teacher scoring", unit="candidate")
            for future in progress:
                item = future_to_item[future]
                try:
                    score = future.result()
                    with WRITE_LOCK:
                        append_jsonl(scores_path, score)
                    stats["completed"] += 1
                    stats[f"task:{score['task_type']}"] += 1
                    stats[f"accept:{score['accept']}"] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    with WRITE_LOCK:
                        append_jsonl(failed_path, {"candidate_id": item["candidate_id"], "error": str(exc), "item": item, "created_at": int(time.time())})
                progress.set_postfix({"ok": stats["completed"], "failed": stats["failed"]})

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "pending_items": len(items),
        "resume_done": len(done),
        "stats": dict(stats),
    }
    all_scores = load_jsonl(scores_path)
    if all_scores:
        summary["total_scores"] = len(all_scores)
        summary["accepted_scores"] = sum(1 for item in all_scores if item.get("accept"))
        summary["mean_total_score"] = round(sum(float(item.get("total_score") or 0) for item in all_scores) / len(all_scores), 2)
        summary["hard_failures"] = dict(Counter(failure for item in all_scores for failure in item.get("hard_failures", [])))
    write_json(output_dir / "score_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
