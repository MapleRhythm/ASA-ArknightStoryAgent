#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    executable = Path(sys.executable).as_posix().lower()
    return conda_env == "train" or "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if should_use_train_overrides():
    if TRAIN_PYTHON_OVERLAY_DIR.exists():
        sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
    if TRAIN_OVERRIDE_DIR.exists():
        sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    MINIRAG_GRAPH_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)
from goldenglow.data.sft_teacher import TeacherApiConfig, call_teacher_api  # noqa: E402
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    HypothesisDocument,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    _resolve_referential_question,
    build_follow_up_hypothesis_queries,
    build_hypothesis,
    build_retrieval_query,
    extract_json_object,
    merge_hypotheses,
    normalize_conclusion_payload,
    normalize_hypothesis_payload,
    summarize_evidence_for_trace,
)
from goldenglow.inference import CPUInferencePipeline  # noqa: E402
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402


DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "configs" / "runtime_inference_gpu.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "llama_factory" / "api_grounded_sft_deepseek_v1"
STUDENT_SYSTEM = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
CONCLUSION_FIELDS = {
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
}
DEFAULT_COMPLEX_QUESTION_THEMES = (
    "大炎、岁兽、真龙、不反、岁陵危机、权柄代价",
    "阴谋识破、真相曝光、主使与掩盖手段",
    "政治斗争、夺权、继承、合法性与牺牲",
    "源石、感染者、城市工业事故、舆论栽赃",
    "人物关系变化、同盟破裂、误解消除、立场转变",
    "灾难危机的成因、爆发、解决方案与代价",
    "国家或移动城市的历史时间线、旧事真相、遗留问题",
    "行动目的、计划失败、反转结果、幕后动机",
    "多方视角中的同一事件：误会、隐瞒、真实目的",
    "非主角人物在支线活动中的关键选择和后果",
)
ANSWER_INSUFFICIENT_MARKERS = (
    "未出现",
    "未提及",
    "无法确认",
    "无法回答",
    "缺少直接证据",
    "证据不足",
    "没有提到",
    "不涉及",
    "错误前提",
    "无法据此",
    "未直接",
    "尚未在证据中",
)
QUESTION_RISK_MARKERS = (
    "事先知道",
    "真正策划者",
    "皇室阴谋",
    "外部协议",
    "背叛内容",
    "血祭",
    "不合法",
    "秘密调动",
    "封锁消息",
    "决裂",
    "净化海嗣",
    "克莱布拉松",
    "A6小组",
    "贫民窟源石加工事故",
)
SOURCE_NAME_MARKERS = (
    "第八章",
    "第十章",
    "第十一章",
    "怒号光明",
    "未尽篇章",
    "主线",
)


class RetrievalOnlyGenerator:
    backend_name = "retrieval-only"

    def describe_runtime(self) -> dict[str, Any]:
        return {"generator_backend": self.backend_name}

    def generate(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("RetrievalOnlyGenerator must not be used for local generation.")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def stable_key(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def truncate_text(text: str, max_chars: int | None) -> str:
    normalized = str(text or "").strip()
    if max_chars is None or max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    visible_chars = max(0, max_chars - 32)
    omitted = max(0, len(normalized) - visible_chars)
    return normalized[:visible_chars].rstrip() + f"\n...[truncated {omitted} chars]"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_path(value: str | Path | None, default: Path | None = None) -> Path | None:
    raw = value if value not in (None, "") else default
    if raw in (None, ""):
        return None
    path = raw if isinstance(raw, Path) else Path(str(raw))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_value(cli_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def load_question_items(
    path: Path | None,
    inline_questions: list[str],
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if path is not None:
        if not path.exists():
            raise SystemExit(f"questions file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("questions") or payload.get("items") or payload.get("data") or []
            if not isinstance(payload, list):
                raise SystemExit(f"JSON question file must contain a list: {path}")
            for item in payload:
                items.append(normalize_question_item(item))
        elif path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    items.append(normalize_question_item(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        else:
            for line in text.splitlines():
                question = line.strip()
                if question:
                    items.append(normalize_question_item(question))
    for question in inline_questions:
        if question.strip():
            items.append(normalize_question_item(question.strip()))
    if not items and not allow_empty:
        raise SystemExit("No questions provided. Use --questions-file or --question.")
    return items


def normalize_question_item(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        question = value.strip()
        item: dict[str, Any] = {"question": question}
    elif isinstance(value, dict):
        item = dict(value)
        question = str(item.get("question") or item.get("query") or "").strip()
        item["question"] = question
    else:
        raise ValueError(f"Unsupported question item: {type(value).__name__}")
    if not question:
        raise ValueError("Question item has empty question")
    dialogue_context = str(item.get("dialogue_context") or item.get("context") or "").strip()
    item["dialogue_context"] = dialogue_context
    item["question_key"] = str(item.get("question_key") or item.get("id") or stable_key(question, dialogue_context))
    return item


def coerce_hypothesis(item: dict[str, Any]) -> HypothesisDocument:
    question = str(item["question"])
    dialogue_context = str(item.get("dialogue_context") or "")
    base = build_hypothesis(question, dialogue_context)
    provided = item.get("hypothesis")
    if provided is None:
        provided = {
            key: item[key]
            for key in ("intent", "query_type", "entities", "keywords", "expected_answer_type")
            if key in item
        }
    if not isinstance(provided, dict) or not provided:
        return base

    payload = asdict(base)
    for key in ("intent", "query_type", "entities", "keywords", "expected_answer_type"):
        if key in provided:
            payload[key] = provided[key]
    payload["question"] = question
    payload["dialogue_context"] = dialogue_context
    try:
        return normalize_hypothesis_payload(payload, question=question, dialogue_context=dialogue_context)
    except Exception as exc:
        log(f"[warn] invalid provided hypothesis; fallback=heuristic key={item['question_key']} error={exc}")
        return base


def doc_is_question_seed(doc: dict[str, Any]) -> bool:
    text = str(doc.get("clean_text") or "").strip()
    source_path = str(doc.get("source_path") or "")
    if len(text) < 160:
        return False
    if "[uc]info" in source_path:
        return False
    if "charword_table" in source_path or "handbook_info_table" in source_path:
        return False
    if any(marker in text for marker in ("<@ba.kw>", "获得干员", "晋升阶段", "信赖提升")):
        return False
    return True


def sample_seed_doc_groups(
    documents: list[dict[str, Any]],
    *,
    target_groups: int,
    group_size: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        if not doc_is_question_seed(doc):
            continue
        key = str(doc.get("source_path") or doc.get("story_id") or doc.get("activity_id") or "")
        if key:
            grouped[key].append(doc)
    candidate_groups: list[list[dict[str, Any]]] = []
    for docs in grouped.values():
        ordered = sorted(docs, key=lambda item: str(item.get("id") or ""))
        if len(ordered) >= group_size:
            for start in range(0, len(ordered), group_size):
                group = ordered[start : start + group_size]
                if len(group) >= max(1, group_size // 2):
                    candidate_groups.append(group)
        elif ordered:
            candidate_groups.append(ordered)
    rng = random.Random(seed)
    rng.shuffle(candidate_groups)
    return candidate_groups[:target_groups]


def render_seed_docs_for_question_generation(docs: list[dict[str, Any]], *, max_chars_per_doc: int) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[剧情片段 {index}]",
                    f"id: {doc.get('id') or ''}",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or doc.get('story_id') or ''}",
                    f"stage_code: {doc.get('stage_code') or doc.get('stage_id') or ''}",
                    f"source_path: {doc.get('source_path') or ''}",
                    "clean_text:",
                    truncate_text(str(doc.get("clean_text") or ""), max_chars_per_doc),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_question_generation_prompt(
    docs: list[dict[str, Any]],
    *,
    count: int,
    max_chars_per_doc: int,
) -> str:
    return "\n".join(
        [
            f"任务: 基于下面同一剧情段落生成 {count} 个真实用户可能会问的《明日方舟》剧情问题。",
            "",
            "输出必须是单个 JSON 对象，字段只有 questions。",
            "questions 是数组，每个元素字段为 question,hypothesis。",
            "hypothesis 字段包含 intent,query_type,entities,keywords,expected_answer_type。",
            "",
            "出题规则:",
            "1. 问题必须能由给定剧情片段或同一章节相邻文本回答，不要凭空引入其他作品或未出现实体。",
            "2. 问题不要提到“片段/证据/chunk/上文/文本中”。",
            "3. 优先生成需要结论归纳的问题：为什么、真相是什么、识破了什么、某场危机是什么、某行动目的是什么。",
            "4. 避免太简单的台词复读题；避免干员语音、信赖触摸、档案数值问题。",
            "5. question 不超过 60 个汉字，必须包含明确人物/组织/地点/事件锚点。",
            "6. intent 只能是 plot_fact,plot_reasoning,timeline,character_relation,event_summary,compare。",
            "7. query_type 只能是 fact,relation,causality,reasoning,reveal,mystery,answerability。",
            "8. entities 写 2-6 个明确实体；keywords 写 4-10 个自然检索词。",
            "9. 不要输出答案。",
            "",
            "剧情片段:",
            render_seed_docs_for_question_generation(docs, max_chars_per_doc=max_chars_per_doc),
        ]
    )


def parse_generated_questions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        return []
    items: list[dict[str, Any]] = []
    for raw in raw_questions:
        if isinstance(raw, str):
            candidate = {"question": raw}
        elif isinstance(raw, dict):
            candidate = dict(raw)
        else:
            continue
        question = str(candidate.get("question") or "").strip()
        if len(question) < 4:
            continue
        item = normalize_question_item(candidate)
        if isinstance(candidate.get("hypothesis"), dict):
            item["hypothesis"] = candidate["hypothesis"]
        items.append(item)
    return items


def load_question_themes(path: Path | None, inline_themes: list[str]) -> list[str]:
    themes: list[str] = []
    if path is not None:
        if not path.exists():
            raise SystemExit(f"question themes file not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("themes") or payload.get("items") or []
            if not isinstance(payload, list):
                raise SystemExit(f"theme JSON must contain a list: {path}")
            themes.extend(str(item).strip() for item in payload if str(item).strip())
        else:
            themes.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    themes.extend(theme.strip() for theme in inline_themes if theme.strip())
    if not themes:
        themes = list(DEFAULT_COMPLEX_QUESTION_THEMES)
    seen: set[str] = set()
    output: list[str] = []
    for theme in themes:
        if theme not in seen:
            seen.add(theme)
            output.append(theme)
    return output


def build_theme_question_generation_prompt(*, theme: str, count: int) -> str:
    return "\n".join(
        [
            f"任务: 围绕一个《明日方舟》剧情主题生成 {count} 个复杂但可检索的用户问题。",
            f"主题: {theme}",
            "",
            "输出必须是单个 JSON 对象，字段只有 questions。",
            "questions 是数组，每个元素字段为 question,hypothesis,difficulty,why_complex。",
            "hypothesis 字段包含 intent,query_type,entities,keywords,expected_answer_type。",
            "",
            "复杂度要求:",
            "1. 问题必须是剧情理解题，优先覆盖原因、动机、真相、危机本质、行动代价、关系变化、时间线串联。",
            "2. 每个问题应至少需要 2 个事实点才能回答，不能被单句台词或单个名词直接回答。",
            "3. 只能围绕你确信官方剧情中真实存在的事件、人物和行动出题；不确定是否存在的情节不要写。",
            "4. 不要编造具体前提，例如不存在的合作、背叛协议、血祭仪式、贫民窟事故、某人救某孩子等。",
            "5. 允许使用你已有的《明日方舟》剧情知识出题，但问题必须能在官方剧情文本中召回证据。",
            "6. 不要输出冷门到几乎无法检索的同人、猜测、二创或纯设定脑补问题。",
            "7. 不要生成“X是谁”“X是什么职业”“某关卡发生了什么”这种简单模板。",
            "8. 不要生成需要数值、卡池、强度、语音、档案字段才能回答的问题。",
            "9. question 必须像真实用户提问，不要提到“证据/检索/文本/官方原文/chunk”。",
            "10. question 不超过 70 个汉字，必须包含明确锚点实体。",
            "11. 同一批问题不要只换一个名字套模板。",
            "12. 优先选择较稳定、官方文本中多次出现的锚点，例如真龙/不反/岁陵、贝希曼/卡拉顿、科西切/塔露拉、风雪过境改革、沃伦姆德、愚人号、覆潮之下、叙拉古人、莱茵生命。",
            "",
            "hypothesis 要求:",
            "1. intent 只能是 plot_fact,plot_reasoning,timeline,character_relation,event_summary,compare。",
            "2. query_type 只能是 fact,relation,causality,reasoning,reveal,mystery,answerability。",
            "3. entities 写 2-8 个明确实体或事件名。",
            "4. keywords 写 6-12 个自然检索词，包含别名、地点、行动、结果或关键术语。",
            "5. expected_answer_type 写具体答案类型，例如“事件真相/阴谋内容”“危机成因与处理方式”“行动动机和代价”。",
            "6. 不要输出答案。",
        ]
    )


def generate_theme_question_items_with_api(
    *,
    api_config: TeacherApiConfig,
    output_dir: Path,
    target_count: int,
    seed: int,
    themes: list[str],
    theme_requests: int,
    questions_per_theme: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    raw_path = output_dir / "generated_theme_question_raw_responses.jsonl"
    questions_path = output_dir / "generated_theme_questions.jsonl"
    rng = random.Random(seed)
    shuffled_themes = list(themes)
    rng.shuffle(shuffled_themes)
    request_count = max(1, theme_requests)
    while len(shuffled_themes) < request_count:
        shuffled_themes.extend(themes)
    selected_themes = shuffled_themes[:request_count]
    system_prompt = "你是《明日方舟》剧情检索训练数据出题教师，擅长生成复杂但可由官方剧情召回证据回答的问题。只输出 JSON。"
    output: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for request_index, theme in enumerate(selected_themes, start=1):
        if len(output) >= target_count:
            break
        user_prompt = build_theme_question_generation_prompt(
            theme=theme,
            count=max(1, questions_per_theme),
        )
        raw_text, raw_response = call_teacher_api(
            api_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        append_jsonl(
            raw_path,
            {
                "request_index": request_index,
                "theme": theme,
                "raw_text": raw_text,
                "raw_response": raw_response,
            },
        )
        payload = extract_json_object(raw_text) or {}
        for item in parse_generated_questions(payload):
            question = re.sub(r"\s+", "", str(item.get("question") or ""))
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            item["question_key"] = str(item.get("question_key") or stable_key(item["question"], theme))
            item["source"] = "api_theme_question_generation"
            item["question_theme"] = theme
            output.append(item)
            append_jsonl(questions_path, item)
            if len(output) >= target_count:
                break
        log(
            f"[theme-question-gen] request={request_index}/{request_count} "
            f"theme={theme} questions={len(output)}/{target_count}"
        )
    if len(output) < target_count:
        log(f"[warn] generated only {len(output)} theme questions, target={target_count}")
    return output[:target_count]


def generate_question_items_with_api(
    *,
    api_config: TeacherApiConfig,
    documents: list[dict[str, Any]],
    output_dir: Path,
    target_count: int,
    seed: int,
    docs_per_group: int,
    questions_per_group: int,
    max_chars_per_doc: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    raw_path = output_dir / "generated_question_raw_responses.jsonl"
    questions_path = output_dir / "generated_questions.jsonl"
    groups = sample_seed_doc_groups(
        documents,
        target_groups=max(target_count * 2, 16),
        group_size=max(1, docs_per_group),
        seed=seed,
    )
    system_prompt = "你是《明日方舟》剧情检索训练数据出题教师。只输出 JSON。"
    output: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for group_index, docs in enumerate(groups, start=1):
        if len(output) >= target_count:
            break
        user_prompt = build_question_generation_prompt(
            docs,
            count=max(1, questions_per_group),
            max_chars_per_doc=max_chars_per_doc,
        )
        raw_text, raw_response = call_teacher_api(
            api_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        append_jsonl(
            raw_path,
            {
                "group_index": group_index,
                "seed_doc_ids": [doc.get("id") for doc in docs],
                "raw_text": raw_text,
                "raw_response": raw_response,
            },
        )
        payload = extract_json_object(raw_text) or {}
        for item in parse_generated_questions(payload):
            question = re.sub(r"\s+", "", str(item.get("question") or ""))
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            item["question_key"] = str(item.get("question_key") or stable_key(item["question"], "api-generated-question"))
            item["source"] = "api_question_generation"
            output.append(item)
            append_jsonl(questions_path, item)
            if len(output) >= target_count:
                break
        log(f"[question-gen] group={group_index}/{len(groups)} questions={len(output)}/{target_count}")
    if len(output) < target_count:
        log(f"[warn] generated only {len(output)} questions, target={target_count}")
    return output[:target_count]


def build_initial_queries(question: str, hypothesis: HypothesisDocument) -> list[str]:
    queries = [
        _resolve_referential_question(question, hypothesis.entities),
        build_retrieval_query(hypothesis),
    ]
    queries.extend(build_follow_up_hypothesis_queries(question, hypothesis))
    seen: set[str] = set()
    output: list[str] = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def validate_record_quality(
    *,
    record: dict[str, Any],
    assistant_payload: dict[str, Any],
    question_item: dict[str, Any],
    request_payload: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    question = str(question_item.get("question") or "")
    answer = str(assistant_payload.get("answer") or "")
    if assistant_payload.get("next_action") != "answer_directly":
        reasons.append(f"action:{assistant_payload.get('next_action')}")
    if not answer.strip():
        reasons.append("empty_answer")
    if any(marker in answer for marker in ANSWER_INSUFFICIENT_MARKERS):
        reasons.append("answer_insufficient_marker")
    if any(marker in question for marker in QUESTION_RISK_MARKERS):
        reasons.append("question_risk_marker")
    evidence_prompt = str(request_payload.get("user_prompt") or "")
    if any(marker in question for marker in SOURCE_NAME_MARKERS) and not any(
        marker in evidence_prompt for marker in SOURCE_NAME_MARKERS
    ):
        reasons.append("source_name_not_in_evidence")
    hypothesis = question_item.get("hypothesis") if isinstance(question_item.get("hypothesis"), dict) else {}
    missing_entities = []
    for entity in hypothesis.get("entities") or []:
        entity = str(entity).strip()
        if len(entity) >= 2 and entity not in evidence_prompt:
            missing_entities.append(entity)
    if len(missing_entities) >= 2:
        reasons.append("many_entities_missing_in_evidence")
    prompt = ((record.get("conversations") or [{}])[0] or {}).get("value", "")
    if len(str(prompt)) < 1000:
        reasons.append("prompt_too_short")
    return not reasons, reasons


def make_pipeline(
    retriever: ArknightsHybridRetriever,
    query_config: QueryConfig,
    inference_cfg: dict[str, Any],
) -> CPUInferencePipeline:
    return CPUInferencePipeline(
        retriever=retriever,
        generator=RetrievalOnlyGenerator(),  # type: ignore[arg-type]
        query_config=query_config,
        max_retrieval_rounds=1,
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(inference_cfg.get("prompt_evidence_max_chars_per_doc", 1800)),
        prompt_conclusion_evidence_max_total_chars=int(
            inference_cfg.get("prompt_conclusion_evidence_max_total_chars", 24000)
        ),
        enable_mmr=False,
        enable_pyramid_order=False,
        enable_crag_refinement=False,
        self_consistency_samples=1,
        answer_grounding_mode=str(inference_cfg.get("answer_grounding_mode", "weak")),
        web_context_config={"enabled": False},
    )


def retrieve_evidence(
    pipeline: CPUInferencePipeline,
    *,
    question: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    expansion_record: dict[str, Any] | None = None
    if pipeline.query_config.minirag_chapter_isolation and pipeline.query_config.minirag_auto_second_retrieval:
        dense_hits, sparse_hits, evidence, expansion_record = pipeline._retrieve_first_round_with_scoped_minirag_expansion(  # noqa: SLF001
            question,
            hypothesis,
            queries,
        )
    else:
        dense_hits, sparse_hits, evidence = pipeline._retrieve_round(question, hypothesis, queries)  # noqa: SLF001
    trace = {
        "round": 1,
        "queries": queries,
        "hypothesis": asdict(hypothesis),
        "dense_hit_count": len(dense_hits),
        "sparse_hit_count": len(sparse_hits),
        "evidence_count": len(evidence),
        "evidence_summary": summarize_evidence_for_trace(evidence, limit=8),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    if expansion_record is not None:
        trace["minirag_chapter_expansion"] = expansion_record
    return evidence, trace


def render_raw_evidence_for_teacher(
    evidence: list[dict[str, Any]],
    *,
    top_k: int,
    max_chars_per_doc: int,
    max_total_chars: int,
) -> str:
    blocks: list[str] = []
    total_chars = 0
    seen_doc_ids: set[str] = set()
    for index, item in enumerate(evidence[: max(1, top_k)], start=1):
        doc = item.get("document") or {}
        doc_id = str(doc.get("id") or "")
        if doc_id and doc_id in seen_doc_ids:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        clean_text = truncate_text(str(doc.get("clean_text") or doc.get("search_text") or ""), max_chars_per_doc)
        block = "\n".join(
            [
                f"[证据 {index}]",
                f"id: {doc_id}",
                f"activity_name: {doc.get('activity_name') or ''}",
                f"story_name: {doc.get('story_name') or doc.get('story_id') or ''}",
                f"stage_code: {doc.get('stage_code') or doc.get('stage_id') or ''}",
                f"avg_tag: {doc.get('avg_tag') or ''}",
                f"source_path: {doc.get('source_path') or ''}",
                f"rerank_score: {item.get('rerank_score') if item.get('rerank_score') is not None else ''}",
                f"fusion_score: {item.get('fusion_score') if item.get('fusion_score') is not None else ''}",
                f"chain_roles: {','.join(item.get('evidence_chain_roles') or [])}",
                "clean_text:",
                clean_text,
            ]
        )
        if max_total_chars > 0 and total_chars + len(block) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 200:
                blocks.append(truncate_text(block, remaining))
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks)


def build_teacher_system_prompt() -> str:
    return "\n".join(
        [
            "你是《明日方舟》剧情 RAG 数据生成教师。",
            "你会收到本地检索流程返回的原始剧情证据块，并为学生模型生成 conclusion_generation 的标准 JSON 输出。",
            "只输出单个 JSON 对象，不要输出 markdown，不要解释你的思考过程。",
        ]
    )


def build_teacher_user_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
    evidence_text: str,
    evidence_count: int,
    trace: dict[str, Any],
    allow_retrieve_more: bool,
    round_index: int,
    max_rounds: int,
) -> str:
    scope_info = (trace.get("minirag_chapter_expansion") or {}) if isinstance(trace, dict) else {}
    scope_lines = []
    if scope_info:
        scope_lines = [
            f"chapter_scope: {scope_info.get('chapter_scope') or ''}",
            f"chapter_scope_label: {scope_info.get('chapter_scope_label') or ''}",
            f"scope_dominance_ratio: {scope_info.get('scope_dominance_ratio') or ''}",
            f"graph_hit_count: {scope_info.get('graph_hit_count') or 0}",
            "second_pass_queries:",
            json.dumps(scope_info.get("second_pass_queries") or [], ensure_ascii=False),
        ]
    if allow_retrieve_more:
        retrieve_more_rules = [
            "10. 证据无法定位核心实体或缺关键桥接时可以 retrieve_more。",
            "11. retrieve_more 时 answer 必须为空字符串，missing_slots 写 2-5 个具体缺口；follow_up_hypothesis 必须非空。",
            "12. retrieve_more 的 follow_up_hypothesis 用于下一轮真实召回，可以结合你的剧情知识补充实体、别名、地点、事件名和关键词；这些补充不能直接写进 answer。",
        ]
    else:
        retrieve_more_rules = [
            "10. 当前只生成最终 conclusion 样本，不要输出 retrieve_more。",
            "11. 如果证据不足以回答核心问题，选择 abstain，并在 answer 中说明缺少什么直接证据。",
            "12. 你可以结合自己的剧情知识判断证据是否相关，但不能把证据外事实写进 answer。",
        ]
    return "\n".join(
        [
            "任务: 基于当前本地检索原始证据，生成一个用于训练 4B 学生模型的 conclusion_generation JSON。",
            "",
            f"用户原问题: {question}",
            f"多轮问答上下文: {dialogue_context or '无'}",
            f"当前检索轮次: {round_index}/{max_rounds}",
            "当前假设文档(JSON):",
            json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2),
            "",
            "本轮检索查询:",
            json.dumps(queries, ensure_ascii=False, indent=2),
            "",
            "MiniRAG 章节隔离扩展信息:",
            "\n".join(scope_lines) if scope_lines else "无",
            "",
            f"当前证据数量: {evidence_count}",
            "当前原始证据块:",
            evidence_text or "无",
            "",
            "输出 schema:",
            json.dumps(
                {
                    "question": question,
                    "next_action": "answer_directly|retrieve_more|clarify_user|abstain",
                    "answer": "string",
                    "missing_slots": ["string"],
                    "clarification_question": "string",
                    "follow_up_hypothesis": {
                        "question": question,
                        "query_type": "fact|relation|causality|reasoning|reveal|mystery|answerability",
                        "entities": ["string"],
                        "keywords": ["string"],
                        "expected_answer_type": "string",
                        "dialogue_context": dialogue_context,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "严格规则:",
            "1. 输出字段只能包含 question,next_action,answer,missing_slots,clarification_question,follow_up_hypothesis。",
            "2. question 必须原样等于用户原问题。",
            "3. 如果证据已经能回答核心问题，选择 answer_directly，并给出最小充分答案；不要因为缺少所有外围细节就 abstain。",
            "4. answer 可以做直接推理和概括，但新增事实必须能由证据原文或证据 metadata 直接支撑。",
            "5. 禁止编造证据中没有出现的活动名、章节名、主线编号、地点、人物身份或剧情来源。",
            "6. 如果 evidence metadata/text 没有出现某个章节名、活动名或主线编号，answer 里绝对不能写该来源名称。",
            "7. 对阴谋/真相/识破/曝光类问题，只要证据能确认主使、关键行动、掩盖对象或结果，就 answer_directly；答案可用“现有证据可确认”限定。",
            "8. 对为什么/原因/动机题，先回答直接原因，再补背景；不要把无关背景改写成主要原因。",
            "9. 如果证据只支持部分事实，answer 中明确区分“可确认部分”和“仍缺部分”，但 next_action 仍可为 answer_directly。",
            *retrieve_more_rules,
            "13. answer_directly/abstain/clarify_user 时 follow_up_hypothesis 必须为 null。",
            "14. clarify_user 只用于用户问题本身歧义无法消解；一般剧情问题不要追问用户。",
            "15. 第一字符必须是 {，最后字符必须是 }。",
        ]
    )


def sanitize_teacher_payload(payload: dict[str, Any], *, question: str, max_round_reached: bool) -> dict[str, Any]:
    cleaned = {key: value for key, value in payload.items() if key in CONCLUSION_FIELDS}
    cleaned.setdefault("question", question)
    cleaned["question"] = question
    action = str(cleaned.get("next_action") or "").strip()
    if action not in {"answer_directly", "retrieve_more", "clarify_user", "abstain"}:
        cleaned["next_action"] = "abstain" if max_round_reached else "retrieve_more"
    if not isinstance(cleaned.get("missing_slots"), list):
        value = cleaned.get("missing_slots")
        cleaned["missing_slots"] = [str(value)] if value not in (None, "", []) else []
    if cleaned.get("clarification_question") is None:
        cleaned["clarification_question"] = ""
    else:
        cleaned["clarification_question"] = str(cleaned.get("clarification_question") or "").strip()
    action = str(cleaned.get("next_action") or "").strip()
    if action in {"answer_directly", "abstain", "clarify_user"}:
        cleaned["follow_up_hypothesis"] = None
    if action != "clarify_user":
        cleaned["clarification_question"] = ""
    if action == "retrieve_more":
        cleaned["answer"] = ""
    else:
        cleaned["answer"] = str(cleaned.get("answer") or "").strip()
    if action == "abstain" and not cleaned["answer"]:
        cleaned["answer"] = "现有检索证据不足以确认该问题。"
    return cleaned


def conclusion_to_payload(conclusion: Any, *, question: str) -> dict[str, Any]:
    follow_up = conclusion.follow_up_hypothesis
    return {
        "question": question,
        "next_action": conclusion.next_action,
        "answer": conclusion.answer,
        "missing_slots": conclusion.missing_slots,
        "clarification_question": conclusion.clarification_question,
        "follow_up_hypothesis": asdict(follow_up) if follow_up is not None else None,
    }


def initial_hypothesis_payload(hypothesis: HypothesisDocument) -> dict[str, Any]:
    return asdict(hypothesis)


def follow_up_hypothesis_payload(hypothesis: HypothesisDocument) -> dict[str, Any]:
    payload = asdict(hypothesis)
    payload.pop("intent", None)
    return payload


def call_teacher_with_validation(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    question: str,
    dialogue_context: str,
    current_intent: str,
    attempts: int,
    allow_retrieve_more: bool,
    max_round_reached: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = user_prompt
    last_error = ""
    last_raw_text = ""
    last_raw_response: dict[str, Any] = {}
    for attempt in range(1, max(1, attempts) + 1):
        raw_text, raw_response = call_teacher_api(
            api_config,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        last_raw_text = raw_text
        last_raw_response = raw_response
        payload = extract_json_object(raw_text)
        if payload is None:
            last_error = "no JSON object found in teacher response"
        else:
            cleaned = sanitize_teacher_payload(payload, question=question, max_round_reached=max_round_reached)
            try:
                conclusion = normalize_conclusion_payload(
                    cleaned,
                    question=question,
                    dialogue_context=dialogue_context,
                    current_intent=current_intent,
                    max_round_reached=max_round_reached,
                )
                return conclusion_to_payload(conclusion, question=question), {
                    "attempt": attempt,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "validation_error": "",
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        prompt = "\n".join(
            [
                user_prompt,
                "",
                "上一次输出未通过本地 schema 校验。",
                f"校验错误: {last_error}",
                "请重新输出合法的单个 JSON 对象。"
                + (
                    "证据不足时可以 retrieve_more，但必须给出合法 follow_up_hypothesis。"
                    if allow_retrieve_more and not max_round_reached
                    else "当前是最终训练样本生成，不允许 retrieve_more；若证据不足请输出 abstain。"
                ),
            ]
        )
    raise RuntimeError(
        "teacher validation failed after "
        f"{attempts} attempts: {last_error}; raw_preview={last_raw_text[:1000]}; "
        f"response_preview={json.dumps(last_raw_response, ensure_ascii=False)[:1000]}"
    )


def build_student_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    evidence_text: str,
    round_index: int = 1,
    max_rounds: int = 1,
) -> str:
    lines = [
        f"task: {CONCLUSION_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {dialogue_context}")
    lines.extend(
        [
            f"hypothesis: {compact_json(asdict(hypothesis))}",
            f"round: {round_index}/{max_rounds}",
            "evidence_blocks:",
            evidence_text or "无",
            "output_schema: conclusion_v2",
        ]
    )
    return "\n".join(lines)


def build_student_hypothesis_prompt(*, question: str, dialogue_context: str) -> str:
    lines = [
        f"task: {INITIAL_HYPOTHESIS_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {dialogue_context}")
    lines.append("output_schema: hypothesis_v2")
    return "\n".join(lines)


def build_student_follow_up_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: HypothesisDocument,
    conclusion_payload: dict[str, Any],
    evidence_text: str,
    round_index: int,
    max_rounds: int,
) -> str:
    lines = [
        f"task: {FOLLOW_UP_HYPOTHESIS_TASK_TYPE}",
        f"question: {question}",
    ]
    if dialogue_context:
        lines.append(f"dialogue_context: {dialogue_context}")
    lines.extend(
        [
            f"hypothesis: {compact_json(asdict(hypothesis))}",
            f"round: {round_index}/{max_rounds}",
            f"previous_conclusion: {compact_json(conclusion_payload)}",
            "evidence_blocks:",
            evidence_text or "无",
            "output_schema: follow_up_hypothesis_v2",
        ]
    )
    return "\n".join(lines)


def make_sft_record(
    *,
    record_id: str,
    task_type: str,
    item: dict[str, Any],
    hypothesis: HypothesisDocument,
    student_prompt: str,
    assistant_payload: dict[str, Any],
    queries: list[str],
    evidence: list[dict[str, Any]] | None,
    trace: dict[str, Any],
    api_config: TeacherApiConfig,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "tool",
        "system": STUDENT_SYSTEM,
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": student_prompt},
            {"from": "gpt", "value": compact_json(assistant_payload)},
        ],
        "meta": {
            "category": "tool",
            "task_family": task_type,
            "generation_mode": "api_grounded_retrieval_raw_text_v2",
            "teacher_model": api_config.model,
            "teacher_api_type": api_config.api_type,
            "source_question_key": item["question_key"],
            "source_question": item["question"],
            "source_dialogue_context": item.get("dialogue_context") or "",
            "hypothesis": asdict(hypothesis),
            "retrieval_queries": queries,
            "evidence_doc_ids": [(hit.get("document") or {}).get("id") for hit in (evidence or [])[:20]],
            "retrieval_trace": trace,
        },
    }


def make_dataset_info(dataset_name: str, splits: list[str]) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": dict(ROLE_TAGS),
        }

    return {f"{dataset_name}_{split}": entry(f"{split}.json") for split in splits}


def export_splits(records: list[dict[str, Any]], output_dir: Path, *, train_ratio: float, val_ratio: float, seed: int) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str((record.get("meta") or {}).get("source_question_key") or record.get("id") or "")
        by_question[key].append(record)
    keys = list(by_question)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_cut = int(len(keys) * train_ratio)
    val_cut = train_cut + int(len(keys) * val_ratio)
    if len(keys) == 1:
        split_keys = {"train": set(keys), "val": set(), "test": set()}
    else:
        split_keys = {
            "train": set(keys[:train_cut]),
            "val": set(keys[train_cut:val_cut]),
            "test": set(keys[val_cut:]),
        }
        if not split_keys["train"]:
            first = keys[0]
            split_keys["train"].add(first)
            split_keys["val"].discard(first)
            split_keys["test"].discard(first)

    summary: dict[str, Any] = {"questions": len(keys), "records": len(records), "splits": {}}
    written_splits: list[str] = []
    for split in ("train", "val", "test"):
        split_records = [record for key in keys if key in split_keys[split] for record in by_question[key]]
        if not split_records:
            continue
        write_json(output_dir / f"{split}.json", split_records)
        written_splits.append(split)
        action_counter: Counter[str] = Counter()
        task_counter: Counter[str] = Counter(str(record.get("task_type") or "") for record in split_records)
        for record in split_records:
            if record.get("task_type") != CONCLUSION_TASK_TYPE:
                continue
            try:
                action = json.loads(record["conversations"][1]["value"]).get("next_action")
            except Exception:
                action = "<invalid>"
            action_counter[str(action)] += 1
        summary["splits"][split] = {
            "questions": len(split_keys[split]),
            "records": len(split_records),
            "task_counts": dict(task_counter),
            "conclusion_actions": dict(action_counter),
        }
    write_json(output_dir / "dataset_info.json", make_dataset_info(output_dir.name, written_splits))
    return summary


def build_api_config(args: argparse.Namespace, runtime_config: dict[str, Any]) -> TeacherApiConfig:
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}
    api_type = args.api_type or generator_cfg.get("api_type") or generator_cfg.get("backend") or "chat_completions"
    if api_type not in {"chat_completions", "anthropic_messages", "responses"}:
        api_type = "chat_completions"
    base_url = args.api_base or generator_cfg.get("api_base_url") or generator_cfg.get("base_url") or "https://api.deepseek.com"
    model = args.model or generator_cfg.get("model") or "deepseek-v4-flash"
    api_key_env = args.api_key_env or generator_cfg.get("api_key_env") or "DEEPSEEK_API_KEY"
    timeout = int(args.timeout or generator_cfg.get("timeout") or generator_cfg.get("timeout_seconds") or 180)
    max_tokens = int(args.max_output_tokens or generator_cfg.get("max_output_tokens") or generator_cfg.get("max_tokens") or 4096)
    temperature = float(args.temperature if args.temperature is not None else generator_cfg.get("temperature", 0.1))
    json_mode = bool(generator_cfg.get("response_format_json", True))
    if args.disable_json_mode:
        json_mode = False
    return TeacherApiConfig(
        api_type=str(api_type),
        base_url=str(base_url),
        model=str(model),
        api_key_env=str(api_key_env),
        timeout_seconds=timeout,
        temperature=temperature,
        max_output_tokens=max_tokens,
        json_mode=json_mode,
        auth_header=args.auth_header,
    )


def build_query_config(args: argparse.Namespace, runtime_config: dict[str, Any]) -> tuple[QueryConfig, dict[str, Any]]:
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    enable_minirag = bool(resolve_value(args.enable_minirag, retrieval_cfg, "enable_minirag", True))
    minirag_chapter_isolation = bool(
        resolve_value(args.minirag_chapter_isolation, retrieval_cfg, "minirag_chapter_isolation", True)
    )
    minirag_auto_second_retrieval = bool(
        resolve_value(args.minirag_auto_second_retrieval, retrieval_cfg, "minirag_auto_second_retrieval", True)
    )
    if not enable_minirag:
        minirag_chapter_isolation = False
        minirag_auto_second_retrieval = False
    return QueryConfig(
        dense_top_k=int(resolve_value(args.dense_top_k, retrieval_cfg, "dense_top_k", 120)),
        sparse_top_k=int(resolve_value(args.sparse_top_k, retrieval_cfg, "sparse_top_k", 120)),
        minirag_top_k=int(resolve_value(args.minirag_top_k, retrieval_cfg, "minirag_top_k", 120 if enable_minirag else 0)),
        fusion_top_k=int(resolve_value(args.fusion_top_k, retrieval_cfg, "fusion_top_k", 80)),
        rerank_top_k=int(resolve_value(args.rerank_top_k, retrieval_cfg, "rerank_top_k", 32)),
        minirag_weight=float(resolve_value(args.minirag_weight, retrieval_cfg, "minirag_weight", 0.35 if enable_minirag else 0.0)),
        minirag_mode_weights={
            str(key): float(value)
            for key, value in (retrieval_cfg.get("minirag_mode_weights") or {}).items()
        },
        minirag_fusion_mode=str(resolve_value(args.minirag_fusion_mode, retrieval_cfg, "minirag_fusion_mode", "score")),
        minirag_chapter_isolation=minirag_chapter_isolation,
        minirag_auto_second_retrieval=minirag_auto_second_retrieval,
        minirag_scope_seed_top_k=int(
            resolve_value(args.minirag_scope_seed_top_k, retrieval_cfg, "minirag_scope_seed_top_k", 40)
        ),
        minirag_expansion_query_top_k=int(
            resolve_value(args.minirag_expansion_query_top_k, retrieval_cfg, "minirag_expansion_query_top_k", 8)
        ),
        reranker_candidate_top_k=int(
            resolve_value(args.reranker_candidate_top_k, retrieval_cfg, "reranker_candidate_top_k", 120)
        ),
        enable_neighbor_expansion=bool(
            resolve_value(args.enable_neighbor_expansion, retrieval_cfg, "enable_neighbor_expansion", False)
        ),
        neighbor_max_seed_docs=int(resolve_value(args.neighbor_max_seed_docs, retrieval_cfg, "neighbor_max_seed_docs", 24)),
        neighbor_story_window=int(resolve_value(args.neighbor_story_window, retrieval_cfg, "neighbor_story_window", 2)),
        neighbor_activity_story_sort_window=int(
            resolve_value(args.neighbor_activity_story_sort_window, retrieval_cfg, "neighbor_activity_story_sort_window", 1)
        ),
        rerank_batch_size=int(resolve_value(args.rerank_batch_size, retrieval_cfg, "rerank_batch_size", 4)),
    ), retrieval_cfg


def build_retriever(args: argparse.Namespace, runtime_config: dict[str, Any]) -> tuple[ArknightsHybridRetriever, QueryConfig, dict[str, Any]]:
    query_config, retrieval_cfg = build_query_config(args, runtime_config)
    device = str(resolve_value(args.device, retrieval_cfg, "device", "cuda"))
    enable_reranker = bool(resolve_value(args.enable_reranker, retrieval_cfg, "enable_reranker", True))
    if args.no_reranker:
        enable_reranker = False
    reranker_model = None
    if enable_reranker:
        configured_reranker = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
        reranker_model = resolve_path(args.reranker_model, configured_reranker or RERANKER_MODEL_DIR)
        if reranker_model is None or not (reranker_model / "config.json").exists():
            raise SystemExit(f"Invalid reranker model path: {reranker_model}")

    enable_minirag = bool(resolve_value(args.enable_minirag, retrieval_cfg, "enable_minirag", True))
    minirag_index_path = None
    if enable_minirag and query_config.minirag_weight > 0:
        minirag_index_path = resolve_path(args.minirag_index, retrieval_cfg.get("minirag_index_path") or MINIRAG_GRAPH_PATH)
        if minirag_index_path is None or not minirag_index_path.exists():
            raise SystemExit(f"MiniRAG index not found: {minirag_index_path}")
    else:
        query_config.minirag_weight = 0.0
        query_config.minirag_chapter_isolation = False
        query_config.minirag_auto_second_retrieval = False

    index_dir = resolve_path(args.index_dir, PROJECT_ROOT / "indexes" / "arknights_story")
    documents_path = resolve_path(args.documents, DOCUMENTS_PATH if args.index_dir is None else index_dir / "documents.jsonl")
    faiss_index_path = resolve_path(args.faiss_index, FAISS_INDEX_PATH if args.index_dir is None else index_dir / "faiss.index")
    bm25_tokens_path = resolve_path(args.bm25_tokens, BM25_TOKENS_PATH if args.index_dir is None else index_dir / "bm25_tokens.pkl")
    embedding_model = resolve_path(args.embedding_model, EMBEDDING_MODEL_DIR)
    if not documents_path or not documents_path.exists():
        raise SystemExit(f"documents path not found: {documents_path}")
    if not faiss_index_path or not faiss_index_path.exists():
        raise SystemExit(f"faiss index path not found: {faiss_index_path}")
    if not bm25_tokens_path or not bm25_tokens_path.exists():
        raise SystemExit(f"bm25 tokens path not found: {bm25_tokens_path}")
    if not embedding_model or not embedding_model.exists():
        raise SystemExit(f"embedding model path not found: {embedding_model}")

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=int(retrieval_cfg.get("reranker_max_length", 1024)),
        documents_path=documents_path,
        faiss_index_path=faiss_index_path,
        bm25_tokens_path=bm25_tokens_path,
        minirag_index_path=minirag_index_path,
        device=device,
    )
    return retriever, query_config, retrieval_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded conclusion_generation SFT data via API teacher and local raw retrieval evidence."
    )
    parser.add_argument("--questions-file", type=Path, default=None, help="Text, JSONL, or JSON list. JSON items may include hypothesis.")
    parser.add_argument("--question", action="append", default=[], help="Inline question. Can be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="Keep generating candidates until this many accepted training records are collected.",
    )
    parser.add_argument(
        "--max-candidate-questions",
        type=int,
        default=None,
        help="Safety cap for generated/processed question candidates when --target-records is set.",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Only keep high-confidence answer_directly records; write rejected records with reasons.",
    )
    parser.add_argument(
        "--generate-questions",
        type=int,
        default=0,
        help="Generate this many question items from sampled local story chunks via API before conclusion generation.",
    )
    parser.add_argument(
        "--generate-theme-questions",
        type=int,
        default=0,
        help="Generate this many complex question items directly from API themes before conclusion generation.",
    )
    parser.add_argument(
        "--theme-requests",
        type=int,
        default=10,
        help="Number of API requests for --generate-theme-questions. Each request uses one theme.",
    )
    parser.add_argument(
        "--questions-per-theme",
        type=int,
        default=10,
        help="Question count requested from each theme-generation API call.",
    )
    parser.add_argument("--question-theme", action="append", default=[], help="Inline question theme. Can be passed multiple times.")
    parser.add_argument("--question-themes-file", type=Path, default=None, help="Text/JSON file with question themes.")
    parser.add_argument("--question-seed-docs-per-group", type=int, default=3)
    parser.add_argument("--questions-per-seed-group", type=int, default=3)
    parser.add_argument("--question-seed-max-chars-per-doc", type=int, default=1200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run retrieval and write teacher_requests.jsonl, but do not call API.")
    parser.add_argument(
        "--full-chain",
        action="store_true",
        help="Generate full-chain SFT records: initial hypothesis, per-round conclusion, and follow-up hypothesis when retrieving more.",
    )
    parser.add_argument("--max-rounds", type=int, default=2, help="Maximum retrieval rounds in --full-chain mode.")
    parser.add_argument(
        "--allow-retrieve-more",
        action="store_true",
        help="Allow teacher outputs with retrieve_more follow_up_hypothesis. Default generates final answer/abstain records.",
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--evidence-top-k", type=int, default=24)
    parser.add_argument("--teacher-evidence-max-chars-per-doc", type=int, default=2200)
    parser.add_argument("--teacher-evidence-max-total-chars", type=int, default=32000)
    parser.add_argument("--student-evidence-max-chars-per-doc", type=int, default=1400)
    parser.add_argument("--student-evidence-max-total-chars", type=int, default=18000)

    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "responses"), default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--auth-header", choices=("bearer", "x-api-key", "both"), default="bearer")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--documents", type=Path, default=None)
    parser.add_argument("--faiss-index", type=Path, default=None)
    parser.add_argument("--bm25-tokens", type=Path, default=None)
    parser.add_argument("--embedding-model", type=Path, default=None)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--enable-reranker", dest="enable_reranker", action="store_true", default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--enable-minirag", dest="enable_minirag", action="store_true", default=None)
    parser.add_argument("--disable-minirag", dest="enable_minirag", action="store_false")
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--minirag-top-k", type=int, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default=None)
    parser.add_argument("--enable-minirag-chapter-isolation", dest="minirag_chapter_isolation", action="store_true", default=None)
    parser.add_argument("--disable-minirag-chapter-isolation", dest="minirag_chapter_isolation", action="store_false")
    parser.add_argument("--enable-minirag-auto-second-retrieval", dest="minirag_auto_second_retrieval", action="store_true", default=None)
    parser.add_argument("--disable-minirag-auto-second-retrieval", dest="minirag_auto_second_retrieval", action="store_false")
    parser.add_argument("--minirag-scope-seed-top-k", type=int, default=None)
    parser.add_argument("--minirag-expansion-query-top-k", type=int, default=None)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=None)
    parser.add_argument("--enable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_true", default=None)
    parser.add_argument("--disable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_false")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=None)
    parser.add_argument("--neighbor-story-window", type=int, default=None)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = load_runtime_config(resolve_path(args.runtime_config, DEFAULT_RUNTIME_CONFIG) or DEFAULT_RUNTIME_CONFIG)
    items = load_question_items(
        resolve_path(args.questions_file),
        args.question,
        allow_empty=args.generate_questions > 0 or args.generate_theme_questions > 0,
    )

    api_config = build_api_config(args, runtime_config)
    if not args.dry_run and not os.environ.get(api_config.api_key_env):
        raise SystemExit(f"Missing API key env var: {api_config.api_key_env}")
    if args.dry_run and (args.generate_questions > 0 or args.generate_theme_questions > 0) and not items:
        raise SystemExit("--dry-run cannot generate questions without calling API. Pass --question/--questions-file or remove --dry-run.")
    full_chain = bool(args.full_chain)
    max_rounds = min(2, max(1, int(args.max_rounds if full_chain else 1)))
    allow_retrieve_more = bool(args.allow_retrieve_more or full_chain)

    retriever, query_config, _retrieval_cfg = build_retriever(args, runtime_config)
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    pipeline = make_pipeline(retriever, query_config, inference_cfg)
    generated_question_count = 0
    generated_theme_question_count = 0
    target_records = max(0, int(args.target_records)) if args.target_records is not None else None
    if target_records is not None and args.generate_theme_questions > 0:
        args.generate_theme_questions = max(args.generate_theme_questions, target_records * 3)
        args.theme_requests = max(
            args.theme_requests,
            (args.generate_theme_questions + max(1, args.questions_per_theme) - 1) // max(1, args.questions_per_theme),
        )
    if args.generate_theme_questions > 0:
        themes = load_question_themes(resolve_path(args.question_themes_file), args.question_theme)
        generated_theme_items = generate_theme_question_items_with_api(
            api_config=api_config,
            output_dir=output_dir,
            target_count=args.generate_theme_questions,
            seed=args.seed,
            themes=themes,
            theme_requests=args.theme_requests,
            questions_per_theme=args.questions_per_theme,
        )
        generated_theme_question_count = len(generated_theme_items)
        items.extend(generated_theme_items)
    if args.generate_questions > 0:
        generated_items = generate_question_items_with_api(
            api_config=api_config,
            documents=retriever.documents,
            output_dir=output_dir,
            target_count=args.generate_questions,
            seed=args.seed,
            docs_per_group=args.question_seed_docs_per_group,
            questions_per_group=args.questions_per_seed_group,
            max_chars_per_doc=args.question_seed_max_chars_per_doc,
        )
        generated_question_count = len(generated_items)
        items.extend(generated_items)
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    if not items:
        raise SystemExit("No questions available after loading/generation.")

    records_path = output_dir / "records.jsonl"
    requests_path = output_dir / "teacher_requests.jsonl"
    raw_path = output_dir / "teacher_raw_responses.jsonl"
    failed_path = output_dir / "failed.jsonl"
    quality_rejected_path = output_dir / "quality_rejected_records.jsonl"
    existing_keys: set[str] = set()
    if args.skip_existing and records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str((record.get("meta") or {}).get("source_question_key") or "")
                if key:
                    existing_keys.add(key)

    records: list[dict[str, Any]] = []
    if args.skip_existing and records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    accepted_records = len(records)

    system_prompt = build_teacher_system_prompt()
    started_all = time.perf_counter()
    for index, item in enumerate(items, start=1):
        if target_records is not None and accepted_records >= target_records:
            break
        question_key = str(item["question_key"])
        if question_key in existing_keys:
            log(f"[skip {index}/{len(items)}] key={question_key}")
            continue
        question = str(item["question"])
        dialogue_context = str(item.get("dialogue_context") or "")
        started = time.perf_counter()
        try:
            hypothesis = coerce_hypothesis(item)
            if full_chain and not args.dry_run:
                initial_record = make_sft_record(
                    record_id=f"{question_key}-{INITIAL_HYPOTHESIS_TASK_TYPE}",
                    task_type=INITIAL_HYPOTHESIS_TASK_TYPE,
                    item=item,
                    hypothesis=hypothesis,
                    student_prompt=build_student_hypothesis_prompt(
                        question=question,
                        dialogue_context=dialogue_context,
                    ),
                    assistant_payload=initial_hypothesis_payload(hypothesis),
                    queries=[],
                    evidence=None,
                    trace={"stage": INITIAL_HYPOTHESIS_TASK_TYPE},
                    api_config=api_config,
                )
                records.append(initial_record)
                append_jsonl(records_path, initial_record)

            queries = build_initial_queries(question, hypothesis)
            generated_for_question = 1 if full_chain and not args.dry_run else 0
            final_action = ""
            for round_index in range(1, max_rounds + 1):
                evidence, trace = retrieve_evidence(
                    pipeline,
                    question=question,
                    hypothesis=hypothesis,
                    queries=queries,
                )
                trace["round"] = round_index
                trace["max_rounds"] = max_rounds
                teacher_evidence_text = render_raw_evidence_for_teacher(
                    evidence,
                    top_k=args.evidence_top_k,
                    max_chars_per_doc=args.teacher_evidence_max_chars_per_doc,
                    max_total_chars=args.teacher_evidence_max_total_chars,
                )
                student_evidence_text = render_raw_evidence_for_teacher(
                    evidence,
                    top_k=args.evidence_top_k,
                    max_chars_per_doc=args.student_evidence_max_chars_per_doc,
                    max_total_chars=args.student_evidence_max_total_chars,
                )
                max_round_reached = round_index >= max_rounds
                round_can_retrieve_more = allow_retrieve_more and not max_round_reached
                user_prompt = build_teacher_user_prompt(
                    question=question,
                    dialogue_context=dialogue_context,
                    hypothesis=hypothesis,
                    queries=queries,
                    evidence_text=teacher_evidence_text,
                    evidence_count=len(evidence),
                    trace=trace,
                    allow_retrieve_more=round_can_retrieve_more,
                    round_index=round_index,
                    max_rounds=max_rounds,
                )
                append_jsonl(
                    requests_path,
                    {
                        "question_key": question_key,
                        "question": question,
                        "round": round_index,
                        "max_rounds": max_rounds,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "retrieval_trace": trace,
                        "evidence_doc_ids": [
                            (hit.get("document") or {}).get("id")
                            for hit in evidence[: args.evidence_top_k]
                        ],
                    },
                )
                if args.dry_run:
                    log(
                        f"[dry-run {index}/{len(items)}] key={question_key} round={round_index}/{max_rounds} "
                        f"evidence={len(evidence)} elapsed={time.perf_counter() - started:.1f}s"
                    )
                    break

                assistant_payload, raw_info = call_teacher_with_validation(
                    api_config,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    question=question,
                    dialogue_context=dialogue_context,
                    current_intent=hypothesis.intent,
                    attempts=args.attempts,
                    allow_retrieve_more=round_can_retrieve_more,
                    max_round_reached=max_round_reached,
                )
                append_jsonl(
                    raw_path,
                    {
                        "question_key": question_key,
                        "question": question,
                        "round": round_index,
                        **raw_info,
                    },
                )
                student_prompt = build_student_prompt(
                    question=question,
                    dialogue_context=dialogue_context,
                    hypothesis=hypothesis,
                    evidence_text=student_evidence_text,
                    round_index=round_index,
                    max_rounds=max_rounds,
                )
                record = make_sft_record(
                    record_id=f"{question_key}-{CONCLUSION_TASK_TYPE}-{round_index:02d}",
                    task_type=CONCLUSION_TASK_TYPE,
                    item=item,
                    hypothesis=hypothesis,
                    student_prompt=student_prompt,
                    assistant_payload=assistant_payload,
                    queries=queries,
                    evidence=evidence,
                    trace=trace,
                    api_config=api_config,
                )
                request_payload = {
                    "question_key": question_key,
                    "question": question,
                    "round": round_index,
                    "max_rounds": max_rounds,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "retrieval_trace": trace,
                    "evidence_doc_ids": [
                        (hit.get("document") or {}).get("id")
                        for hit in evidence[: args.evidence_top_k]
                    ],
                }
                if args.quality_filter:
                    is_valid, reject_reasons = validate_record_quality(
                        record=record,
                        assistant_payload=assistant_payload,
                        question_item=item,
                        request_payload=request_payload,
                    )
                    if not is_valid:
                        append_jsonl(
                            quality_rejected_path,
                            {
                                "question_key": question_key,
                                "question": question,
                                "reasons": reject_reasons,
                                "assistant_payload": assistant_payload,
                                "evidence_doc_ids": request_payload["evidence_doc_ids"],
                            },
                        )
                        final_action = str(assistant_payload.get("next_action") or "")
                        break
                records.append(record)
                append_jsonl(records_path, record)
                accepted_records += 1
                generated_for_question += 1
                final_action = str(assistant_payload.get("next_action") or "")
                if target_records is not None and accepted_records >= target_records:
                    break

                follow_up_payload = assistant_payload.get("follow_up_hypothesis")
                if final_action != "retrieve_more" or not isinstance(follow_up_payload, dict):
                    break
                follow_up = normalize_hypothesis_payload(
                    follow_up_payload,
                    question=question,
                    dialogue_context=dialogue_context,
                    current_intent=hypothesis.intent,
                )
                if full_chain:
                    follow_up_record = make_sft_record(
                        record_id=f"{question_key}-{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}-{round_index:02d}",
                        task_type=FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                        item=item,
                        hypothesis=hypothesis,
                        student_prompt=build_student_follow_up_prompt(
                            question=question,
                            dialogue_context=dialogue_context,
                            hypothesis=hypothesis,
                            conclusion_payload=assistant_payload,
                            evidence_text=student_evidence_text,
                            round_index=round_index,
                            max_rounds=max_rounds,
                        ),
                        assistant_payload=follow_up_hypothesis_payload(follow_up),
                        queries=queries,
                        evidence=evidence,
                        trace=trace,
                        api_config=api_config,
                    )
                    records.append(follow_up_record)
                    append_jsonl(records_path, follow_up_record)
                    generated_for_question += 1
                hypothesis = merge_hypotheses(hypothesis, follow_up)
                queries = [build_retrieval_query(hypothesis), *build_follow_up_hypothesis_queries(question, hypothesis)]
            if not args.dry_run:
                log(
                    f"[ok {index}/{len(items)}] key={question_key} action={final_action} "
                    f"records={generated_for_question} accepted={accepted_records} elapsed={time.perf_counter() - started:.1f}s"
                )
        except Exception as exc:
            payload = {
                "question_key": question_key,
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
            append_jsonl(failed_path, payload)
            log(f"[fail {index}/{len(items)}] key={question_key} error={payload['error']}")

    if not args.dry_run and records:
        split_summary = export_splits(
            records,
            output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        split_summary = {"questions": len(items), "records": len(records), "splits": {}}

    build_summary = {
        "output_dir": str(output_dir),
        "runtime_config": str(resolve_path(args.runtime_config, DEFAULT_RUNTIME_CONFIG)),
        "dry_run": args.dry_run,
        "allow_retrieve_more": args.allow_retrieve_more,
        "teacher_model": api_config.model,
        "api_type": api_config.api_type,
        "input_questions": len(items),
        "generated_questions": generated_question_count,
        "generated_theme_questions": generated_theme_question_count,
        "records": len(records),
        "target_records": target_records,
        "quality_filter": args.quality_filter,
        "quality_rejected": count_lines(quality_rejected_path),
        "failed": count_lines(failed_path),
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "query_config": asdict(query_config),
        "splits": split_summary.get("splits", {}),
    }
    write_json(output_dir / "build_summary.json", build_summary)
    print(json.dumps(build_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
