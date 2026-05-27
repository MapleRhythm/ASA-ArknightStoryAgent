#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1_planner_rebalanced"
)
DEFAULT_TEST_SOURCE_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_current_short_prompt_v1"
DEFAULT_MINIRAG_GRAPH = PROJECT_ROOT / "indexes/arknights_story_minirag_v3/graph.json"

RUNTIME_TASKS = {
    "user_question_hypothesis_generation",
    "follow_up_hypothesis_generation",
    "conclusion_generation",
}
INTENTS = {
    "character_relation",
    "compare",
    "event_summary",
    "out_of_scope",
    "persona_chat",
    "plot_fact",
    "plot_reasoning",
    "timeline",
}
QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
IDENTITY_WORDS = ("是谁", "身份", "身世", "来历", "真相", "关系", "父", "母", "后人")
CAUSAL_WORDS = ("为什么", "为何", "原因", "导致", "怎么会", "动机", "目的")
REVEAL_WORDS = ("真相", "阴谋", "秘密", "幕后", "主使", "识破", "揭露", "暴露")
MYSTERY_WORDS = ("谜", "究竟", "到底", "怎么回事")
FACT_WORDS = ("哪里", "哪一", "什么时候", "多少", "什么是", "是什么")
PRONOUNS = {"她", "他", "它", "她们", "他们", "它们", "这位", "那位", "user", "assistant"}
NOISY_TERMS = {
    "",
    "无",
    "问题",
    "用户问题",
    "当前",
    "证据",
    "剧情",
    "回答",
    "关系",
    "身份",
    "原因",
    "真相",
    "联系",
    "互动",
    "角色",
    "职责",
    "内容",
    "重罪",
    "帮凶",
    "阶段",
    "stage",
    "选中干员",
    "选中干员2",
    "信赖提升",
    "信赖提升后交谈",
    "信赖提升后交谈2",
    "干员语音",
    "交谈2",
    "任务",
    "负责",
    "立场",
    "什么",
    "为什么",
    "怎么",
    "如何",
}
ENTITY_RE = re.compile(r"[\u4e00-\u9fff·]{2,16}|[A-Za-z][A-Za-z0-9_.-]{1,31}")
BAD_HINT_RE = re.compile(r"[\[\]{}#\"'“”‘’…]|chunk|obt/|activities/|info/|charword/|level_", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assistant_text(record: dict[str, Any]) -> str:
    for message in reversed(record.get("conversations") or []):
        if message.get("from") in {"gpt", "assistant"}:
            return str(message.get("value") or message.get("content") or "")
    return str(record.get("output") or "")


def user_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in record.get("conversations") or []:
        if message.get("from") in {"human", "user"}:
            parts.append(str(message.get("value") or message.get("content") or ""))
    return "\n".join(parts)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def extract_labeled_section(text: str, label: str, next_labels: list[str]) -> str:
    start = text.find(label)
    if start < 0:
        return ""
    start += len(label)
    end = len(text)
    for next_label in next_labels:
        index = text.find(next_label, start)
        if index >= 0:
            end = min(end, index)
    return text[start:end].strip(" \n:：")


def extract_question(prompt: str, payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "").strip()
    if question:
        return question
    for pattern in (r"用户问题[:：]\s*(.+)", r"用户原问题[:：]\s*(.+)"):
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
    return ""


def compact_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen or value in PRONOUNS:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def valid_hint_term(value: str) -> bool:
    value = str(value).strip().strip("\"'")
    if not value or value in NOISY_TERMS or value in PRONOUNS:
        return False
    if len(value) < 2 or len(value) > 16:
        return False
    if value.startswith(("activity:", "story:", "stage:")):
        return False
    if BAD_HINT_RE.search(value):
        return False
    if set(value) <= {".", "。", "，", ",", "、", "-", "_", "·"}:
        return False
    if value.count(".") >= 2:
        return False
    if value.endswith(("语音", "交谈")) or "信赖" in value:
        return False
    return True


def infer_intent(question: str, expected_answer_type: str = "") -> str:
    text = question + " " + expected_answer_type
    if any(word in text for word in ("关系", "父", "母", "同伴", "朋友")):
        return "character_relation"
    if any(word in text for word in ("比较", "区别", "不同")):
        return "compare"
    if any(word in text for word in ("时间线", "先后", "后来")):
        return "timeline"
    if any(word in text for word in CAUSAL_WORDS):
        return "plot_reasoning"
    if any(word in text for word in ("概括", "经过", "发生了什么")):
        return "event_summary"
    return "plot_fact"


def infer_query_type(question: str, intent: str = "", expected_answer_type: str = "") -> str:
    text = question + " " + intent + " " + expected_answer_type
    if any(word in text for word in REVEAL_WORDS):
        return "reveal"
    if any(word in text for word in MYSTERY_WORDS):
        return "mystery"
    if any(word in text for word in CAUSAL_WORDS):
        return "causality"
    if intent == "character_relation" or "关系" in text or any(word in text for word in IDENTITY_WORDS):
        return "relation"
    if "可回答" in text or ("是什么" in text and "为什么" in text):
        return "answerability"
    if intent in {"plot_reasoning", "event_summary"}:
        return "reasoning"
    if any(word in text for word in FACT_WORDS):
        return "fact"
    return "reasoning"


def clean_hypothesis_payload(payload: dict[str, Any], *, follow_up: bool = False) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    expected_answer_type = str(payload.get("expected_answer_type") or "综合剧情问答").strip()
    intent = str(payload.get("intent") or "").strip()
    if intent not in INTENTS:
        intent = infer_intent(question, expected_answer_type)
    query_type = str(payload.get("query_type") or "").strip()
    if query_type not in QUERY_TYPES:
        query_type = infer_query_type(question, intent, expected_answer_type)
    entities = dedupe([str(item) for item in payload.get("entities") or []], 8)
    keywords = dedupe([str(item) for item in payload.get("keywords") or []], 14)
    if not keywords:
        keywords = dedupe(entities + [token for token in ENTITY_RE.findall(question) if token not in NOISY_TERMS], 14)
    result: dict[str, Any] = {
        "question": question,
        "query_type": query_type,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": str(payload.get("dialogue_context") or "").strip(),
    }
    if not follow_up:
        result = {"question": question, "intent": intent, **{k: v for k, v in result.items() if k != "question"}}
    return result


def clean_conclusion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    action = str(payload.get("next_action") or "").strip()
    if action not in RETRIEVAL_ACTIONS:
        action = "abstain"
    answer = str(payload.get("answer") or "").strip()
    missing_slots = [str(item).strip() for item in payload.get("missing_slots") or [] if str(item).strip()]
    clarification_question = str(payload.get("clarification_question") or "").strip()
    follow_up = payload.get("follow_up_hypothesis")
    clean_follow_up = None
    if action == "retrieve_more":
        if isinstance(follow_up, dict):
            clean_follow_up = clean_hypothesis_payload(follow_up, follow_up=True)
        else:
            clean_follow_up = clean_hypothesis_payload(
                {
                    "question": question,
                    "entities": [],
                    "keywords": missing_slots,
                    "expected_answer_type": "补充检索",
                },
                follow_up=True,
            )
        answer = ""
    elif action in {"answer_directly", "abstain"} and not answer:
        action = "abstain"
        answer = "现有证据不足以稳定回答该问题。"
    elif action == "clarify_user" and not clarification_question:
        clarification_question = "请补充你想询问的具体角色、章节或事件。"
    return {
        "question": question,
        "next_action": action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
        "follow_up_hypothesis": clean_follow_up,
    }


class MiniRAGHintBuilder:
    def __init__(self, graph_path: Path | None, *, max_entities: int, max_relations: int) -> None:
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.entity_names: set[str] = set()
        self.relations_by_entity: dict[str, list[dict[str, str]]] = {}
        self.cooccurrence: dict[str, list[str]] = {}
        if graph_path and graph_path.exists():
            payload = load_json(graph_path)
            self.entity_names = {
                str(key)
                for key in (payload.get("entity_to_doc_indices") or {}).keys()
                if valid_hint_term(str(key))
            }
            self.cooccurrence = {
                str(key): [str(item) for item in value[: max_entities * 4] if valid_hint_term(str(item))]
                for key, value in (payload.get("entity_cooccurrence") or {}).items()
                if isinstance(value, list) and valid_hint_term(str(key))
            }
            for relation in payload.get("teacher_relations") or []:
                if not isinstance(relation, dict):
                    continue
                head = str(relation.get("head") or "").strip()
                tail = str(relation.get("tail") or "").strip()
                rel = str(relation.get("relation") or "").strip()
                if not (valid_hint_term(head) and valid_hint_term(tail) and valid_hint_term(rel)):
                    continue
                item = {"head": head, "relation": rel, "tail": tail}
                self.relations_by_entity.setdefault(head, []).append(item)
                self.relations_by_entity.setdefault(tail, []).append(item)

    def build(self, *, question: str, payload: dict[str, Any], prompt: str) -> str:
        seed_text = " ".join(
            [
                question,
                " ".join(str(item) for item in payload.get("entities") or []),
                " ".join(str(item) for item in payload.get("keywords") or []),
                prompt[:1600],
            ]
        )
        candidates = dedupe([token for token in ENTITY_RE.findall(seed_text) if valid_hint_term(token)], 24)
        matched = [item for item in candidates if item in self.entity_names or item in self.relations_by_entity]
        for entity in sorted(self.entity_names, key=len, reverse=True):
            if len(matched) >= self.max_entities:
                break
            if not valid_hint_term(entity) or len(entity) < 3:
                continue
            if entity in seed_text and entity not in matched:
                matched.append(entity)
        matched = dedupe([item for item in matched + [item for item in payload.get("entities") or []] if valid_hint_term(item)], self.max_entities)

        relations: list[str] = []
        seen_relations: set[str] = set()
        for entity in matched:
            for relation in self.relations_by_entity.get(entity, []):
                rendered = f"{relation['head']}-{relation['relation']}-{relation['tail']}"
                if rendered in seen_relations:
                    continue
                if any(term in seed_text for term in (relation["head"], relation["tail"], relation["relation"])):
                    seen_relations.add(rendered)
                    relations.append(rendered)
                if len(relations) >= self.max_relations:
                    break
            if len(relations) >= self.max_relations:
                break

        neighbors: list[str] = []
        for entity in matched:
            neighbors.extend(self.cooccurrence.get(entity, [])[:3])
        neighbors = dedupe([item for item in neighbors if item not in matched and valid_hint_term(item)], self.max_entities)

        if not matched and not relations and not neighbors:
            return "none"
        chunks = []
        if matched:
            chunks.append("entities=" + ",".join(matched))
        if relations:
            chunks.append("relations=" + ";".join(relations))
        if neighbors:
            chunks.append("neighbors=" + ",".join(neighbors))
        return " | ".join(chunks)


def extract_current_hypothesis(prompt: str) -> str:
    raw = extract_json_after_label(prompt, "当前假设文档(JSON)") or extract_labeled_section(
        prompt,
        "当前假设文档(JSON)",
        ["历史生成结果", "历史检索上下文", "当前检索轮次", "当前证据"],
    )
    return compact_text(raw, 520)


def extract_json_after_label(text: str, label: str) -> str:
    start = text.find(label)
    if start < 0:
        return ""
    brace_start = text.find("{", start + len(label))
    if brace_start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(brace_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : index + 1]
    return ""


def extract_dialogue_context(prompt: str, payload: dict[str, Any]) -> str:
    value = str(payload.get("dialogue_context") or "").strip()
    if value:
        return compact_text(value, 260)
    for label in ("多轮上下文", "多轮问答上下文"):
        raw = extract_labeled_section(
            prompt,
            label,
            ["当前假设文档", "历史生成结果", "当前检索轮次", "当前证据", "请生成", "字段要求"],
        )
        if raw and raw != "无":
            return compact_text(raw, 260)
    return ""


def extract_round(prompt: str) -> str:
    match = re.search(r"当前检索轮次[:：]\s*第?\s*(\d+)\s*轮\s*/\s*最多\s*(\d+)\s*轮", prompt)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return ""


def extract_evidence_brief(prompt: str, *, max_items: int, max_chars: int) -> list[str]:
    marker = "当前证据"
    index = prompt.rfind(marker)
    if index < 0:
        return []
    raw = prompt[index + len(marker) :].strip(" \n:：")
    raw = raw.split("请基于", 1)[0].split("输出要求", 1)[0].strip()
    items: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed[:max_items]:
                if isinstance(item, dict):
                    label = item.get("id") or item.get("stage_code") or item.get("story_name") or item.get("activity_name") or "evidence"
                    text = item.get("clean_text") or item.get("text") or item.get("search_text") or ""
                    items.append(f"{label}: {compact_text(text, max_chars)}")
                else:
                    items.append(compact_text(str(item), max_chars))
            return items
    except Exception:
        pass
    blocks = re.split(r"(?=\[证据\s*\d+\])", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        label = ""
        text_parts: list[str] = []
        for line in lines:
            if line.startswith("id:") or line.startswith("stage_code:") or line.startswith("story_name:"):
                label = label or line.split(":", 1)[-1].strip()
            elif not line.startswith(("[证据", "activity_name:", "avg_tag:", "source_path:", "chain_roles:", "clean_text:")):
                text_parts.append(line)
        text = " ".join(text_parts) or block
        items.append(f"{label or 'evidence'}: {compact_text(text, max_chars)}")
        if len(items) >= max_items:
            break
    if not items and raw:
        items.append(compact_text(raw, max_chars))
    return items[:max_items]


def short_prompt_for_record(
    *,
    task_type: str,
    source_prompt: str,
    output_payload: dict[str, Any],
    hints: str,
    max_evidence_items: int,
    max_evidence_chars: int,
) -> str:
    question = extract_question(source_prompt, output_payload)
    lines = [f"task: {task_type}", f"question: {question}"]
    dialogue_context = extract_dialogue_context(source_prompt, output_payload)
    if dialogue_context:
        lines.append(f"dialogue_context: {dialogue_context}")
    if task_type == "user_question_hypothesis_generation":
        lines.append(f"minirag_hints: {hints}")
        lines.append("output_schema: hypothesis_v2")
    elif task_type == "follow_up_hypothesis_generation":
        current_hypothesis = extract_current_hypothesis(source_prompt)
        if current_hypothesis:
            lines.append(f"current_hypothesis: {current_hypothesis}")
        evidence = extract_evidence_brief(source_prompt, max_items=max_evidence_items, max_chars=max_evidence_chars)
        if evidence:
            lines.append("evidence_brief:")
            lines.extend(f"{index}. {item}" for index, item in enumerate(evidence, start=1))
        lines.append(f"minirag_hints: {hints}")
        lines.append("output_schema: follow_up_hypothesis_v2")
    else:
        current_hypothesis = extract_current_hypothesis(source_prompt)
        if current_hypothesis:
            lines.append(f"hypothesis: {current_hypothesis}")
        round_text = extract_round(source_prompt)
        if round_text:
            lines.append(f"round: {round_text}")
        evidence = extract_evidence_brief(source_prompt, max_items=max_evidence_items, max_chars=max_evidence_chars)
        if evidence:
            lines.append("evidence_brief:")
            lines.extend(f"{index}. {item}" for index, item in enumerate(evidence, start=1))
        if output_payload.get("missing_slots"):
            lines.append("missing_slots: " + "; ".join(str(item) for item in output_payload["missing_slots"][:5]))
        lines.append(f"minirag_hints: {hints}")
        lines.append("output_schema: conclusion_v2")
    return "\n".join(lines)


def convert_record(
    record: dict[str, Any],
    *,
    hint_builder: MiniRAGHintBuilder,
    max_evidence_items: int,
    max_evidence_chars: int,
) -> tuple[dict[str, Any] | None, str | None]:
    task_type = str(record.get("task_type") or "")
    if task_type not in RUNTIME_TASKS:
        return None, "skip_non_runtime_task"
    payload = parse_json_object(assistant_text(record))
    if payload is None:
        return None, "invalid_assistant_json"
    source_prompt = user_text(record)
    if task_type == "user_question_hypothesis_generation":
        output_payload = clean_hypothesis_payload(payload, follow_up=False)
    elif task_type == "follow_up_hypothesis_generation":
        output_payload = clean_hypothesis_payload(payload, follow_up=True)
    else:
        output_payload = clean_conclusion_payload(payload)
    question = extract_question(source_prompt, output_payload)
    if not question:
        return None, "missing_question"
    hints = hint_builder.build(question=question, payload=output_payload, prompt=source_prompt)
    prompt = short_prompt_for_record(
        task_type=task_type,
        source_prompt=source_prompt,
        output_payload=output_payload,
        hints=hints,
        max_evidence_items=max_evidence_items,
        max_evidence_chars=max_evidence_chars,
    )
    converted = {
        "id": str(record.get("id") or f"short-{random.getrandbits(64):016x}"),
        "task_type": task_type,
        "bucket": "tool",
        "system": "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。",
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(output_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "meta": {
            **(record.get("meta") or {}),
            "short_prompt_schema": "current_pipeline_v1",
            "source_task_type": task_type,
        },
    }
    return converted, None


def synthesize_follow_up_record(conclusion_record: dict[str, Any]) -> dict[str, Any] | None:
    if conclusion_record.get("task_type") != "conclusion_generation":
        return None
    output_payload = parse_json_object(assistant_text(conclusion_record))
    if not isinstance(output_payload, dict) or output_payload.get("next_action") != "retrieve_more":
        return None
    follow_up = output_payload.get("follow_up_hypothesis")
    if not isinstance(follow_up, dict):
        return None
    prompt = user_text(conclusion_record)
    lines = []
    for line in prompt.splitlines():
        if line.startswith("task: conclusion_generation"):
            lines.append("task: follow_up_hypothesis_generation")
        elif line.startswith("output_schema:"):
            lines.append("output_schema: follow_up_hypothesis_v2")
        elif line.startswith("round:") or line.startswith("hypothesis:") or line.startswith("question:") or line.startswith("dialogue_context:"):
            lines.append(line)
        elif line.startswith("missing_slots:") or line.startswith("minirag_hints:"):
            lines.append(line)
        elif line.startswith("evidence_brief:") or re.match(r"\d+\. ", line):
            lines.append(line)
    if not lines:
        return None
    return {
        "id": str(conclusion_record.get("id") or "unknown") + "-synthetic-follow-up",
        "task_type": "follow_up_hypothesis_generation",
        "bucket": "tool",
        "system": "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。",
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": "\n".join(lines)},
            {"from": "gpt", "value": json.dumps(follow_up, ensure_ascii=False, separators=(",", ":"))},
        ],
        "meta": {
            **(conclusion_record.get("meta") or {}),
            "short_prompt_schema": "current_pipeline_v1",
            "source_task_type": "conclusion_generation",
            "synthetic_task": "follow_up_hypothesis_generation",
        },
    }


def load_split(source_dir: Path, split: str) -> list[dict[str, Any]]:
    path = source_dir / f"{split}.json"
    return load_json(path) if path.exists() else []


def convert_split(
    records: list[dict[str, Any]],
    *,
    hint_builder: MiniRAGHintBuilder,
    max_evidence_items: int,
    max_evidence_chars: int,
    synthesize_follow_up: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    converted: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_outputs: set[str] = set()
    for record in records:
        item, reason = convert_record(
            record,
            hint_builder=hint_builder,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
        )
        if item is None:
            stats[reason or "skipped"] += 1
            continue
        signature = item["task_type"] + "\n" + item["conversations"][0]["value"] + "\n" + item["conversations"][1]["value"]
        if signature in seen_outputs:
            stats["duplicate"] += 1
            continue
        seen_outputs.add(signature)
        converted.append(item)
        stats["converted"] += 1
        stats[f"task:{item['task_type']}"] += 1
        if synthesize_follow_up:
            synthetic = synthesize_follow_up_record(item)
            if synthetic is not None:
                synthetic_signature = (
                    synthetic["task_type"]
                    + "\n"
                    + synthetic["conversations"][0]["value"]
                    + "\n"
                    + synthetic["conversations"][1]["value"]
                )
                if synthetic_signature not in seen_outputs:
                    seen_outputs.add(synthetic_signature)
                    converted.append(synthetic)
                    stats["synthetic_follow_up"] += 1
                    stats["task:follow_up_hypothesis_generation"] += 1
    return converted, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short-prompt current-schema 4B SFT data with compact MiniRAG hints.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--test-source-dir", type=Path, default=DEFAULT_TEST_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minirag-graph", type=Path, default=DEFAULT_MINIRAG_GRAPH)
    parser.add_argument("--max-evidence-items", type=int, default=6)
    parser.add_argument("--max-evidence-chars", type=int, default=220)
    parser.add_argument("--max-minirag-entities", type=int, default=8)
    parser.add_argument("--max-minirag-relations", type=int, default=6)
    parser.add_argument("--no-synthesize-follow-up", action="store_true")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output directory already exists: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    hint_builder = MiniRAGHintBuilder(
        args.minirag_graph if args.minirag_graph.is_absolute() else PROJECT_ROOT / args.minirag_graph,
        max_entities=args.max_minirag_entities,
        max_relations=args.max_minirag_relations,
    )

    summary: dict[str, Any] = {
        "source_dir": str(args.source_dir),
        "test_source_dir": str(args.test_source_dir),
        "minirag_graph": str(args.minirag_graph),
        "splits": {},
    }
    for split in ("train", "val"):
        records = load_split(args.source_dir, split)
        converted, stats = convert_split(
            records,
            hint_builder=hint_builder,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            synthesize_follow_up=not args.no_synthesize_follow_up,
        )
        write_json(output_dir / f"{split}.json", converted)
        summary["splits"][split] = dict(stats)

    test_records = load_split(args.source_dir, "test")
    if not test_records and args.test_source_dir:
        test_records = load_split(args.test_source_dir, "test")
    converted_test, test_stats = convert_split(
        test_records,
        hint_builder=hint_builder,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        synthesize_follow_up=not args.no_synthesize_follow_up,
    )
    if converted_test:
        write_json(output_dir / "test.json", converted_test)
        summary["splits"]["test"] = dict(test_stats)

    dataset_name = output_dir.name
    dataset_info = {
        f"{dataset_name}_train": {"file_name": "train.json", "formatting": "sharegpt", "columns": {"messages": "conversations"}},
        f"{dataset_name}_val": {"file_name": "val.json", "formatting": "sharegpt", "columns": {"messages": "conversations"}},
    }
    if converted_test:
        dataset_info[f"{dataset_name}_test"] = {
            "file_name": "test.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
