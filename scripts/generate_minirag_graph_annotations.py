#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import DOCUMENTS_PATH  # noqa: E402
from goldenglow.data.sft_teacher import (  # noqa: E402
    TeacherApiConfig,
    call_teacher_api,
    parse_teacher_json,
)


DOC_ID_IN_FLAT_ENTITY_RE = re.compile(r'\],\s*"([^"]+#chunk-\d{4})"\s*,')
COMPACT_RELATIONS_KEY_RE = re.compile(r'(?<!\])\s*,\s*"relations"\s*:')
MISSING_ENTITIES_ARRAY_CLOSE_RE = re.compile(r'(?<!\])\]\],\s*"relations"\s*:')
BROKEN_RELATIONS_KEY_RE = re.compile(r'\]\],\s*\[\s*"relations"\s*:')
MISSING_RELATION_ARRAY_RE = re.compile(r'\],\s*"\[\s*"([^"]+)"')
RELATION_BARE_DOC_IDS_RE = re.compile(
    r'(\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*)'
    r'("[^"]+#chunk-\d{4}"(?:\s*,\s*"[^"]+#chunk-\d{4}")*)'
    r'(\s*,\s*"[^"]*"\s*\])'
)
BARE_RELATION_ARRAY_RE = re.compile(
    r'(?<=\])\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*\['
)
BARE_RELATION_WITH_DOC_IDS_RE = re.compile(
    r'(?<=\])\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*'
    r'("[^"]+#chunk-\d{4}"(?:\s*,\s*"[^"]+#chunk-\d{4}")*)\s*,\s*("[^"]*")'
)
EXTRA_RELATIONS_ARRAY_CLOSE_RE = re.compile(
    r'\]\],\s*(\[\s*"(?![^"]+#chunk-\d{4})[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*\[)'
)
MISSING_COMMA_BETWEEN_ARRAY_ITEMS_RE = re.compile(r'\]\s+(\[\s*"[^"]+"\s*,)')
STRAY_QUOTE_AFTER_ARRAY_RE = re.compile(r'(\[[^\[\]]*\])"\s*(?=\])')
STRAY_OBJECT_CLOSE_IN_ARRAY_RE = re.compile(r'\]\s*}\s*,\s*(\[\s*"[^"]+"\s*,)')
EXTRA_ENTITY_ARRAY_CLOSE_RE = re.compile(r'\]\],\s*(\[\s*"[^"]+#chunk-\d{4}"\s*,)')
EXTRA_BARE_ENTITY_ARRAY_CLOSE_RE = re.compile(r'\]\],\s*("[^"]+#chunk-\d{4}"\s*,)')
MISSING_ENTITY_ROW_CLOSE_RE = re.compile(
    r'("(?:person|organization|place|event|item|concept|other)"\s*,\s*\[[^\[\]]*\])\s*,\s*(\[\s*)?("[^"]+#chunk-\d{4}"\s*,)'
)


def _balance_square_brackets_before_final_brace(value: str) -> str:
    """Append missing array closers before the final object brace, ignoring strings."""
    stripped = value.rstrip()
    if not stripped.endswith("}"):
        return value
    depth = 0
    in_string = False
    escaped = False
    for char in stripped[:-1]:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
    if depth <= 0:
        return value
    return stripped[:-1] + ("]" * depth) + "}"


def _repair_compact_teacher_json(value: str) -> str:
    repaired = value
    previous = ""
    while repaired != previous:
        previous = repaired
        repaired = DOC_ID_IN_FLAT_ENTITY_RE.sub(r'],["\1",', repaired)
        repaired = EXTRA_ENTITY_ARRAY_CLOSE_RE.sub(r'], \1', repaired)
        repaired = EXTRA_BARE_ENTITY_ARRAY_CLOSE_RE.sub(r'], [\1', repaired)
        repaired = MISSING_ENTITY_ROW_CLOSE_RE.sub(r'\1], [\3', repaired)
        repaired = BROKEN_RELATIONS_KEY_RE.sub(r']],"relations":', repaired, count=1)
        repaired = MISSING_RELATION_ARRAY_RE.sub(r'],["\1"', repaired)
        repaired = RELATION_BARE_DOC_IDS_RE.sub(r'\1[\2]\3', repaired)
        repaired = BARE_RELATION_WITH_DOC_IDS_RE.sub(r'], ["\1", "\2", "\3", [\4], \5', repaired)
        repaired = BARE_RELATION_ARRAY_RE.sub(r'], ["\1", "\2", "\3", [', repaired)
        repaired = EXTRA_RELATIONS_ARRAY_CLOSE_RE.sub(r'], \1', repaired)
        repaired = MISSING_COMMA_BETWEEN_ARRAY_ITEMS_RE.sub(r'], \1', repaired)
        repaired = STRAY_QUOTE_AFTER_ARRAY_RE.sub(r'\1', repaired)
        repaired = STRAY_OBJECT_CLOSE_IN_ARRAY_RE.sub(r'], \1', repaired)
        repaired = repaired.replace(']"],"relations":', ']],"relations":')
        repaired = MISSING_ENTITIES_ARRAY_CLOSE_RE.sub(r']]],"relations":', repaired, count=1)
    return _balance_square_brackets_before_final_brace(repaired)


REQUEST_RATE_LOCK = threading.Lock()
LAST_REQUEST_STARTED_AT = 0.0


def wait_for_request_slot(interval_seconds: float) -> None:
    """Ensure API request starts are spaced out across worker threads."""
    global LAST_REQUEST_STARTED_AT
    if interval_seconds <= 0:
        return
    with REQUEST_RATE_LOCK:
        now = time.monotonic()
        wait_seconds = LAST_REQUEST_STARTED_AT + interval_seconds - now
        if wait_seconds > 0:
            print(f"[rate-limit] sleep={wait_seconds:.1f}s", flush=True)
            time.sleep(wait_seconds)
        LAST_REQUEST_STARTED_AT = time.monotonic()


GRAPH_PROMPT_SYSTEM = (
    "你是《明日方舟》剧情知识图谱标注器。"
    "任务是从给定 chunk 中抽取可回溯的实体和关系，用于 MiniRAG 图召回。"
    "必须只输出严格 JSON，不要 markdown，不要解释。"
)


GRAPH_PROMPT_TEMPLATE = """请从下面 {batch_size} 个连续或近邻剧情 chunk 中抽取实体与关系。

本任务用于构建 MiniRAG 异构图。API 按调用次数收费，因此每次请求已经尽量塞入较多 chunks；
你必须完整处理输入中的每一个 chunk，不要只处理前几条。若上下文很长，仍然要逐 chunk 输出。

输出 JSON schema 必须严格为紧凑数组格式，禁止使用 markdown：
{{
  "chunks": [
    ["doc_id", [["实体名", "person|organization|place|event|item|concept|other", ["可选别名"]]]]
  ],
  "relations": [
    ["head实体名", "关系短语", "tail实体名", ["支撑该关系的doc_id，可包含多个"], "能支撑该关系的原文短句或跨chunk证据摘要"]
  ]
}}

规则：
1. 每个 chunk 尽量抽 4-12 个关键实体，短 chunk 可少于 4 个；不要输出普通代词、语气词、泛泛名词。
2. 关系在整个输入 batch 级别抽取，不要局限于单个 chunk；很多关系需要前后文共同支撑。
3. 实体名必须是原文中出现的命名实体、组织、地点、事件、道具、制度/概念名；禁止代词和问句残片。
4. relation 用短中文短语，例如：所属、亲属、上下级、师徒、合作、敌对、保护、追查、导致、发生在、使用、提及、关联、决定、动机、原因、结果、揭示、伪装、背叛、阻止、前往、来自。
5. 本 batch 目标关系数：至少 {target_relations} 条；如果原文信息足够，优先输出更多关系。不要只抽几十条。
6. evidence_doc_ids 必须来自输入 chunks；单 chunk 支撑就填 1 个 doc_id，跨 chunk 支撑就填 2-5 个 doc_id。
7. evidence 可以是原文短句；跨 chunk 关系可以写简短证据摘要，但必须能由 evidence_doc_ids 对应原文支撑。
8. 允许抽跨 chunk 关系，但禁止使用输入之外的设定脑补关系。
9. 优先抽会帮助问答召回的关系：人物-组织、人物-地点、人物-事件、动机-行动、原因-结果、身份-伪装、敌对/合作、前文伏笔-后文揭示。
10. 禁止输出未出现在输入中的 doc_id；输出 chunks 数量应等于输入 chunks 数量。
11. 为减少 JSON 损坏，输出必须使用双引号，字符串内部不要换行；evidence 控制在 80 字以内。

输入 chunks：
{chunk_blocks}
"""


GRAPH_COMPACT_PROMPT_TEMPLATE = """请从下面 {batch_size} 个连续或近邻剧情 chunk 中抽取实体与跨 chunk 关系。

本任务用于构建 MiniRAG 异构图。为了避免超长 JSON 损坏，本轮不要输出 chunks 外壳，
而是把实体压平成顶层 entities 数组，关系放进顶层 relations。

输出 JSON schema 必须严格为：
{{
  "entities": [
    ["输入中的完整doc_id", "实体名", "person|organization|place|event|item|concept|other", ["可选别名"]]
  ],
  "relations": [
    ["head实体名", "关系短语", "tail实体名", ["支撑该关系的doc_id，可包含多个"], "能支撑该关系的原文短句或跨chunk证据摘要"]
  ]
}}

规则：
1. entities 必须覆盖每个输入 doc_id；每个 doc 尽量抽 4-10 个关键实体，短文本可少于 4 个。
2. 只抽能帮助召回的关系：人物-组织、人物-地点、人物-事件、动机-行动、原因-结果、身份-伪装、敌对/合作、前文伏笔-后文揭示。
3. relation 用短中文短语，例如：所属、亲属、上下级、师徒、合作、敌对、保护、追查、导致、发生在、使用、提及、关联、决定、动机、原因、结果、揭示、伪装、背叛、阻止、前往、来自。
4. 本 batch 目标关系数：至少 {target_relations} 条；如果原文信息足够，优先输出更多关系。
5. evidence_doc_ids 必须来自输入 chunks 的 metadata.doc_id；禁止输出短 id，如 "chunk-0001"。
6. 单 chunk 支撑就填 1 个 doc_id，跨 chunk 支撑就填 2-5 个 doc_id。
7. evidence 控制在 80 字以内，字符串内部不要换行。
8. 禁止使用输入之外的设定脑补关系。
9. 输出必须是单个 JSON 对象，且只有 `entities` 和 `relations` 两个字段。
10. 禁止把 doc_id 当 JSON key；实体必须写成数组项 `["doc_id","实体名","类型",["别名"]]`。

输入 chunks：
{chunk_blocks}
"""


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_name(value: str, *, max_len: int = 64) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return (cleaned or "unknown")[:max_len]


def batch_label(batch_id: int | str) -> str:
    if isinstance(batch_id, int):
        return f"{batch_id:05d}"
    return safe_name(str(batch_id), max_len=32)


def batch_file_stem(batch_id: int | str, batch: list[dict[str, Any]]) -> str:
    first = safe_name(str(batch[0].get("id") or "first")) if batch else "empty"
    last = safe_name(str(batch[-1].get("id") or "last")) if batch else "empty"
    return f"batch-{batch_label(batch_id)}-{first}-{last}"


def load_existing_doc_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(payload.get("doc_id") or "").strip()
            if doc_id:
                doc_ids.add(doc_id)
            for item in payload.get("doc_ids") or []:
                doc_id = str(item or "").strip()
                if doc_id:
                    doc_ids.add(doc_id)
    return doc_ids


def load_doc_ids_from_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                doc_ids.add(text)
                continue
            if isinstance(payload, dict):
                for doc_id in payload.get("doc_ids") or []:
                    if str(doc_id).strip():
                        doc_ids.add(str(doc_id).strip())
                doc_id = str(payload.get("doc_id") or "").strip()
                if doc_id:
                    doc_ids.add(doc_id)
            elif isinstance(payload, str) and payload.strip():
                doc_ids.add(payload.strip())
    return doc_ids


def load_low_relation_doc_ids(path: Path, *, threshold: int) -> set[str]:
    if not path.exists():
        return set()
    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "batch_id" not in payload:
                continue
            relations = payload.get("relations") or []
            if len(relations) <= threshold:
                for doc_id in payload.get("doc_ids") or []:
                    if str(doc_id).strip():
                        doc_ids.add(str(doc_id).strip())
    return doc_ids


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def pack_by_prompt_chars(
    items: list[dict[str, Any]],
    *,
    max_prompt_chars: int,
    max_batch_size: int,
    max_chunk_chars: int,
) -> list[list[dict[str, Any]]]:
    """Greedily pack chunks by prompt character budget.

    MiniMax's 200k context is token-based, but CJK tokenization varies by API.
    Character budget is a conservative proxy that keeps requests large without
    hard-coding tokenizer-specific logic.
    """
    if max_prompt_chars <= 0:
        return chunked(items, max_batch_size)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = len(GRAPH_PROMPT_TEMPLATE) + 1024
    for doc in items:
        rendered = format_chunk(doc, len(current) + 1, max_chars=max_chunk_chars)
        rendered_chars = len(rendered) + 2
        if (
            current
            and (len(current) >= max_batch_size or current_chars + rendered_chars > max_prompt_chars)
        ):
            batches.append(current)
            current = []
            current_chars = len(GRAPH_PROMPT_TEMPLATE) + 1024
            rendered = format_chunk(doc, 1, max_chars=max_chunk_chars)
            rendered_chars = len(rendered) + 2
        current.append(doc)
        current_chars += rendered_chars
    if current:
        batches.append(current)
    return batches


def format_chunk(doc: dict[str, Any], index: int, *, max_chars: int) -> str:
    text = str(doc.get("clean_text") or doc.get("search_text") or "")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    meta = {
        "doc_id": doc.get("id"),
        "activity_name": doc.get("activity_name"),
        "story_name": doc.get("story_name"),
        "stage_code": doc.get("stage_code"),
        "avg_tag": doc.get("avg_tag"),
    }
    return (
        f"[chunk {index}]\n"
        f"metadata: {json.dumps(meta, ensure_ascii=False)}\n"
        f"clean_text:\n{text}"
    )


def build_prompt(
    batch: list[dict[str, Any]],
    *,
    max_chunk_chars: int,
    target_relations: int,
    compact_schema: bool = False,
) -> str:
    chunk_blocks = "\n\n".join(
        format_chunk(doc, index, max_chars=max_chunk_chars)
        for index, doc in enumerate(batch, start=1)
    )
    template = GRAPH_COMPACT_PROMPT_TEMPLATE if compact_schema else GRAPH_PROMPT_TEMPLATE
    return template.format(
        batch_size=len(batch),
        target_relations=target_relations,
        chunk_blocks=chunk_blocks,
    )


def batch_text_chars(batch: list[dict[str, Any]], *, max_chunk_chars: int) -> int:
    total = 0
    for doc in batch:
        text = str(doc.get("clean_text") or doc.get("search_text") or "")
        total += min(len(text), max_chunk_chars)
    return total


def compute_target_relations(
    batch: list[dict[str, Any]],
    *,
    max_chunk_chars: int,
    relations_per_100_chunks: int,
    relation_chars_per_target: int,
) -> int:
    chunk_target = round(len(batch) * relations_per_100_chunks / 100)
    if relation_chars_per_target > 0:
        char_target = round(batch_text_chars(batch, max_chunk_chars=max_chunk_chars) / relation_chars_per_target)
        target = min(chunk_target, char_target)
    else:
        target = chunk_target
    if len(batch) <= 40:
        lower_bound = 12
    elif len(batch) <= 120:
        lower_bound = 40
    else:
        lower_bound = 60
    return max(lower_bound, min(420, target))


def normalize_entity_item(entity: Any) -> dict[str, Any] | None:
    if isinstance(entity, dict):
        name = str(entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "other").strip() or "other"
        raw_aliases = entity.get("aliases") or []
    elif isinstance(entity, list) and len(entity) >= 1:
        name = str(entity[0] or "").strip()
        entity_type = str(entity[1] if len(entity) >= 2 else "other").strip() or "other"
        raw_aliases = entity[2] if len(entity) >= 3 else []
    else:
        return None
    if len(name) < 2:
        return None
    aliases = [
        str(alias).strip()
        for alias in (raw_aliases if isinstance(raw_aliases, list) else [])
        if str(alias).strip()
    ][:8]
    return {"name": name, "type": entity_type, "aliases": aliases}


def normalize_chunk_item(item: Any, valid_doc_ids: set[str]) -> dict[str, Any] | None:
    if isinstance(item, dict):
        doc_id = str(item.get("doc_id") or "").strip()
        raw_entities = item.get("entities") or []
    elif isinstance(item, list) and len(item) >= 1:
        doc_id = str(item[0] or "").strip()
        raw_entities = item[1] if len(item) >= 2 else []
    else:
        return None
    if doc_id not in valid_doc_ids:
        return None
    entities = []
    for entity in raw_entities or []:
        normalized = normalize_entity_item(entity)
        if normalized is not None:
            entities.append(normalized)
    return {"doc_id": doc_id, "entities": entities, "relations": []}


def normalize_entities_by_doc(payload: dict[str, Any], valid_doc_ids: set[str]) -> list[dict[str, Any]]:
    entities_by_doc = payload.get("entities_by_doc")
    if not isinstance(entities_by_doc, dict):
        return []
    output: list[dict[str, Any]] = []
    for doc_id, raw_entities in entities_by_doc.items():
        normalized_doc_id = str(doc_id or "").strip()
        if normalized_doc_id not in valid_doc_ids:
            continue
        entities = []
        for entity in raw_entities or []:
            normalized = normalize_entity_item(entity)
            if normalized is not None:
                entities.append(normalized)
        output.append({"doc_id": normalized_doc_id, "entities": entities, "relations": []})
    return output


def normalize_flat_entities(payload: dict[str, Any], valid_doc_ids: set[str]) -> list[dict[str, Any]]:
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        return []
    entities_by_doc: dict[str, list[dict[str, Any]]] = {}
    for item in raw_entities:
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id") or "").strip()
            entity = normalize_entity_item(item)
        elif isinstance(item, list) and len(item) >= 2:
            doc_id = str(item[0] or "").strip()
            entity = normalize_entity_item(item[1:])
        else:
            continue
        if doc_id not in valid_doc_ids or entity is None:
            continue
        entities_by_doc.setdefault(doc_id, []).append(entity)
    return [
        {"doc_id": doc_id, "entities": entities, "relations": []}
        for doc_id, entities in entities_by_doc.items()
    ]


def normalize_relation_item(relation: Any, valid_doc_ids: set[str]) -> dict[str, Any] | None:
    if isinstance(relation, dict):
        head = str(relation.get("head") or "").strip()
        rel = str(relation.get("relation") or "").strip()
        tail = str(relation.get("tail") or "").strip()
        raw_doc_ids = relation.get("evidence_doc_ids") or relation.get("doc_ids") or []
        evidence = str(relation.get("evidence") or "").strip()
        fallback_doc_id = str(relation.get("doc_id") or relation.get("evidence_id") or "").strip()
    elif isinstance(relation, list) and len(relation) >= 3:
        head = str(relation[0] or "").strip()
        rel = str(relation[1] or "").strip()
        tail = str(relation[2] or "").strip()
        raw_doc_ids = relation[3] if len(relation) >= 4 else []
        evidence = str(relation[4] if len(relation) >= 5 else "").strip()
        fallback_doc_id = ""
    else:
        return None
    if not head or not rel or not tail:
        return None
    if isinstance(raw_doc_ids, str):
        stripped = raw_doc_ids.strip()
        if stripped.startswith("["):
            try:
                parsed_doc_ids = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_doc_ids = [stripped]
            raw_doc_ids = parsed_doc_ids
        else:
            raw_doc_ids = [stripped]
    evidence_doc_ids = [
        str(doc_id).strip()
        for doc_id in (raw_doc_ids if isinstance(raw_doc_ids, list) else [])
        if str(doc_id).strip() in valid_doc_ids
    ][:8]
    if not evidence_doc_ids and fallback_doc_id in valid_doc_ids:
        evidence_doc_ids = [fallback_doc_id]
    return {
        "head": head,
        "relation": rel,
        "tail": tail,
        "evidence_doc_ids": evidence_doc_ids,
        "evidence": evidence[:160],
    }


def parse_teacher_json_for_graph(raw_text: str, *, compact_schema: bool) -> dict[str, Any]:
    try:
        return parse_teacher_json(raw_text)
    except Exception:
        if not compact_schema:
            raise
    candidate = raw_text.strip()
    # Common MiniMax compact-schema errors:
    # 1. Flat entity entries after the first one miss their opening '['.
    # 2. The model transitions from entities to relations without closing the entities array.
    # 3. The model sometimes emits an extra ']' between flat entity rows.
    # 4. The model sometimes omits the closing ']' for a flat entity row.
    repaired = _repair_compact_teacher_json(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    repaired = COMPACT_RELATIONS_KEY_RE.sub(r'],"relations":', repaired, count=1)
    repaired = _repair_compact_teacher_json(repaired)
    return json.loads(repaired)


def normalize_payload(
    payload: dict[str, Any],
    batch: list[dict[str, Any]],
    *,
    batch_id: int | str,
    require_chunks: bool,
    compact_schema: bool,
) -> list[dict[str, Any]]:
    valid_doc_ids = {str(doc.get("id") or "") for doc in batch}
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) and require_chunks:
        raise ValueError("teacher payload must contain chunks list")
    output: list[dict[str, Any]] = []
    if compact_schema:
        output.extend(normalize_flat_entities(payload, valid_doc_ids))
        output.extend(normalize_entities_by_doc(payload, valid_doc_ids))
    if isinstance(chunks, list):
        for item in chunks:
            normalized_chunk = normalize_chunk_item(item, valid_doc_ids)
            if normalized_chunk is not None:
                output.append(normalized_chunk)

    batch_relations = []
    for relation in payload.get("relations") or []:
        normalized_relation = normalize_relation_item(relation, valid_doc_ids)
        if normalized_relation is not None:
            batch_relations.append(normalized_relation)
    output.append(
        {
            "batch_id": f"batch-{batch_label(batch_id)}",
            "doc_ids": [str(doc.get("id") or "") for doc in batch],
            "relations": batch_relations,
        }
    )
    return output


def run_batch(
    batch_id: int | str,
    batch: list[dict[str, Any]],
    *,
    api_config: TeacherApiConfig,
    max_chunk_chars: int,
    relations_per_100_chunks: int,
    relation_chars_per_target: int,
    raw_dir: Path,
    min_relation_yield_ratio: float,
    api_retries: int,
    retry_sleep: float,
    request_interval: float,
    compact_schema: bool,
) -> dict[str, Any]:
    started = time.time()
    target_relations = compute_target_relations(
        batch,
        max_chunk_chars=max_chunk_chars,
        relations_per_100_chunks=relations_per_100_chunks,
        relation_chars_per_target=relation_chars_per_target,
    )
    user_prompt = build_prompt(
        batch,
        max_chunk_chars=max_chunk_chars,
        target_relations=target_relations,
        compact_schema=compact_schema,
    )
    print(
        f"[start] batch={batch_label(batch_id)} docs={len(batch)} "
        f"prompt_chars={len(user_prompt)} target_relations={target_relations}",
        flush=True,
    )
    last_exc: Exception | None = None
    for attempt in range(1, max(1, api_retries) + 1):
        try:
            wait_for_request_slot(request_interval)
            raw_text, raw_payload = call_teacher_api(
                api_config,
                system_prompt=GRAPH_PROMPT_SYSTEM,
                user_prompt=user_prompt,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            retryable = any(token in str(exc) for token in ("HTTPError 429", "URLError", "timed out", "upstream_error"))
            if not retryable or attempt >= max(1, api_retries):
                raise
            sleep_seconds = retry_sleep * attempt
            print(
                f"[api-retry] batch={batch_label(batch_id)} attempt={attempt}/{api_retries} "
                f"sleep={sleep_seconds:.1f}s error={exc}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(f"API failed without exception: {last_exc}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_stem = batch_file_stem(batch_id, batch)
    (raw_dir / f"{raw_stem}.raw.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (raw_dir / f"{raw_stem}.response.txt").write_text(raw_text, encoding="utf-8")
    try:
        parsed = parse_teacher_json_for_graph(raw_text, compact_schema=compact_schema)
    except Exception as exc:
        preview = raw_text[:240].replace("\n", "\\n")
        raise ValueError(
            f"Model response was not valid JSON: {exc}; response_preview={preview!r}; "
            f"response_file={raw_dir / f'{raw_stem}.response.txt'}"
        ) from exc
    records = normalize_payload(
        parsed,
        batch,
        batch_id=batch_id,
        require_chunks=not compact_schema,
        compact_schema=compact_schema,
    )
    relation_count = sum(
        len(item.get("relations") or [])
        for item in records
        if item.get("batch_id")
    )
    min_relation_count = (
        max(8, round(target_relations * min_relation_yield_ratio))
        if min_relation_yield_ratio > 0
        else 0
    )
    if min_relation_count > 0 and relation_count < min_relation_count:
        raise ValueError(
            f"relation yield too low: {relation_count}/{target_relations} "
            f"(min={min_relation_count})"
        )
    return {
        "batch_id": batch_id,
        "records": records,
        "raw_text": raw_text,
        "raw_payload": raw_payload,
        "target_relations": target_relations,
        "latency_seconds": time.time() - started,
        "split_batches": 1,
    }


def should_split_failure(exc: Exception) -> bool:
    message = str(exc)
    if "contains non-latin-1 characters" in message or "codec can't encode" in message:
        return False
    if "contains only thinking blocks" in message or "No text in anthropic messages payload" in message:
        return False
    if "HTTPError 429" in message or "upstream_error" in message:
        return False
    if "Teacher API HTTPError" in message and "429" in message:
        return False
    return True


def run_batch_with_split_retry(
    batch_id: int | str,
    batch: list[dict[str, Any]],
    *,
    api_config: TeacherApiConfig,
    max_chunk_chars: int,
    relations_per_100_chunks: int,
    relation_chars_per_target: int,
    raw_dir: Path,
    split_retries: int,
    min_retry_batch_size: int,
    min_relation_yield_ratio: float,
    api_retries: int,
    retry_sleep: float,
    request_interval: float,
    compact_schema: bool,
) -> dict[str, Any]:
    try:
        return run_batch(
            batch_id,
            batch,
            api_config=api_config,
            max_chunk_chars=max_chunk_chars,
            relations_per_100_chunks=relations_per_100_chunks,
            relation_chars_per_target=relation_chars_per_target,
            raw_dir=raw_dir,
            min_relation_yield_ratio=min_relation_yield_ratio,
            api_retries=api_retries,
            retry_sleep=retry_sleep,
            request_interval=request_interval,
            compact_schema=compact_schema,
        )
    except Exception as exc:
        if (
            not should_split_failure(exc)
            or split_retries <= 0
            or len(batch) <= max(1, min_retry_batch_size)
        ):
            raise
        midpoint = len(batch) // 2
        if midpoint <= 0:
            raise
        left = batch[:midpoint]
        right = batch[midpoint:]
        left_id = f"{batch_label(batch_id)}a"
        right_id = f"{batch_label(batch_id)}b"
        print(
            f"[retry-split] batch={batch_label(batch_id)} docs={len(batch)} "
            f"error={exc}; retry as {left_id}({len(left)}) + {right_id}({len(right)})",
            flush=True,
        )
        left_result = run_batch_with_split_retry(
            left_id,
            left,
            api_config=api_config,
            max_chunk_chars=max_chunk_chars,
            relations_per_100_chunks=relations_per_100_chunks,
            relation_chars_per_target=relation_chars_per_target,
            raw_dir=raw_dir,
            split_retries=split_retries - 1,
            min_retry_batch_size=min_retry_batch_size,
            min_relation_yield_ratio=min_relation_yield_ratio,
            api_retries=api_retries,
            retry_sleep=retry_sleep,
            request_interval=request_interval,
            compact_schema=compact_schema,
        )
        right_result = run_batch_with_split_retry(
            right_id,
            right,
            api_config=api_config,
            max_chunk_chars=max_chunk_chars,
            relations_per_100_chunks=relations_per_100_chunks,
            relation_chars_per_target=relation_chars_per_target,
            raw_dir=raw_dir,
            split_retries=split_retries - 1,
            min_retry_batch_size=min_retry_batch_size,
            min_relation_yield_ratio=min_relation_yield_ratio,
            api_retries=api_retries,
            retry_sleep=retry_sleep,
            request_interval=request_interval,
            compact_schema=compact_schema,
        )
        return {
            "batch_id": batch_id,
            "records": left_result["records"] + right_result["records"],
            "raw_text": "",
            "raw_payload": {},
            "target_relations": left_result["target_relations"] + right_result["target_relations"],
            "latency_seconds": left_result["latency_seconds"] + right_result["latency_seconds"],
            "split_batches": left_result["split_batches"] + right_result["split_batches"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MiniRAG entity/relation annotations with a teacher API.")
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/processed/minirag_graph_annotations/chunk_graph_annotations.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/processed/minirag_graph_annotations/raw"))
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("data/processed/minirag_graph_annotations/failed_batches.jsonl"),
        help="Final failed batches are appended here with doc_ids for targeted retry.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Number of documents to process; 0 means all.")
    parser.add_argument(
        "--doc-ids-file",
        type=Path,
        default=None,
        help="Optional plain-text or JSONL file containing doc_ids to process.",
    )
    parser.add_argument(
        "--retry-low-relations-from",
        type=Path,
        default=None,
        help="Existing annotation JSONL; only docs from low-relation batch records are processed.",
    )
    parser.add_argument(
        "--low-relation-threshold",
        type=int,
        default=0,
        help="Used with --retry-low-relations-from; batches with relation count <= threshold are selected.",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=170000,
        help=(
            "Greedy packing budget for one request. Use 160000-180000 for a "
            "200k-context teacher; set <=0 to use fixed --batch-size only."
        ),
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--relations-per-100-chunks", type=int, default=70)
    parser.add_argument(
        "--relation-chars-per-target",
        type=int,
        default=250,
        help=(
            "Cap target relations by text density: target <= total_input_text_chars / this value. "
            "Use 0 to disable the character-density cap."
        ),
    )
    parser.add_argument(
        "--min-relation-yield-ratio",
        type=float,
        default=0.25,
        help="Treat a parsed response as failed if relation_count < target_relations * this ratio.",
    )
    parser.add_argument(
        "--split-retries",
        type=int,
        default=2,
        help="On JSON/API failure, recursively split a batch this many times and retry.",
    )
    parser.add_argument(
        "--min-retry-batch-size",
        type=int,
        default=40,
        help="Do not split failed batches at or below this number of chunks.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--api-type",
        choices=("responses", "chat_completions", "anthropic_messages"),
        default="responses",
    )
    parser.add_argument("--api-base", default=os.environ.get("TEACHER_API_BASE", "https://api.svips.org"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--auth-header", choices=("bearer", "x-api-key", "both"), default="bearer")
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "MiniMax-M2.7-highspeed"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=60000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--anthropic-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Anthropic-compatible JSON extraction, request no thinking blocks when supported by the provider.",
    )
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between API request starts across all worker threads.",
    )
    parser.add_argument(
        "--compact-schema",
        action="store_true",
        help="Ask for top-level entities_by_doc + relations instead of nested chunks. More stable for long outputs.",
    )
    parser.add_argument(
        "--relations-only",
        action="store_true",
        help="Deprecated alias for --compact-schema.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents_path = resolve_path(args.documents)
    output_path = resolve_path(args.output)
    raw_dir = resolve_path(args.raw_dir)
    failed_output_path = resolve_path(args.failed_output)
    docs = read_jsonl(documents_path)
    selected_doc_ids: set[str] | None = None
    if args.doc_ids_file is not None:
        selected_doc_ids = load_doc_ids_from_file(resolve_path(args.doc_ids_file))
    if args.retry_low_relations_from is not None:
        low_relation_doc_ids = load_low_relation_doc_ids(
            resolve_path(args.retry_low_relations_from),
            threshold=args.low_relation_threshold,
        )
        selected_doc_ids = (
            low_relation_doc_ids
            if selected_doc_ids is None
            else selected_doc_ids & low_relation_doc_ids
        )
    if selected_doc_ids is not None:
        docs = [doc for doc in docs if str(doc.get("id") or "") in selected_doc_ids]
    docs = docs[args.start :]
    if args.limit > 0:
        docs = docs[: args.limit]
    if args.resume:
        done = load_existing_doc_ids(output_path)
        docs = [doc for doc in docs if str(doc.get("id") or "") not in done]
    batches = pack_by_prompt_chars(
        docs,
        max_prompt_chars=args.max_prompt_chars,
        max_batch_size=args.batch_size,
        max_chunk_chars=args.max_chunk_chars,
    )
    print(
        json.dumps(
            {
                "documents": len(docs),
                "batches": len(batches),
                "max_batch_size": args.batch_size,
                "max_prompt_chars": args.max_prompt_chars,
                "max_chunk_chars": args.max_chunk_chars,
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        for batch_id, batch in enumerate(batches[:3], start=1):
            target_relations = compute_target_relations(
                batch,
                max_chunk_chars=args.max_chunk_chars,
                relations_per_100_chunks=args.relations_per_100_chunks,
                relation_chars_per_target=args.relation_chars_per_target,
            )
            prompt = build_prompt(
                batch,
                max_chunk_chars=args.max_chunk_chars,
                target_relations=target_relations,
                compact_schema=args.compact_schema or args.relations_only,
            )
            print(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "prompt_chars": len(prompt),
                        "docs": len(batch),
                        "target_relations": target_relations,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    api_config = TeacherApiConfig(
        api_type=args.api_type,
        base_url=args.api_base,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        json_mode=True,
        auth_header=args.auth_header,
        anthropic_disable_thinking=args.anthropic_disable_thinking,
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(
                run_batch_with_split_retry,
                batch_id,
                batch,
                api_config=api_config,
                max_chunk_chars=args.max_chunk_chars,
                relations_per_100_chunks=args.relations_per_100_chunks,
                relation_chars_per_target=args.relation_chars_per_target,
                raw_dir=raw_dir,
                split_retries=args.split_retries,
                min_retry_batch_size=args.min_retry_batch_size,
                min_relation_yield_ratio=args.min_relation_yield_ratio,
                api_retries=args.api_retries,
                retry_sleep=args.retry_sleep,
                request_interval=args.request_interval,
                compact_schema=args.compact_schema or args.relations_only,
            ): (batch_id, batch)
            for batch_id, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            batch_id, _batch = futures[future]
            try:
                result = future.result()
                append_jsonl(output_path, result["records"])
                completed += 1
                relation_count = sum(
                    len(item.get("relations") or [])
                    for item in result["records"]
                    if item.get("batch_id")
                )
                entity_count = sum(len(item.get("entities") or []) for item in result["records"])
                print(
                    f"[done] batch={batch_id}/{len(batches)} records={len(result['records'])} "
                    f"entities={entity_count} relations={relation_count}/{result['target_relations']} "
                    f"split_batches={result['split_batches']} latency={result['latency_seconds']:.1f}s "
                    f"completed={completed} failed={failed}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                append_jsonl_record(
                    failed_output_path,
                    {
                        "batch_id": batch_id,
                        "doc_ids": [str(doc.get("id") or "") for doc in _batch],
                        "doc_count": len(_batch),
                        "error": str(exc),
                        "first_doc_id": str(_batch[0].get("id") or "") if _batch else "",
                        "last_doc_id": str(_batch[-1].get("id") or "") if _batch else "",
                    },
                )
                print(
                    f"[failed] batch={batch_id}/{len(batches)} error={exc} "
                    f"completed={completed} failed={failed} failed_output={failed_output_path}",
                    flush=True,
                )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
