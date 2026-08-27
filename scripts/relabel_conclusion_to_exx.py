#!/usr/bin/env python3
"""Re-label legacy conclusion prompts with full visible evidence and E-only citations.

The teacher receives only the question, retrieval hypothesis, round and the
hydrated candidate evidence.  Legacy answers and metadata fields such as
``gold`` or ``answer_focus`` are never sent to the API.  Outputs are written to
a fresh directory and are resumable JSONL files; source datasets are read-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = Path(__file__).with_name("convert_grounded_training_to_exx.py")
SPEC = importlib.util.spec_from_file_location("asa_exx_converter", CONVERTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import converter: {CONVERTER_PATH}")
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)

PROTOCOL = CONVERTER.PROTOCOL
SYSTEM_PROMPT = CONVERTER.SYSTEM_PROMPT
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
ALLOWED_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
DOC_PREFIXES = ("[uc]info/", "[uc]info\\")


@dataclasses.dataclass
class Task:
    task_id: str
    split: str
    question: str
    hypothesis: str
    round_value: str
    evidence: list[Any]
    source_refs: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "round": self.round_value,
            "evidence": [dataclasses.asdict(item) for item in self.evidence],
            "source_refs": self.source_refs,
        }


class LabelError(RuntimeError):
    pass


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    line = compact_json(value) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def source_files(source_dirs: list[Path]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for source_dir in source_dirs:
        for split in ("train", "val", "test"):
            path = source_dir / f"{split}.json"
            if path.exists():
                files.append((path.resolve(), split))
    return files


def doc_id_variants(doc_id: str) -> list[str]:
    """Return exact id first, followed by legacy namespace fallbacks."""
    exact = doc_id.strip()
    variants = [exact]
    uc_prefix = "[uc]info/"
    if exact.startswith(uc_prefix):
        variants.append(exact[len(uc_prefix) :])
    else:
        variants.append(uc_prefix + exact)
    return list(dict.fromkeys(item for item in variants if item))


def hydrate_documents(documents_path: Path, needed_ids: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    candidate_ids = {variant for needed in needed_ids for variant in doc_id_variants(needed)}
    document_text: dict[str, str] = {}
    scanned = 0
    duplicate_exact_ids: Counter[str] = Counter()
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(row.get("id", "")).strip()
            if doc_id not in candidate_ids:
                continue
            text = str(row.get("clean_text") or row.get("search_text") or "").strip()
            if not text:
                continue
            if doc_id in document_text and document_text[doc_id] != text:
                duplicate_exact_ids[doc_id] += 1
                continue
            document_text[doc_id] = text
    hydrated: dict[str, str] = {}
    fallback_resolutions: dict[str, str] = {}
    for needed in needed_ids:
        variants = doc_id_variants(needed)
        if variants[0] in document_text:
            hydrated[needed] = document_text[variants[0]]
            continue
        fallback = next((variant for variant in variants[1:] if variant in document_text), None)
        if fallback:
            hydrated[needed] = document_text[fallback]
            fallback_resolutions[needed] = fallback
    report = {
        "documents_path": str(documents_path),
        "documents_sha256": sha256_file(documents_path),
        "documents_scanned": scanned,
        "needed_doc_ids": len(needed_ids),
        "hydrated_doc_ids": len(hydrated),
        "missing_doc_ids": sorted(needed_ids - hydrated.keys()),
        "fallback_resolutions": fallback_resolutions,
        "duplicate_exact_id_conflicts": dict(duplicate_exact_ids),
    }
    return hydrated, report


def canonical_task_key(question: str, hypothesis: str, round_value: str, evidence: list[Any]) -> str:
    payload = {
        "question": question,
        "hypothesis": hypothesis,
        "round": round_value,
        "doc_ids": [item.doc_id for item in evidence],
    }
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def collect_tasks(
    source_dirs: list[Path], documents_path: Path
) -> tuple[list[Task], list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    needed_ids: set[str] = set()
    parse_rejects: list[dict[str, Any]] = []

    for path, split in source_files(source_dirs):
        rows = read_json(path)
        input_manifest.append(
            {"path": str(path), "split": split, "records": len(rows), "sha256": sha256_file(path)}
        )
        for record in rows:
            if record.get("task_type") != "conclusion_generation":
                continue
            try:
                prompt = record["conversations"][0]["value"]
                question, hypothesis, round_value, evidence = CONVERTER.parse_short_prompt(prompt)
            except Exception as exc:
                parse_rejects.append(
                    {
                        "id": record.get("id"),
                        "source_path": str(path),
                        "split": split,
                        "reason": f"prompt_parse_failed:{type(exc).__name__}:{exc}",
                    }
                )
                continue
            if not question or not evidence:
                parse_rejects.append(
                    {
                        "id": record.get("id"),
                        "source_path": str(path),
                        "split": split,
                        "reason": "empty_question_or_evidence",
                    }
                )
                continue
            needed_ids.update(item.doc_id for item in evidence)
            raw_rows.append(
                {
                    "source_path": str(path),
                    "source_id": str(record.get("id", "")),
                    "split": split,
                    "question": question,
                    "hypothesis": hypothesis,
                    "round": round_value,
                    "evidence": evidence,
                }
            )

    hydrated, hydration_report = hydrate_documents(documents_path, needed_ids)
    complete_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        missing = [item.doc_id for item in row["evidence"] if item.doc_id not in hydrated]
        if missing:
            parse_rejects.append(
                {
                    "id": row["source_id"],
                    "source_path": row["source_path"],
                    "split": row["split"],
                    "reason": "full_evidence_not_found",
                    "missing_doc_ids": missing,
                }
            )
            continue
        row["evidence"] = [
            CONVERTER.Evidence(item.label, item.doc_id, hydrated[item.doc_id]) for item in row["evidence"]
        ]
        row["task_id"] = canonical_task_key(
            row["question"], row["hypothesis"], row["round"], row["evidence"]
        )[:24]
        complete_rows.append(row)

    # A question may have multiple evidence packs.  If any version is held out,
    # all versions of that question stay out of train to avoid leakage.
    question_split: dict[str, str] = {}
    for row in complete_rows:
        question = row["question"]
        split = row["split"]
        if question not in question_split or SPLIT_PRIORITY[split] > SPLIT_PRIORITY[question_split[question]]:
            question_split[question] = split

    grouped: dict[str, dict[str, Any]] = {}
    for row in complete_rows:
        task_id = row["task_id"]
        if task_id not in grouped:
            grouped[task_id] = row | {"source_refs": []}
        grouped[task_id]["source_refs"].append(
            {"path": row["source_path"], "id": row["source_id"], "split": row["split"]}
        )
    tasks: list[Task] = []
    for task_id, row in grouped.items():
        tasks.append(
            Task(
                task_id=task_id,
                split=question_split[row["question"]],
                question=row["question"],
                hypothesis=row["hypothesis"],
                round_value=row["round"],
                evidence=row["evidence"],
                source_refs=row["source_refs"],
            )
        )
    tasks.sort(key=lambda item: (SPLIT_PRIORITY[item.split], item.task_id))
    report = {
        "inputs": input_manifest,
        "legacy_conclusion_rows": len(raw_rows),
        "unique_full_evidence_tasks": len(tasks),
        "task_splits": dict(Counter(item.split for item in tasks)),
        "parse_or_hydration_rejects": len(parse_rejects),
        "hydration": hydration_report,
        "policy": {
            "uses_legacy_assistant_output": False,
            "uses_gold_metadata": False,
            "teacher_inputs": ["question", "hypothesis", "round", "full_candidate_evidence"],
            "train_val_test_question_isolation": True,
        },
    }
    return tasks, parse_rejects, report


def teacher_prompt(task: Task) -> str:
    evidence_text = "\n".join(f"[{item.label}]\n{item.text}" for item in task.evidence)
    return "\n".join(
        (
            "你是《明日方舟》剧情问答系统的证据动作标注员。",
            "只根据下方当前可见候选证据判断，不得使用你记忆中的游戏知识，不得假设候选之外的事实。",
            f"问题：{task.question}",
            f"当前检索假设：{task.hypothesis or '{}'}",
            f"轮次：{task.round_value or 'unknown'}",
            "候选证据（全文）：",
            evidence_text,
            "",
            f"输出协议：{PROTOCOL}",
            "只输出一个JSON对象，禁止markdown和解释。",
            "若证据足够回答，输出：",
            '{"next_action":"answer_directly","supported_facts":[{"fact":"可直接用于回答问题的原子事实","evidence_ids":["E2"]}]}',
            "若证据不足且轮次尚未用尽，输出：",
            '{"next_action":"retrieve_more","follow_up_hypothesis":{"question":"...","query_type":"fact","entities":[],"keywords":[],"expected_answer_type":"...","dialogue_context":""}}',
            "若轮次已用尽且仍不足，输出：",
            '{"next_action":"abstain","reason":"现有证据不足以确认。"}',
            "严格规则：",
            "1. answer_directly时，supported_facts合起来必须完整回答问题的全部核心信息需求；若关键部分缺证据，不得只回答其中一半，未到最大轮次就retrieve_more，已到最大轮次就abstain。",
            "特别注意：当问题询问“为什么某种超常现象会发生”时，仅证明该现象确实发生不等于解释了原因；证据未说明机制或原因时，不得把现象本身当作原因。",
            "2. 只保留回答问题必需的可核验事实；每条fact只表达一个可独立判断真假的断言，不能用“且、并、因此”把多个独立断言塞进一条。",
            "3. 每条fact绑定1至2个证据编号，编号只能来自上方当前可见的E编号。",
            "4. evidence_ids表示对应证据的正文在语义上直接支持该fact，不是只要主题相关即可。",
            "5. 不输出quote，不复制引文，不输出final_answer、answer、inferred_facts、confidence或missing_slots。",
            "6. 候选已足够时不要过度retrieve_more；候选不支持关键答案时不要猜测或勉强answer_directly。",
            "7. 问题中的“真实身份、究竟、真正”等措辞不代表必然存在隐藏身份；若证据已明确普通身份和目的，就据证据作答，不要为了寻找更深秘密而补检索。",
            "8. 不按连接词、标点或疑问词机械拆题；只按问题的核心信息需求和证据实际支持的事实组织supported_facts。",
            "9. 输出前在内部逐条复核：每个fact是否被所列E正文直接支持、所有fact合起来是否完整回答问题、是否存在不必要弃答或补检索；不要输出复核过程。",
        )
    )


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise LabelError("response_has_no_json_object")
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LabelError(f"response_invalid_json:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise LabelError("response_json_not_object")
    return value


def validate_label(payload: dict[str, Any], task: Task) -> dict[str, Any]:
    action = payload.get("next_action")
    if action not in ALLOWED_ACTIONS:
        raise LabelError("invalid_next_action")
    serialized = compact_json(payload)
    for forbidden in ("quote", "final_answer", "inferred_facts", '"answer"', "missing_slots"):
        if forbidden in serialized:
            raise LabelError(f"forbidden_field:{forbidden}")
    visible_ids = {item.label for item in task.evidence}
    if action == "answer_directly":
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not facts or len(facts) > 8:
            raise LabelError("invalid_supported_facts")
        clean_facts: list[dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                raise LabelError("fact_not_object")
            text = str(fact.get("fact", "")).strip()
            ids = fact.get("evidence_ids")
            if not text or not isinstance(ids, list) or not 1 <= len(ids) <= 2:
                raise LabelError("fact_or_evidence_ids_invalid")
            normalized_ids: list[str] = []
            for evidence_id in ids:
                normalized = str(evidence_id).strip().strip("[]").upper()
                if normalized not in visible_ids:
                    raise LabelError(f"evidence_id_not_visible:{normalized}")
                if normalized not in normalized_ids:
                    normalized_ids.append(normalized)
            clean_facts.append({"fact": text, "evidence_ids": normalized_ids})
        return {"next_action": action, "supported_facts": clean_facts}
    if action == "retrieve_more":
        round_match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", task.round_value or "")
        if round_match and int(round_match.group(1)) >= int(round_match.group(2)):
            raise LabelError("retrieve_more_at_max_round")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict) or not str(follow_up.get("question", "")).strip():
            raise LabelError("retrieve_more_missing_follow_up")
        return {"next_action": action, "follow_up_hypothesis": follow_up}
    return {"next_action": "abstain", "reason": "现有证据不足以确认。"}


def semantic_verifier_prompt(task: Task, proposed: dict[str, Any]) -> str:
    evidence_text = "\n".join(f"[{item.label}]\n{item.text}" for item in task.evidence)
    return "\n".join(
        (
            "你是第二位独立的RAG标注审核员。请只依据当前可见证据审核候选标签并给出最终修正版。",
            f"问题：{task.question}",
            f"轮次：{task.round_value or 'unknown'}",
            "当前可见证据（全文）：",
            evidence_text,
            "候选标签：",
            compact_json(proposed),
            "只输出JSON：",
            '{"valid":true,"issues":[],"label":{...grounded_action_exx_v1最终标签...}}',
            "审核要求：",
            "1. 对每个fact逐条检查：其所列每个E正文必须在语义上直接支持该断言；主题相关不等于支持。",
            "2. answer_directly的facts合起来必须覆盖问题全部核心信息需求；只支持部分答案时应改为retrieve_more或最大轮次abstain。",
            "特别注意：仅证明某现象发生，不能回答该现象为什么发生；若证据没有给出机制/原因，必须按轮次retrieve_more或abstain。",
            "3. 证据已足够时必须纠正过度retrieve_more/abstain；不得因“真实、究竟、真正”等措辞臆测存在隐藏信息。",
            "4. 每个fact只含一个可独立判断真假的断言，绑定1至2个当前存在的E编号。",
            "5. 最终label不含quote、answer、final_answer、inferred_facts、confidence或missing_slots。",
            "6. 不得参考游戏常识或候选之外的信息。issues只写简短错误标签，不写思维过程。",
        )
    )


def semantic_verify(
    task: Task, proposed: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, api_meta = api_request(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        prompt=semantic_verifier_prompt(task, proposed),
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    wrapper = parse_json_object(raw)
    label = wrapper.get("label")
    if not isinstance(label, dict):
        raise LabelError("semantic_verifier_missing_label")
    final_label = validate_label(label, task)
    return final_label, {
        "valid": bool(wrapper.get("valid")),
        "issues": wrapper.get("issues") if isinstance(wrapper.get("issues"), list) else [],
        "api": api_meta,
    }


def api_request(
    *, endpoint: str, api_key: str, model: str, prompt: str, timeout: float, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的RAG证据标注员，只输出JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    # The servers often inherit stale localhost proxy variables.  Labelling
    # must use a direct connection unless an explicit opener is supplied.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise LabelError(f"http_{exc.code}:{detail}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise LabelError(f"network_error:{exc}") from exc
    try:
        content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LabelError("api_response_missing_content") from exc
    return str(content), {
        "id": response_body.get("id"),
        "model": response_body.get("model"),
        "usage": response_body.get("usage"),
        "finish_reason": response_body.get("choices", [{}])[0].get("finish_reason"),
    }


def label_one(task: Task, args: argparse.Namespace) -> dict[str, Any]:
    prompt = teacher_prompt(task)
    last_error = ""
    for attempt in range(1, args.max_attempts + 1):
        try:
            raw, api_meta = api_request(
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt if attempt == 1 else prompt + f"\n上一轮格式校验失败：{last_error}\n请重新输出完整合法JSON。",
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
            first_pass = validate_label(parse_json_object(raw), task)
            payload = first_pass
            verifier_meta = None
            if args.semantic_verify:
                payload, verifier_meta = semantic_verify(task, first_pass, args)
            return {
                "task_id": task.task_id,
                "split": task.split,
                "label": payload,
                "first_pass_label": first_pass,
                "attempt": attempt,
                "api": api_meta,
                "semantic_verifier": verifier_meta,
                "source_refs": task.source_refs,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if attempt < args.max_attempts:
                time.sleep(min(2**attempt, 8))
    raise LabelError(last_error)


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id") and isinstance(row.get("label"), dict):
                completed[row["task_id"]] = row
    return completed


def build_sft(output_dir: Path, tasks: list[Task], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_map = {task.task_id: task for task in tasks}
    for task_id, result in labels.items():
        task = task_map.get(task_id)
        if task is None:
            continue
        record = {
            "id": f"relabel-{task.task_id}",
            "task_type": "grounded_action_generation",
            "bucket": "teacher_relabel_exx",
            "system": SYSTEM_PROMPT,
            "tools": [],
            "conversations": [
                {
                    "from": "human",
                    "value": CONVERTER.render_prompt(
                        task.question, task.hypothesis, task.round_value, task.evidence
                    ),
                },
                {"from": "gpt", "value": compact_json(result["label"])},
            ],
            "meta": {
                "schema": PROTOCOL,
                "teacher_model": result.get("api", {}).get("model"),
                "conversion_method": "teacher_visible_full_evidence_only",
                "source_refs": task.source_refs,
            },
        }
        CONVERTER.validate_record(record)
        rows_by_split[task.split].append(record)
    sft_dir = output_dir / "sft"
    sft_dir.mkdir(exist_ok=True)
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda item: item["id"])
        write_json(sft_dir / f"{split}.json", rows)
    dataset_info = {}
    for split in rows_by_split:
        dataset_info[f"teacher_relabel_exx_v1_{split}"] = {
            "file_name": f"{split}.json",
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
        }
    write_json(sft_dir / "dataset_info.json", dataset_info)
    return {split: len(rows) for split, rows in rows_by_split.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--documents-path", type=Path, default=PROJECT_ROOT / "indexes/arknights_story/documents.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--include-split", action="append", choices=("train", "val", "test"), default=[])
    parser.add_argument("--no-semantic-verify", dest="semantic_verify", action="store_false")
    parser.set_defaults(semantic_verify=True)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be in [1,16]")
    args.api_key = os.environ.get(args.api_key_env, "").strip()
    if not args.prepare_only and not args.api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, rejects, prepare_report = collect_tasks(
        [path.resolve() for path in args.source_dir], args.documents_path.resolve()
    )
    write_json(output_dir / "tasks.json", [task.as_dict() for task in tasks])
    write_json(output_dir / "prepare_audit.json", prepare_report)
    with (output_dir / "prepare_rejects.jsonl").open("w", encoding="utf-8") as handle:
        for reject in rejects:
            handle.write(compact_json(reject) + "\n")
    if args.prepare_only:
        print(json.dumps(prepare_report, ensure_ascii=False, indent=2))
        return 0

    labels_path = output_dir / "labels.jsonl"
    failures_path = output_dir / "failures.jsonl"
    completed = load_completed(labels_path)
    pending = [task for task in tasks if task.task_id not in completed]
    if args.include_split:
        selected_splits = set(args.include_split)
        pending = [task for task in pending if task.split in selected_splits]
    if args.task_id:
        requested_ids = set(args.task_id)
        pending = [task for task in tasks if task.task_id in requested_ids]
    if args.limit > 0:
        pending = pending[: args.limit]
    lock = threading.Lock()
    run_stats: Counter[str] = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(label_one, task, args): task for task in pending}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                run_stats["failed"] += 1
                append_jsonl(
                    failures_path,
                    {"task_id": task.task_id, "error": f"{type(exc).__name__}:{exc}"},
                    lock,
                )
            else:
                append_jsonl(labels_path, result, lock)
                completed[task.task_id] = result
                run_stats["completed"] += 1
                run_stats[f"action:{result['label']['next_action']}"] += 1
            done = run_stats["completed"] + run_stats["failed"]
            if done % 10 == 0 or done == len(pending):
                print(
                    compact_json(
                        {
                            "done_this_run": done,
                            "pending_this_run": len(pending),
                            "total_completed": len(completed),
                            "stats": dict(run_stats),
                        }
                    ),
                    flush=True,
                )

    output_counts = build_sft(output_dir, tasks, completed)
    audit = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_requested": args.model,
        "endpoint": args.endpoint,
        "workers": args.workers,
        "semantic_verify": args.semantic_verify,
        "prepare": prepare_report,
        "completed_labels": len(completed),
        "pending_labels": len(tasks) - len(completed),
        "run_stats": dict(run_stats),
        "sft_outputs": output_counts,
    }
    write_json(output_dir / "audit.json", audit)
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    write_json(
        output_dir / "manifest.json",
        {
            "protocol": PROTOCOL,
            "created_at_utc": audit["created_at_utc"],
            "files": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_files
            ],
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
