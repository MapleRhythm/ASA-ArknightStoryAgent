#!/usr/bin/env python3
"""Recalibrate Exx v3 SFT gold evidence bindings with independent GLM judges.

The run mode creates a deterministic fact manifest and appends one JSONL
judgement row per fact.  Each fact receives one merged-evidence judgement and
one additional judgement per cited E-ID.  The finalize mode applies the
strict policy to the original JSON datasets and writes clean JSONL datasets,
change logs and a machine-readable report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE = "${REMOTE_STORE}/exx_grounding_v1/data/exx_grounding_v3_sft_minimal559_glm53_20260903"
DEFAULT_OUTDIR = "${REMOTE_STORE}/exx_grounding_v1/data/gold_recalibrated_20260904"
ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"
PROXY_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)
EVIDENCE_HEADER_RE = re.compile(r"(?m)^\[(E\d+)\]\s*(?:\n|$)")
STOP_LINE_RE = re.compile(r"(?m)^(?:output_schema|action_set|rules):")
QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(\d+)\s*/\s*(\d+)\s*$")

for _name in PROXY_NAMES:
    os.environ.pop(_name, None)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def parse_evidence(prompt: str) -> dict[str, str]:
    marker = re.search(r"(?m)^evidence:\s*(?:\n|$)", prompt)
    if not marker:
        return {}
    body = prompt[marker.end() :]
    stop = STOP_LINE_RE.search(body)
    if stop:
        body = body[: stop.start()]
    matches = list(EVIDENCE_HEADER_RE.finditer(body))
    evidence: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            evidence[match.group(1)] = text
    return evidence


def parse_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt)
    return match.group(1).strip() if match else ""


def parse_query_type(prompt: str) -> str:
    hypothesis = re.search(r"(?m)^hypothesis:\s*(\{.*\})\s*$", prompt)
    if hypothesis:
        try:
            value = json.loads(hypothesis.group(1))
            if isinstance(value, dict) and value.get("query_type"):
                return str(value["query_type"])
        except Exception:
            pass
    match = re.search(r'"query_type"\s*:\s*"([^"]+)"', prompt)
    return match.group(1) if match else "unknown"


def parse_round(prompt: str) -> tuple[int, int]:
    match = ROUND_RE.search(prompt)
    return (int(match.group(1)), int(match.group(2))) if match else (1, 1)


def question_key(row: dict[str, Any]) -> str | None:
    meta = row.get("meta")
    if isinstance(meta, str):
        try:
            value = json.loads(meta)
            if isinstance(value, dict) and value.get("question_key") is not None:
                return str(value["question_key"])
        except Exception:
            pass
    if isinstance(meta, dict) and meta.get("question_key") is not None:
        return str(meta["question_key"])
    return None


def fact_id(split: str, row: dict[str, Any], fact_index: int, claim: str, eids: list[str]) -> str:
    raw = "\0".join(
        [split, str(row.get("id") or ""), str(fact_index), claim, ",".join(eids)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def extract_facts(split: str, path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON array")
    tasks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("task_type") != "grounded_action_generation":
            continue
        conversations = row.get("conversations") or []
        if not conversations:
            continue
        prompt = str((conversations[0] or {}).get("value") or "")
        try:
            output = json.loads(str((conversations[-1] or {}).get("value") or ""))
        except Exception:
            continue
        if not isinstance(output, dict) or output.get("next_action") != "answer_directly":
            continue
        evidence = parse_evidence(prompt)
        if not evidence:
            continue
        for index, fact in enumerate(output.get("supported_facts") or []):
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("fact") or "").strip()
            eids = [str(eid) for eid in fact.get("evidence_ids") or [] if str(eid) in evidence]
            if not claim or not eids:
                continue
            fid = fact_id(split, row, index, claim, eids)
            tasks.append(
                {
                    "fact_id": fid,
                    "split": split,
                    "row_id": row.get("id"),
                    "question_key": question_key(row),
                    "fact_index": index,
                    "query": parse_question(prompt),
                    "query_type": parse_query_type(prompt),
                    "claim": claim,
                    "cited_eids": eids,
                    "cited_texts": {eid: evidence[eid] for eid in eids},
                    "original_action": output.get("next_action"),
                    "round": parse_round(prompt),
                }
            )
    return rows, tasks


def all_manifests(base: str, outdir: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    all_tasks: list[dict[str, Any]] = []
    source_rows: dict[str, list[dict[str, Any]]] = {}
    split_tasks: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = str(Path(base) / f"{split}.json")
        if not Path(path).exists():
            source_rows[split] = []
            split_tasks[split] = []
            continue
        rows, tasks = extract_facts(split, path)
        source_rows[split] = rows
        split_tasks[split] = tasks
        all_tasks.extend(tasks)
        print(f"{split}: source_rows={len(rows)} fact_rows={len(tasks)}", flush=True)
    with open(Path(outdir) / "facts_manifest.jsonl", "w", encoding="utf-8") as handle:
        for task in all_tasks:
            handle.write(compact(task) + "\n")
    return all_tasks, source_rows, split_tasks


def stratified_sample(tasks: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    pool = [task for task in tasks if task.get("split") == "train"]
    if count >= len(pool):
        return pool
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in pool:
        groups[str(task.get("query_type") or "unknown")].append(task)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    names = sorted(groups)
    quotas = {name: 1 for name in names if groups[name]}
    while sum(quotas.values()) > count:
        name = max(quotas, key=lambda n: (quotas[n], len(groups[n]), n))
        if quotas[name] > 1:
            quotas[name] -= 1
        else:
            del quotas[name]
    raw = {name: count * len(groups[name]) / len(pool) for name in names}
    while sum(quotas.values()) < count:
        candidates = [name for name in names if name not in quotas or quotas[name] < len(groups[name])]
        name = max(candidates, key=lambda n: (raw[n] - quotas.get(n, 0), len(groups[n]), n))
        quotas[name] = quotas.get(name, 0) + 1
    selected: list[dict[str, Any]] = []
    for name in names:
        selected.extend(groups[name][: quotas.get(name, 0)])
    rng.shuffle(selected)
    return selected[:count]


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")


def parse_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content or "")


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if "```" in raw:
        pieces = raw.split("```")
        raw = next((piece.strip() for piece in pieces if piece.strip().startswith("{")), raw)
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize_support(value: Any, raw: str = "") -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "supported", "support", "entailed", "partial", "支持"}:
        return True
    if text in {"false", "no", "unsupported", "contradicted", "not_supported", "不支持"}:
        return False
    first = str(raw).strip().lower()[:100]
    if "不支持" in first or "unsupported" in first or "contradicted" in first:
        return False
    if "支持" in first or "supported" in first or "entailed" in first:
        return True
    return None


def judge_messages(claim: str, evidence_text: str, merged: bool) -> list[dict[str, str]]:
    system = (
        "你是严格、独立的证据绑定审核员。只能依据给出的证据文本判断，"
        "不能使用外部知识；同主题但没有直接证明主张核心事实时必须判不支持。"
    )
    label = "剧情证据（被引证据合并）" if merged else "剧情证据"
    user = (
        "请判断下面这段【剧情证据】是否直接支持这条【主张】。\n"
        "支持的标准：证据必须直接证明主张的核心事实、主体以及关键关系；"
        "仅背景相关、同人物/同主题、缺少关键断言，或与主张相反，都判不支持。\n\n"
        f"主张：{claim}\n\n"
        f"{label}：\n{evidence_text}\n\n"
        "只输出一个 JSON 对象，不要输出 Markdown。格式："
        '{"supports":true或false,"reason":"一句中文理由"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_judge(
    api_key: str,
    claim: str,
    evidence_text: str,
    *,
    merged: bool,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": judge_messages(claim, evidence_text, merged),
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "medium",
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl_context())
    )
    last_error = ""
    terminal_code = ""
    for attempt in range(1, attempts + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError("no_choices")
            content = parse_content((choices[0].get("message") or {}).get("content"))
            parsed = parse_json_object(content)
            if parsed is None:
                raise RuntimeError("invalid_json")
            support = normalize_support(parsed.get("supports", parsed.get("verdict")), content)
            if support is None:
                raise RuntimeError("unparseable_support")
            return {
                "verdict": "supported" if support else "unsupported",
                "supports": support,
                "reason": str(parsed.get("reason") or "")[:1000],
                "raw_content": content[:4000],
                "attempts": attempt,
                "status": "ok",
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            last_error = f"HTTP {exc.code}: {detail}"
            business_code = ""
            try:
                decoded = json.loads(detail)
                business_code = str((decoded.get("error") or {}).get("code") or decoded.get("code") or "")
            except (json.JSONDecodeError, AttributeError):
                pass
            terminal_code = business_code or str(exc.code)
            if attempt < attempts and (exc.code == 429 or business_code == "1301"):
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
                continue
            if business_code == "1301":
                return {
                    "verdict": "skip",
                    "supports": None,
                    "reason": "content_filter_1301",
                    "raw_content": detail,
                    "attempts": attempt,
                    "status": "skip",
                }
            if exc.code in {400, 401, 402, 403}:
                break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    return {
        "verdict": "error",
        "supports": None,
        "reason": (last_error or f"terminal:{terminal_code}")[:1000],
        "raw_content": "",
        "attempts": attempts,
        "status": "error",
    }


def merged_text(task: dict[str, Any], chunk_chars: int = 600) -> str:
    return "\n\n".join(
        f"[{eid}]\n{str(task['cited_texts'][eid])[:chunk_chars]}"
        for eid in task["cited_eids"]
    )


def judge_task(
    task: dict[str, Any],
    api_key: str,
    timeout: float,
    attempts: int,
    chunk_chars: int,
) -> dict[str, Any]:
    merged = call_judge(
        api_key,
        task["claim"],
        merged_text(task, chunk_chars),
        merged=True,
        timeout=timeout,
        attempts=attempts,
    )
    individual: dict[str, Any] = {}
    for eid in task["cited_eids"]:
        individual[eid] = call_judge(
            api_key,
            task["claim"],
            f"[{eid}]\n{str(task['cited_texts'][eid])[:chunk_chars]}",
            merged=False,
            timeout=timeout,
            attempts=attempts,
        )
    return {
        "fact_id": task["fact_id"],
        "split": task["split"],
        "row_id": task["row_id"],
        "question_key": task["question_key"],
        "fact_index": task["fact_index"],
        "query_type": task["query_type"],
        "claim": task["claim"],
        "cited_eids": task["cited_eids"],
        "merged": merged,
        "individual": individual,
        "judge_model": MODEL,
        "reasoning_effort": "medium",
        "chunk_chars": chunk_chars,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


def complete_judgement(row: dict[str, Any]) -> bool:
    merged = row.get("merged") or {}
    if merged.get("status") == "error":
        return False
    for value in (row.get("individual") or {}).values():
        if isinstance(value, dict) and value.get("status") == "error":
            return False
    return True


def agreement_smoke_tasks(
    all_tasks: list[dict[str, Any]],
    pairs_path: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample facts from the same 250-row positive pool used by AGREEMENT."""
    if not pairs_path or not Path(pairs_path).exists():
        return stratified_sample(all_tasks, count, seed)
    pairs = read_jsonl(pairs_path)
    positive = [pair for pair in pairs if pair.get("side") == "positive"]
    lookup = {
        (str(task.get("row_id")), int(task.get("fact_index"))): task for task in all_tasks
    }
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in positive:
        key = (str(pair.get("record_id")), int(pair.get("fact_index")))
        task = lookup.get(key)
        if task and task["fact_id"] not in seen:
            task = dict(task)
            task["agreement_reference"] = {
                "pair_id": pair.get("pair_id"),
                "evidence_id": pair.get("evidence_id"),
                "chunk_text": pair.get("chunk_text"),
                "glm_source_verdict": pair.get("source_glm_verdict"),
            }
            candidates.append(task)
            seen.add(task["fact_id"])
    if len(candidates) < count:
        return stratified_sample(all_tasks, count, seed)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def run_judgements(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_tasks, _, split_tasks = all_manifests(args.base, str(outdir))
    if args.mode == "smoke":
        tasks = agreement_smoke_tasks(
            all_tasks, args.agreement_pairs, args.smoke_facts, args.seed
        )
        manifest_name = "facts_smoke.jsonl"
        output_name = "judgements_smoke.jsonl"
    else:
        tasks = all_tasks
        manifest_name = "facts_full.jsonl"
        output_name = "judgements_full.jsonl"
    with open(outdir / manifest_name, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(compact(task) + "\n")
    output_path = outdir / output_name
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(output_path)):
        fid = str(row.get("fact_id") or "")
        if fid and complete_judgement(row):
            existing[fid] = row
    pending = [task for task in tasks if task["fact_id"] not in existing]
    api_key = os.environ.get("BIGMODEL_API_KEY", "")
    if not api_key:
        raise SystemExit("BIGMODEL_API_KEY is missing; run after 'source ~/.bigmodel_env'")
    print(
        f"mode={args.mode} facts={len(tasks)} existing={len(existing)} "
        f"pending={len(pending)} workers={min(3, max(1, args.workers))}",
        flush=True,
    )
    lock = threading.Lock()
    with open(output_path, "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, max(1, args.workers))) as pool:
            futures = [
                pool.submit(
                    judge_task,
                    task,
                    api_key,
                    args.timeout,
                    args.attempts,
                    args.chunk_chars,
                )
                for task in pending
            ]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                with lock:
                    handle.write(compact(row) + "\n")
                    handle.flush()
                    existing[str(row["fact_id"])] = row
                if count == 1 or count % 10 == 0 or count == len(futures):
                    print(f"GLM progress {count}/{len(futures)}", flush=True)
    counts = Counter()
    for row in existing.values():
        counts[str((row.get("merged") or {}).get("verdict") or "missing")] += 1
    summary = {
        "mode": args.mode,
        "facts": len(tasks),
        "written_complete": len(existing),
        "merged_supported": counts.get("supported", 0),
        "merged_unsupported": counts.get("unsupported", 0),
        "merged_skip": counts.get("skip", 0),
        "merged_error": counts.get("error", 0),
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


def classify_judgement(judgement: dict[str, Any]) -> dict[str, Any]:
    merged = judgement.get("merged") or {}
    merged_verdict = merged.get("verdict")
    individual = judgement.get("individual") or {}
    eids = [str(eid) for eid in judgement.get("cited_eids") or []]
    supported_eids = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "supported"]
    unsupported_eids = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "unsupported"]
    uncertain_eids = [
        eid for eid in eids if (individual.get(eid) or {}).get("verdict") in {"skip", "error", None}
    ]
    if merged_verdict == "unsupported":
        status = "unsupported"
        kept = []
    elif merged_verdict == "supported":
        if uncertain_eids:
            # Strict fallback: retain only individually confirmed IDs.
            status = "clean_support" if not supported_eids and len(eids) == 1 and not unsupported_eids else "partial"
            kept = supported_eids
            if not kept:
                status = "unsupported"
        elif unsupported_eids:
            status = "partial" if supported_eids else "unsupported"
            kept = supported_eids
        else:
            status = "clean_support"
            kept = eids
    else:
        status = "unsupported"
        kept = []
    return {
        "status": status,
        "merged_verdict": merged_verdict,
        "kept_eids": kept,
        "dropped_eids": [eid for eid in eids if eid not in kept],
        "unsupported_eids": unsupported_eids,
        "uncertain_eids": uncertain_eids,
        "merged_reason": str(merged.get("reason") or ""),
        "individual_reasons": {
            eid: str((individual.get(eid) or {}).get("reason") or "") for eid in eids
        },
    }


def action_for_empty_facts(prompt: str) -> tuple[str, dict[str, Any]]:
    current, total = parse_round(prompt)
    question = parse_question(prompt)
    hypothesis: dict[str, Any] = {}
    match = re.search(r"(?m)^hypothesis:\s*(\{.*\})\s*$", prompt)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                hypothesis = parsed
        except json.JSONDecodeError:
            pass
    follow_up = {
        key: hypothesis.get(key, default)
        for key, default in (
            ("query_type", "fact"),
            ("entities", []),
            ("keywords", []),
            ("expected_answer_type", ""),
            ("dialogue_context", ""),
        )
    }
    follow_up["question"] = (
        f"{question}（请检索能直接证明问题核心事实的证据）"
        if question
        else "请检索能直接证明原问题核心事实的证据"
    )
    if current < total:
        return "retrieve_more", {"follow_up_hypothesis": follow_up}
    return "abstain", {"reason": "现有证据不足以确认。"}


def update_output_row(
    row: dict[str, Any],
    prompt: str,
    new_facts: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    conversations = row.get("conversations") or []
    if not conversations:
        return row, "unknown", "unknown"
    try:
        output = json.loads(str((conversations[-1] or {}).get("value") or ""))
    except Exception:
        return row, "unknown", "unknown"
    original_action = str(output.get("next_action") or "")
    if original_action != "answer_directly":
        return row, original_action, original_action
    updated = dict(output)
    if new_facts:
        updated["next_action"] = "answer_directly"
        updated["supported_facts"] = new_facts
    else:
        next_action, payload = action_for_empty_facts(prompt)
        updated = {"next_action": next_action}
        updated.update(payload)
    new_conversations = [dict(item) if isinstance(item, dict) else item for item in conversations]
    last = dict(new_conversations[-1])
    last["value"] = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
    new_conversations[-1] = last
    result = dict(row)
    result["conversations"] = new_conversations
    return result, original_action, str(updated.get("next_action") or "")


def finalize(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_tasks, source_rows, split_tasks = all_manifests(args.base, str(outdir))
    judgements: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(Path(args.judgements))):
        fid = str(row.get("fact_id") or "")
        if fid:
            judgements[fid] = row
    task_by_id = {task["fact_id"]: task for task in all_tasks}
    classifications: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for task in all_tasks:
        fid = task["fact_id"]
        if fid not in judgements:
            missing.append(task)
            continue
        classifications[fid] = classify_judgement(judgements[fid])
    if missing:
        print(f"WARNING missing judgements={len(missing)}", flush=True)

    changes: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "reasoning_effort": "medium",
        "strict_prompt": {
            "system": "你是严格、独立的证据绑定审核员。只能依据给出的证据文本判断，不能使用外部知识；同主题但没有直接证明主张核心事实时必须判不支持。",
            "criterion": "证据必须直接证明主张的核心事实、主体以及关键关系；仅背景相关、同人物/同主题、缺少关键断言，或与主张相反，都判不支持。",
        },
        "source": {},
        "splits": {},
        "missing_judgements": len(missing),
        "unsupported_examples": [],
    }
    combined_rows: list[dict[str, Any]] = []
    global_counts = Counter()
    global_after = 0
    global_before = 0
    global_multi = 0
    global_partial_multi = 0
    global_action_flips = 0
    for split in ("train", "val", "test"):
        rows = source_rows.get(split, [])
        tasks = split_tasks.get(split, [])
        task_lookup: dict[tuple[Any, int], dict[str, Any]] = {
            (task["row_id"], int(task["fact_index"])): task for task in tasks
        }
        row_classifications: dict[Any, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] ] = defaultdict(list)
        for task in tasks:
            cls = classifications.get(task["fact_id"])
            if cls is not None:
                row_classifications[task["row_id"]].append((task, cls, judgements[task["fact_id"]]))
        strict_rows: list[dict[str, Any]] = []
        split_counts = Counter()
        split_qtype_total = Counter()
        split_qtype_unsupported = Counter()
        split_before = 0
        split_after = 0
        split_multi = 0
        split_partial_multi = 0
        split_action_flips = 0
        split_action_before = Counter()
        split_action_after = Counter()
        for row in rows:
            row_id = row.get("id")
            conversations = row.get("conversations") or []
            prompt = str((conversations[0] or {}).get("value") or "") if conversations else ""
            try:
                output = json.loads(str((conversations[-1] or {}).get("value") or ""))
            except Exception:
                output = {}
            original_facts = output.get("supported_facts") or [] if isinstance(output, dict) else []
            if output.get("next_action") != "answer_directly" or not original_facts:
                strict_row = row
                if isinstance(output, dict) and output.get("next_action"):
                    split_action_before[str(output.get("next_action"))] += 1
                    split_action_after[str(output.get("next_action"))] += 1
                strict_rows.append(strict_row)
                combined_rows.append(strict_row)
                continue
            new_facts: list[dict[str, Any]] = []
            row_entries = sorted(row_classifications.get(row_id, []), key=lambda item: int(item[0]["fact_index"]))
            for task, cls, raw_judgement in row_entries:
                split_before += 1
                global_before += 1
                global_counts[cls["status"]] += 1
                split_counts[cls["status"]] += 1
                split_qtype_total[str(task["query_type"])] += 1
                if cls["status"] == "unsupported":
                    split_qtype_unsupported[str(task["query_type"])] += 1
                if len(task["cited_eids"]) > 1:
                    split_multi += 1
                    global_multi += 1
                if cls["status"] == "partial" and len(task["cited_eids"]) > 1:
                    split_partial_multi += 1
                    global_partial_multi += 1
                if cls["kept_eids"]:
                    split_after += 1
                    global_after += 1
                    new_facts.append(
                        {
                            "fact": task["claim"],
                            "evidence_ids": cls["kept_eids"],
                        }
                    )
                changed = cls["status"] != "clean_support" or cls["kept_eids"] != task["cited_eids"]
                if changed:
                    changes.append(
                        {
                            "entity_type": "fact",
                            "change_type": "drop_fact" if not cls["kept_eids"] else "drop_evidence_ids",
                            "split": split,
                            "row_id": task["row_id"],
                            "question_key": task["question_key"],
                            "fact_id": task["fact_id"],
                            "fact_index": task["fact_index"],
                            "query_type": task["query_type"],
                            "claim": task["claim"],
                            "original_evidence_ids": task["cited_eids"],
                            "kept_evidence_ids": cls["kept_eids"],
                            "dropped_evidence_ids": cls["dropped_eids"],
                            "label": cls["status"],
                            "merged_verdict": cls["merged_verdict"],
                            "merged_reason": cls["merged_reason"],
                            "individual_reasons": cls["individual_reasons"],
                            "uncertain_evidence_ids": cls["uncertain_eids"],
                        }
                    )
                if cls["status"] == "unsupported" and len(report["unsupported_examples"]) < 10:
                    report["unsupported_examples"].append(
                        {
                            "split": split,
                            "row_id": task["row_id"],
                            "question_key": task["question_key"],
                            "fact_id": task["fact_id"],
                            "query_type": task["query_type"],
                            "claim": task["claim"],
                            "original_evidence_ids": task["cited_eids"],
                            "merged_reason": cls["merged_reason"],
                            "individual_reasons": cls["individual_reasons"],
                        }
                    )
            original_action = str(output.get("next_action") or "")
            split_action_before[original_action] += 1
            strict_row, before_action, after_action = update_output_row(row, prompt, new_facts)
            split_action_after[after_action] += 1
            combined_rows.append(strict_row)
            strict_rows.append(strict_row)
            if before_action == "answer_directly" and after_action != "answer_directly":
                split_action_flips += 1
                global_action_flips += 1
                changes.append(
                    {
                        "entity_type": "row",
                        "change_type": "action_flip",
                        "split": split,
                        "row_id": row_id,
                        "question_key": question_key(row),
                        "original_action": before_action,
                        "new_action": after_action,
                        "reason": "所有原 supported_facts 均被严格重判移除。",
                    }
                )
        strict_path = outdir / f"gold_recalibrated_v1_strict_{split}.jsonl"
        with open(strict_path, "w", encoding="utf-8") as handle:
            for row in strict_rows:
                handle.write(compact(row) + "\n")
        report["splits"][split] = {
            "source_rows": len(rows),
            "fact_total_before": split_before,
            "fact_total_after": split_after,
            "label_counts": dict(split_counts),
            "label_rates": {
                name: (split_counts.get(name, 0) / split_before if split_before else 0.0)
                for name in ("clean_support", "partial", "unsupported")
            },
            "multi_eid_facts": split_multi,
            "partial_multi_eid_facts": split_partial_multi,
            "partial_rate_among_multi_eid": split_partial_multi / split_multi if split_multi else 0.0,
            "unsupported_rate_by_query_type": {
                q: split_qtype_unsupported[q] / split_qtype_total[q]
                for q in sorted(split_qtype_total)
            },
            "fact_total_by_query_type": dict(split_qtype_total),
            "action_before": dict(split_action_before),
            "action_after": dict(split_action_after),
            "action_flip_count": split_action_flips,
            "strict_path": str(strict_path),
        }
    combined_path = outdir / "gold_recalibrated_v1_strict.jsonl"
    with open(combined_path, "w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(compact(row) + "\n")
    changes_path = outdir / "gold_recalibrated_changes.jsonl"
    with open(changes_path, "w", encoding="utf-8") as handle:
        for change in changes:
            handle.write(compact(change) + "\n")

    smoke_rows = read_jsonl(str(outdir / "judgements_smoke.jsonl"))
    smoke_counts = Counter()
    smoke_binding_counts = Counter()
    smoke_examples: list[dict[str, Any]] = []
    for judgement in smoke_rows:
        cls = classify_judgement(judgement)
        smoke_counts[cls["status"]] += 1
        for value in (judgement.get("individual") or {}).values():
            smoke_binding_counts[str(value.get("verdict") or "missing")] += 1
        if cls["status"] == "unsupported":
            smoke_examples.append(
                {
                    "fact_id": judgement.get("fact_id"),
                    "query_type": judgement.get("query_type"),
                    "claim": judgement.get("claim"),
                    "reason": cls["merged_reason"],
                }
            )
    smoke_total = sum(smoke_counts.values())
    smoke_binding_total = sum(smoke_binding_counts.values())
    report["smoke"] = {
        "fact_total": smoke_total,
        "label_counts": dict(smoke_counts),
        "fact_unsupported_rate": (
            smoke_counts.get("unsupported", 0) / smoke_total if smoke_total else None
        ),
        "binding_total": smoke_binding_total,
        "binding_label_counts": dict(smoke_binding_counts),
        "binding_unsupported_rate": (
            smoke_binding_counts.get("unsupported", 0) / smoke_binding_total
            if smoke_binding_total
            else None
        ),
        "expected_range": [0.40, 0.55],
        "strictness_cross_validation_pass": bool(
            smoke_binding_total
            and 0.40
            <= smoke_binding_counts.get("unsupported", 0) / smoke_binding_total
            <= 0.55
        ),
        "unsupported_examples": smoke_examples[:10],
    }
    report["source"] = {
        "base_dir": args.base,
        "splits": {split: str(Path(args.base) / f"{split}.json") for split in ("train", "val", "test")},
        "fact_total_before": global_before,
        "fact_total_after": global_after,
        "label_counts": dict(global_counts),
        "label_rates": {
            name: (global_counts.get(name, 0) / global_before if global_before else 0.0)
            for name in ("clean_support", "partial", "unsupported")
        },
        "multi_eid_facts": global_multi,
        "partial_multi_eid_facts": global_partial_multi,
        "partial_rate_among_multi_eid": global_partial_multi / global_multi if global_multi else 0.0,
        "action_flip_count": global_action_flips,
        "change_count": len(changes),
        "strict_combined_path": str(combined_path),
        "changes_path": str(changes_path),
    }
    report_path = outdir / "gold_recalibrated_20260904.report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["smoke"], ensure_ascii=False, sort_keys=True), flush=True)
    print(json.dumps(report["source"], ensure_ascii=False, sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full", "finalize"), required=True)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--judgements", default="")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--smoke-facts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--agreement-pairs",
        default="${REMOTE_STORE}/exx_grounding_v1/eval/verifier_glm_agreement_20260904/pairs_full.jsonl",
    )
    parser.add_argument("--chunk-chars", type=int, default=600)
    args = parser.parse_args()
    if args.mode in {"smoke", "full"}:
        run_judgements(args)
    else:
        if not args.judgements:
            raise SystemExit("--judgements is required for --mode finalize")
        finalize(args)


if __name__ == "__main__":
    main()
