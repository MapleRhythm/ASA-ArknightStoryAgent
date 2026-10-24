#!/usr/bin/env python3
"""Build dense + scene-aware claim/citation hard negatives and GLM labels.

The script is intentionally self-contained so it can be copied to the 248
worker.  It reads the visible evidence directly from the grounded SFT prompt,
embeds claims as queries and evidence chunks as documents with
Qwen3-Embedding-0.6B, gives same-scene candidates a deterministic priority,
then asks GLM once per record to classify the four selected candidates.

For records outside the GLM budget, selected candidates are retained with
``glm_verdict=not_run``.  For validated records, only GLM ``unsupported``
candidates remain in ``hard_negatives``; candidates judged supported are
reported separately as ``suspected_missed_positives``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_IN = "${REMOTE_STORE}/exx_grounding_v1/data/exx_grounding_v3_sft_minimal559_glm53_20260903/train.json"
DEFAULT_OUT = "${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl"
DEFAULT_REPORT = "${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.report.json"
DEFAULT_CACHE = "${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.glm_cache.jsonl"
DEFAULT_V1 = "${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v1_20260904.jsonl"
DEFAULT_MODEL = "${REMOTE_STORE}/rank_v8_coverage_241_mirror/Qwen3-Embedding-0.6B"
GLM_ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

EVIDENCE_HEADER_RE = re.compile(r"(?m)^\[(E\d+)\]\s*(?:\n|$)")
QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
QUERY_TYPE_RE = re.compile(r'"query_type"\s*:\s*"([^"]+)"')
STOP_LINE_RE = re.compile(r"(?m)^(?:output_schema|action_set|rules):")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def parse_scene(text: str) -> tuple[str | None, str | None]:
    """Return (source_name, scene_key) from a visible story header if present."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:8]:
        if "/" not in line or len(line) > 220 or ":" in line:
            continue
        parts = [part.strip() for part in line.split("/") if part.strip()]
        if len(parts) >= 2 and all(len(part) < 140 for part in parts):
            source = parts[0]
            scene = " / ".join(parts[:3])
            return source, scene
    return None, None


def question_and_type(prompt: str) -> tuple[str, str | None]:
    match = QUESTION_RE.search(prompt)
    question = match.group(1).strip() if match else ""
    qtype = None
    hypothesis = re.search(r"(?m)^hypothesis:\s*(\{.*\})\s*$", prompt)
    if hypothesis:
        try:
            value = json.loads(hypothesis.group(1))
            if isinstance(value, dict):
                qtype = value.get("query_type")
        except Exception:
            pass
    if not qtype:
        qtype_match = QUERY_TYPE_RE.search(prompt)
        qtype = qtype_match.group(1) if qtype_match else None
    return question, qtype


def load_records(input_path: str, limit_records: int) -> list[dict[str, Any]]:
    rows = json.load(open(input_path, encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task_type") != "grounded_action_generation":
            continue
        conversations = row.get("conversations") or []
        if not conversations:
            continue
        prompt = str(conversations[0].get("value") or "")
        try:
            output = json.loads(str(conversations[-1].get("value") or ""))
        except Exception:
            continue
        if output.get("next_action") != "answer_directly":
            continue
        evidence = parse_evidence(prompt)
        if not evidence:
            continue
        question, qtype = question_and_type(prompt)
        for fact_index, fact in enumerate(output.get("supported_facts") or []):
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("fact") or "").strip()
            gold = [str(eid) for eid in (fact.get("evidence_ids") or []) if str(eid) in evidence]
            if not claim or not gold:
                continue
            records.append(
                {
                    "record_id": row.get("id"),
                    "question_key": json.loads(row.get("meta", "{}")).get("question_key")
                    if isinstance(row.get("meta"), str)
                    else None,
                    "fact_index": fact_index,
                    "query": question,
                    "query_type": qtype,
                    "claim": claim,
                    "gold": gold,
                    "evidence": evidence,
                }
            )
        # Limit is in source records, not fact rows.  The source has several
        # supported_facts per record, so trim only after processing a record.
        if limit_records > 0:
            seen = len({r["record_id"] for r in records})
            if seen >= limit_records:
                break
    if limit_records > 0:
        allowed = {r["record_id"] for r in records}
        # Preserve all facts from the first N eligible source records.
        ordered: list[str] = []
        for r in records:
            if r["record_id"] not in ordered:
                ordered.append(r["record_id"])
        allowed = set(ordered[:limit_records])
        records = [r for r in records if r["record_id"] in allowed]
    return records


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    query: bool,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if query:
        kwargs["prompt_name"] = "query"
    # Avoid pathological prompts while keeping substantially more context than
    # the old v1 600-character output truncation.
    clipped = [text[:12000] for text in texts]
    return np.asarray(model.encode(clipped, **kwargs), dtype=np.float32)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def trigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", re.sub(r"\[[A-Za-z0-9]+\]", "", text or ""))
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)} if len(normalized) >= 3 else set()


def trigram_similarity(a: str, b: str) -> float:
    aa, bb = trigrams(a), trigrams(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def old_v1_baseline(path: str) -> dict[str, float | int | None]:
    values: list[float] = []
    if not Path(path).exists():
        return {"count": 0, "mean": None, "max": None}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for candidate in row.get("hard_negatives") or []:
                value = candidate.get("sim")
                if isinstance(value, (int, float)):
                    values.append(float(value))
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize_support(value: Any, verdict: Any = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or verdict or "").strip().lower()
    if text in {"true", "yes", "supported", "support", "entailed", "partial"}:
        return True
    if text in {"false", "no", "unsupported", "contradicted", "not_supported"}:
        return False
    return None


def glm_messages(claim: str, candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = {
        "claim": claim,
        "candidates": [{"eid": c["eid"], "text": c["text"]} for c in candidates],
    }
    example = {
        "verdicts": [
            {"eid": "E1", "supports": False, "reason": "仅同主题，未证明主张核心事实。"}
        ]
    }
    system = (
        "你是严格的证据绑定审核员。只依据每条给出的证据文本判断它是否支持主张，"
        "不得使用外部知识，也不能把同主题、同人物或背景相关误判为支持。"
    )
    user = "\n".join(
        [
            "主张：",
            claim,
            "候选证据（逐条独立判断）：",
            compact(payload["candidates"]),
            "规则：证据文本能直接证明主张的核心事实、主体和关键关系时 supports=true；"
            "仅相关但缺少关键断言、只提供背景、或与主张无关时 supports=false。"
            "只输出一个 JSON 对象，不要输出 Markdown。",
            compact(example),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_glm(
    api_key: str,
    claim: str,
    candidates: list[dict[str, Any]],
    *,
    timeout: float,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    body = {
        "model": "glm-5.3",
        "messages": glm_messages(claim, candidates),
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "medium",
    }
    request = urllib.request.Request(
        GLM_ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl_context())
    )
    last_error = ""
    for attempt in range(max_attempts):
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            parsed = parse_json_object(str(content))
            if not parsed:
                raise RuntimeError("invalid_json")
            raw_verdicts = parsed.get("verdicts")
            if not isinstance(raw_verdicts, list):
                raise RuntimeError("missing_verdicts")
            verdicts: dict[str, dict[str, Any]] = {}
            for item in raw_verdicts:
                if not isinstance(item, dict) or not item.get("eid"):
                    continue
                support = normalize_support(item.get("supports"), item.get("verdict"))
                if support is None:
                    continue
                verdicts[str(item["eid"])] = {
                    "glm_verdict": "supported" if support else "unsupported",
                    "glm_supports": support,
                    "glm_reason": str(item.get("reason") or "")[:500],
                }
            expected = {str(c["eid"]) for c in candidates}
            if set(verdicts) != expected:
                raise RuntimeError(f"missing_eids:{sorted(expected - set(verdicts))}")
            return verdicts
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            last_error = f"http_{exc.code}:{detail}"
            if exc.code in {400, 401, 402, 403}:
                break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last_error = str(exc)
        if attempt + 1 < max_attempts:
            time.sleep(2**attempt)
    return {
        str(c["eid"]): {
            "glm_verdict": "error",
            "glm_supports": None,
            "glm_reason": last_error[:500],
        }
        for c in candidates
    }


def cache_key(claim: str, candidates: list[dict[str, Any]]) -> str:
    value = {"claim": claim, "candidates": [{"eid": c["eid"], "text": c["text"]} for c in candidates]}
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def load_cache(path: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return cache
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("cache_key") and isinstance(row.get("verdicts"), dict):
                # A transient 429/network failure must not become a
                # permanent cache hit on a later rerun.
                verdicts = row["verdicts"]
                if any(
                    isinstance(value, dict) and value.get("glm_verdict") == "error"
                    for value in verdicts.values()
                ):
                    cache.pop(str(row["cache_key"]), None)
                else:
                    cache[str(row["cache_key"])] = verdicts
    return cache


def append_cache(path: str, lock: threading.Lock, key: str, verdicts: dict[str, Any]) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(compact({"cache_key": key, "verdicts": verdicts}) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_IN)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--glm-cache", default=DEFAULT_CACHE)
    parser.add_argument("--v1", default=DEFAULT_V1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--glm-record-limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--glm-workers", type=int, default=4)
    parser.add_argument("--glm-timeout", type=float, default=120.0)
    parser.add_argument("--glm-attempts", type=int, default=3)
    parser.add_argument("--skip-glm", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.glm_cache).parent.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input, args.limit_records)
    source_record_count = len({r["record_id"] for r in records})
    print(f"eligible source records={source_record_count} fact rows={len(records)}", flush=True)
    if not records:
        raise SystemExit("no eligible records")

    claims = list(dict.fromkeys(r["claim"] for r in records))
    documents = list(dict.fromkeys(text for r in records for text in r["evidence"].values()))
    print(f"embedding claims={len(claims)} unique evidence chunks={len(documents)}", flush=True)
    model = SentenceTransformer(args.model, device="cuda")
    claim_embeddings = embed_texts(model, claims, query=True, batch_size=args.batch_size)
    document_embeddings = embed_texts(model, documents, query=False, batch_size=args.batch_size)
    claim_index = {text: index for index, text in enumerate(claims)}
    document_index = {text: index for index, text in enumerate(documents)}

    selected_by_record: list[list[dict[str, Any]]] = []
    dense_values: list[float] = []
    scene_hits = 0
    source_hits = 0
    candidate_total = 0
    for record in records:
        gold = set(record["gold"])
        gold_sources: set[str] = set()
        gold_scenes: set[str] = set()
        for eid in gold:
            source, scene = parse_scene(record["evidence"][eid])
            if source:
                gold_sources.add(source)
            if scene:
                gold_scenes.add(scene)
        qvec = claim_embeddings[claim_index[record["claim"]]]
        candidates: list[dict[str, Any]] = []
        for eid, text in record["evidence"].items():
            if eid in gold:
                continue
            source, scene = parse_scene(text)
            same_scene = bool(scene and scene in gold_scenes)
            same_source = bool(source and source in gold_sources)
            sim = float(qvec @ document_embeddings[document_index[text]])
            candidates.append(
                {
                    "eid": eid,
                    "text": text,
                    "cos_sim": round(sim, 6),
                    "source_name": source,
                    "scene": scene,
                    "same_scene": same_scene,
                    "same_source": same_source,
                }
            )
        # Same scene is the strongest confusion signal; same source is the
        # second-best signal; dense cosine breaks ties within each tier.
        candidates.sort(
            key=lambda c: (
                0 if c["same_scene"] else 1 if c["same_source"] else 2,
                -c["cos_sim"],
                c["eid"],
            )
        )
        chosen = candidates[:4]
        for rank, candidate in enumerate(chosen, 1):
            candidate["selection_rank"] = rank
            candidate["selection_tier"] = (
                "same_scene"
                if candidate["same_scene"]
                else "same_source"
                if candidate["same_source"]
                else "dense_only"
            )
            candidate["text"] = candidate["text"][:2000]
            dense_values.append(float(candidate["cos_sim"]))
            scene_hits += int(candidate["same_scene"])
            source_hits += int(candidate["same_source"])
        candidate_total += len(chosen)
        selected_by_record.append(chosen)

    print(
        f"selected candidates={candidate_total} same_scene={scene_hits} same_source={source_hits}",
        flush=True,
    )

    # Validate the first N source records, as requested.  A record may have
    # multiple facts; the budget is counted in fact rows here and is capped by
    # the first N rows, which is conservative and keeps the smoke deterministic.
    if args.skip_glm or args.glm_record_limit == 0:
        validate_indices: set[int] = set()
    elif args.glm_record_limit < 0:
        validate_indices = set(range(len(records)))
    else:
        validate_indices = set(range(min(args.glm_record_limit, len(records))))
    api_key = os.environ.get("BIGMODEL_API_KEY", "")
    cache = load_cache(args.glm_cache)
    cache_lock = threading.Lock()
    glm_results: dict[int, dict[str, Any]] = {}
    if validate_indices and not api_key:
        print("BIGMODEL_API_KEY missing; GLM validation marked error", flush=True)

    def validate(index: int) -> tuple[int, dict[str, Any]]:
        chosen = selected_by_record[index]
        key = cache_key(records[index]["claim"], chosen)
        if key in cache:
            return index, cache[key]
        if not api_key:
            result = {
                c["eid"]: {
                    "glm_verdict": "error",
                    "glm_supports": None,
                    "glm_reason": "missing_BIGMODEL_API_KEY",
                }
                for c in chosen
            }
        else:
            result = call_glm(
                api_key,
                records[index]["claim"],
                chosen,
                timeout=args.glm_timeout,
                max_attempts=args.glm_attempts,
            )
        append_cache(args.glm_cache, cache_lock, key, result)
        return index, result

    if validate_indices:
        print(
            f"GLM validating {len(validate_indices)} fact rows with {args.glm_workers} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max(1, args.glm_workers)) as executor:
            futures = [executor.submit(validate, index) for index in sorted(validate_indices)]
            for done_count, future in enumerate(as_completed(futures), 1):
                index, result = future.result()
                glm_results[index] = result
                if done_count == 1 or done_count % 10 == 0 or done_count == len(futures):
                    print(f"GLM progress {done_count}/{len(futures)}", flush=True)

    report_rows: list[dict[str, Any]] = []
    glm_counter: Counter[str] = Counter()
    missed_positive_count = 0
    error_count = 0
    clean_hardneg_count = 0
    output_rows: list[dict[str, Any]] = []
    for index, (record, chosen) in enumerate(zip(records, selected_by_record)):
        validated = index in validate_indices
        verdicts = glm_results.get(index, {})
        hard_negatives: list[dict[str, Any]] = []
        missed: list[dict[str, Any]] = []
        for candidate in chosen:
            verdict = verdicts.get(candidate["eid"])
            if verdict is None:
                verdict = {
                    "glm_verdict": "not_run",
                    "glm_supports": None,
                    "glm_reason": "outside_glm_budget",
                }
            status = str(verdict.get("glm_verdict") or "error")
            glm_counter[status] += 1
            if status == "supported":
                missed.append(
                    {
                        "eid": candidate["eid"],
                        "text": candidate["text"],
                        "cos_sim": candidate["cos_sim"],
                        "same_scene": candidate["same_scene"],
                        "same_source": candidate["same_source"],
                        "glm_verdict": status,
                        "glm_reason": verdict.get("glm_reason", ""),
                    }
                )
                missed_positive_count += 1
                continue
            if status == "error":
                error_count += 1
                continue
            hard_negatives.append(
                {
                    "eid": candidate["eid"],
                    "text": candidate["text"],
                    "cos_sim": candidate["cos_sim"],
                    "same_scene": candidate["same_scene"],
                    "same_source": candidate["same_source"],
                    "selection_tier": candidate["selection_tier"],
                    "selection_rank": candidate["selection_rank"],
                    "glm_verdict": status,
                    "glm_reason": verdict.get("glm_reason", ""),
                }
            )
            if status == "unsupported":
                clean_hardneg_count += 1
        output_rows.append(
            {
                "record_id": record["record_id"],
                "question_key": record["question_key"],
                "fact_index": record["fact_index"],
                "query": record["query"],
                "query_type": record["query_type"],
                "claim": record["claim"],
                "positive_eids": record["gold"],
                "positive_texts": [record["evidence"][eid][:2000] for eid in record["gold"]],
                "hard_negatives": hard_negatives,
                "suspected_missed_positives": missed,
                "suspected_missed_positive_eids": [item["eid"] for item in missed],
                "glm_validation": {
                    "validated": validated,
                    "verdicts_requested": len(chosen) if validated else 0,
                },
            }
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline = old_v1_baseline(args.v1)
    validated_candidate_count = sum(
        len(selected_by_record[index]) for index in validate_indices
    )
    unsupported_validated = glm_counter["unsupported"]
    supported_validated = glm_counter["supported"]
    report = {
        "version": "binding_hardneg_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": args.input,
        "output_path": args.output,
        "source_record_count": source_record_count,
        "fact_row_count": len(records),
        "embedding_model": args.model,
        "selection": {
            "top_k": 4,
            "candidate_count": candidate_total,
            "same_scene_count": scene_hits,
            "same_scene_fraction": scene_hits / candidate_total if candidate_total else 0.0,
            "same_source_count": source_hits,
            "same_source_fraction": source_hits / candidate_total if candidate_total else 0.0,
            "dense_cosine": {
                "count": len(dense_values),
                "mean": float(np.mean(dense_values)) if dense_values else None,
                "p50": percentile(dense_values, 50),
                "p90": percentile(dense_values, 90),
                "min": float(np.min(dense_values)) if dense_values else None,
                "max": float(np.max(dense_values)) if dense_values else None,
            },
            "trigram_v1_baseline": baseline,
        },
        "glm": {
            "record_rows_validated": len(validate_indices),
            "candidate_count_validated": validated_candidate_count,
            "max_record_budget": args.glm_record_limit,
            "verdict_counts": dict(sorted(glm_counter.items())),
            "unsupported_count": unsupported_validated,
            "supported_count": supported_validated,
            "clean_unsupported_ratio": (
                unsupported_validated / validated_candidate_count
                if validated_candidate_count
                else None
            ),
            "suspected_missed_positive_count": missed_positive_count,
            "error_count": error_count,
            "cache_path": args.glm_cache,
        },
        "output": {
            "rows": len(output_rows),
            "hard_negative_rows": sum(bool(row["hard_negatives"]) for row in output_rows),
            "clean_hard_negative_count": clean_hardneg_count,
            "suspected_missed_positive_rows": sum(
                bool(row["suspected_missed_positives"]) for row in output_rows
            ),
        },
        "notes": [
            "Same-scene candidates are ranked before same-source and dense-only candidates; cosine breaks ties.",
            "For rows outside the GLM budget, hard_negatives carry glm_verdict=not_run and are provisional.",
            "Validated candidates judged supported are removed from hard_negatives and listed as suspected_missed_positives.",
        ],
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
