"""Bounded, resumable full-evidence GLM annotation; no GPU or production writes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


VERSION = "glm_full20_annotation_v2"
ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CODING_PLAN_ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只根据当前可见证据输出合法JSON，不使用模型记忆补充剧情。"
RULES = (
    "next_action只能是answer_directly、retrieve_more或abstain。证据足够时回答；仅能确认部分时"
    "输出可确认部分，不能为了完整而编造。answer_directly只包含next_action和supported_facts，"
    "supported_facts为1至8条原子事实，通常1至4条，不重复，每条只含fact和evidence_ids；"
    "每条事实绑定1至2个确实支持它的当前E编号。用自己的话概括，不复制长引文，不输出quote、"
    "evidence_refs、final_answer或推理过程。证据不足且还有检索轮次时才retrieve_more，"
    "仅输出next_action和follow_up_hypothesis；后者包含question、query_type、entities、keywords、"
    "expected_answer_type，entities和keywords为字符串列表，其余为字符串，可选dialogue_context。"
    "最后一轮若无足够支持回答的证据则abstain，仅含next_action和reason。"
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def json_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("source must be an array or JSONL")
        yield from value


def family(doc_id):
    return "/".join(doc_id.split("/")[:2])


def normalized_question(question):
    return re.sub(r"\W", "", question).lower()


def forbidden(doc_id, excluded):
    return bool(set(doc_id.split("/")) & set(excluded))


def source_question(row):
    """Read questions and evidence IDs only: old answers never enter the task."""
    if isinstance(row.get("question"), str):
        question = row["question"].strip()
        evidence = row.get("evidence") or row.get("documents") or []
        ids = [e.get("doc_id") or e.get("document_id") or e.get("id") for e in evidence]
        ids = [x for x in ids if isinstance(x, str)]
        round_value = row.get("round", "2/2")
    else:
        conversations = row.get("conversations") or []
        human = next((c["value"] for c in conversations if c.get("from") in ("human", "user")), "")
        if not human and conversations:
            human = conversations[0].get("value", "")
        match = re.search(r"(?m)^(?:问题：|question:\s*)(.+)$", human)
        question = match.group(1).strip() if match else ""
        ids = re.findall(r"(?m)^E\d+\s+(\S+)\s*$", human)
        round_match = re.search(r"(?m)^round:\s*(\d+/\d+)", human)
        round_value = round_match.group(1) if round_match else "2/2"
    if not re.fullmatch(r"[123]/[123]", str(round_value)) or (
            int(str(round_value).split("/")[0]) > int(str(round_value).split("/")[1])):
        raise ValueError("unsupported source round; do not silently invent a round")
    return question, list(dict.fromkeys(ids)), str(round_value)


def annotation_round(source_round, policy):
    if policy == "preserve":
        return source_round
    current, total = (int(x) for x in source_round.split("/"))
    # Preserve the actual action boundary, not a fictional third runtime round.
    return "2/2" if current == total else "1/2"


def prompt_for(question, round_value, documents):
    body = (
        f"task: grounded_action_generation\nquestion: {question}\n"
        'hypothesis: {"query_type":"reasoning","expected_answer_type":"剧情解释"}\n'
        f"round: {round_value}\nevidence:\n" +
        "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in documents) +
        "\noutput_schema: grounded_action_exx_v1\nrules: " + RULES
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    prompt = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" +
              body + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return messages, prompt


def check_row(row):
    if row["evidence_count"] != 20 or len(row["documents"]) != 20 or row["truncated"]:
        raise ValueError("full20 required")
    docs = row["documents"]
    if {d["evidence_id"] for d in docs} != {f"E{i}" for i in range(1, 21)}:
        raise ValueError("bad evidence numbers")
    if len({d["document_id"] for d in docs}) != 20:
        raise ValueError("duplicate evidence document")
    if any(not d["text"].strip() or sha(d["text"]) != d["sha256"] for d in docs):
        raise ValueError("bad evidence integrity")
    messages, prompt = prompt_for(row["question"], row["round"], docs)
    if prompt != row["prompt"] or messages != row["teacher_messages"]:
        raise ValueError("prompt mismatch")
    if sha(compact(messages)) != row["teacher_prompt_sha256"]:
        raise ValueError("teacher prompt hash mismatch")


def check_target(value, row):
    """Offline protocol/data check, not keyword-based answer correctness rules."""
    if not isinstance(value, dict):
        raise ValueError("not_object")
    action = value.get("next_action")
    if action == "answer_directly":
        if set(value) != {"next_action", "supported_facts"}:
            raise ValueError("answer_keys")
        facts = value["supported_facts"]
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            raise ValueError("fact_count")
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                raise ValueError("fact_keys")
            if not isinstance(fact["fact"], str) or not fact["fact"].strip():
                raise ValueError("empty_fact")
            refs = fact["evidence_ids"]
            if (not isinstance(refs, list) or not 1 <= len(refs) <= 2 or
                    any(not isinstance(x, str) or x not in {f"E{i}" for i in range(1, 21)} for x in refs)):
                raise ValueError("invalid_binding")
            if len(refs) != len(set(refs)):
                raise ValueError("duplicate_binding")
    elif action == "abstain":
        if set(value) != {"next_action", "reason"} or not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ValueError("abstain_keys")
    elif action == "retrieve_more":
        current_round, max_round = (int(x) for x in row["round"].split("/"))
        if current_round >= max_round:
            raise ValueError("retrieve_after_final_round")
        if set(value) != {"next_action", "follow_up_hypothesis"}:
            raise ValueError("retrieve_keys")
        hypo = value["follow_up_hypothesis"]
        fields = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
        if not isinstance(hypo, dict) or not fields <= set(hypo) or set(hypo) - fields - {"dialogue_context"}:
            raise ValueError("hypothesis_keys")
        for key in ("question", "query_type", "expected_answer_type"):
            if not isinstance(hypo[key], str) or not hypo[key].strip():
                raise ValueError("hypothesis_string")
        for key in ("entities", "keywords"):
            if not isinstance(hypo[key], list) or any(not isinstance(x, str) for x in hypo[key]):
                raise ValueError("hypothesis_list")
        if "dialogue_context" in hypo and not isinstance(hypo["dialogue_context"], str):
            raise ValueError("hypothesis_context")
    else:
        raise ValueError("unknown_action")
    return value


def components(rows):
    """Connect ALL document families and repeated questions, not only E1."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for row in rows:
        nodes = ["q:" + normalized_question(row["question"])] + ["f:" + f for f in row["source_families"]]
        for node in nodes[1:]:
            union(nodes[0], node)
    groups = {}
    for row in rows:
        group = find("q:" + normalized_question(row["question"]))
        groups.setdefault(group, []).append(row)
    return groups


def choose_splits(rows, requested_train, requested_val, seed, allow_fewer=False, heldout_families=()):
    groups = components(rows)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    # Small components are useful holdouts; never break a large connected group.
    names.sort(key=lambda n: len(groups[n]))
    val_names = [n for n in names if any(set(r["source_families"]) & set(heldout_families)
                                       for r in groups[n])]
    val_count = sum(len(groups[n]) for n in val_names)
    remaining = len(rows) - val_count
    minimum_train = min(requested_train, max(1, len(rows) - requested_val))
    for name in names:
        if name in val_names:
            continue
        size = len(groups[name])
        if val_count < requested_val and remaining - size >= minimum_train:
            val_names.append(name)
            val_count += size
            remaining -= size
    if not val_names:
        raise ValueError("cannot create independent validation components; try --single-family")
    def round_robin(selected, limit):
        result, position = [], 0
        while len(result) < limit:
            before = len(result)
            for name in selected:
                if position < len(groups[name]):
                    result.append(groups[name][position])
                    if len(result) == limit:
                        break
            if len(result) == before:
                break
            position += 1
        return result
    splits = {"train": round_robin([n for n in names if n not in val_names], requested_train),
              "val": round_robin(val_names, requested_val)}
    if not allow_fewer and (len(splits["train"]) < requested_train or len(splits["val"]) < requested_val):
        raise ValueError("requested counts unavailable without leakage; use --allow-fewer")
    if not splits["train"] or not splits["val"]:
        raise ValueError("empty split")
    assert_split(splits)
    return splits, {n: len(groups[n]) for n in names}


def assert_split(splits):
    for get_set in (
        lambda rr: {f for r in rr for f in r["source_families"]},
        lambda rr: {d["document_id"] for r in rr for d in r["documents"]},
        lambda rr: {normalized_question(r["question"]) for r in rr},
    ):
        if get_set(splits["train"]) & get_set(splits["val"]):
            raise ValueError("train/val leakage")


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    docs, by_family = {}, {}
    for doc in json_rows(args.documents):
        if doc["id"] in docs:
            raise ValueError("duplicate index document")
        docs[doc["id"]] = doc
        by_family.setdefault(family(doc["id"]), []).append(doc)
    candidates, rejects, seen = [], Counter(), set()
    for source in args.source:
        for index, raw in enumerate(json_rows(source)):
            try:
                question, ids, source_round = source_question(raw)
            except ValueError:
                rejects["unsupported_round"] += 1
                continue
            round_value = annotation_round(source_round, getattr(args, "round_policy", "phase-to-two"))
            if not question or not ids or len(ids) > 20 or any(i not in docs for i in ids):
                rejects["missing_question_or_evidence"] += 1
                continue
            if any(forbidden(i, args.exclude_family) for i in ids):
                rejects["excluded_evaluation_family"] += 1
                continue
            key = (normalized_question(question), round_value)
            if key in seen:
                rejects["duplicate_question_round"] += 1
                continue
            families = {family(i) for i in ids}
            if args.single_family and len(families) != 1:
                rejects["whole_multifamily_sample_excluded"] += 1
                continue
            pool = [d for f in families for d in by_family[f] if d["id"] not in ids
                    and not forbidden(d["id"], args.exclude_family)]
            def distance(doc):
                chapter, _, chunk = doc["id"].partition("#chunk-")
                same = [i for i in ids if i.partition("#chunk-")[0] == chapter]
                delta = min((abs(int(chunk) - int(i.partition("#chunk-")[2]))
                             for i in same if chunk.isdigit() and i.partition("#chunk-")[2].isdigit()),
                            default=999999)
                return not bool(same), delta, doc["id"]
            pool.sort(key=distance)
            selected = [docs[i] for i in ids] + pool[:20 - len(ids)]
            if len(selected) != 20 or any(not d.get("clean_text", "").strip() for d in selected):
                rejects["insufficient_complete_documents"] += 1
                continue
            stable_id = sha(compact({"question": question, "round": round_value,
                                    "original_ids": ids}))[:24]
            random.Random(int(stable_id, 16) ^ args.seed).shuffle(selected)
            documents = [{"evidence_id": f"E{i+1}", "document_id": d["id"],
                          "text": d["clean_text"], "sha256": sha(d["clean_text"])}
                         for i, d in enumerate(selected)]
            messages, prompt = prompt_for(question, round_value, documents)
            length = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if length + args.max_output_tokens + 16 > args.max_context_tokens:
                rejects["whole_sample_over_context_not_truncated"] += 1
                continue
            row = {"id": stable_id, "question": question, "round": round_value,
                   "source_round": source_round,
                   "source": str(source), "source_row_id": str(raw.get("task_id") or raw.get("id") or index),
                   "source_families": sorted({family(d["document_id"]) for d in documents}),
                   "documents": documents, "evidence_count": 20, "truncated": False,
                   "teacher_messages": messages, "teacher_prompt_sha256": sha(compact(messages)),
                   "prompt": prompt, "prompt_tokens": length, "old_labels_used": False}
            check_row(row)
            candidates.append(row)
            seen.add(key)
    splits, group_sizes = choose_splits(candidates, args.train, args.val, args.seed, args.allow_fewer,
                                       getattr(args, "holdout_family", ()))
    out.mkdir()
    rows = [dict(row, split=split) for split, rr in splits.items() for row in rr]
    (out / "requests.jsonl").write_text("".join(compact(r) + "\n" for r in rows))
    manifest = {
        "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "fresh_GLM_full20_annotation", "planned_train": len(splits["train"]),
        "planned_val": len(splits["val"]), "requested_train": args.train, "requested_val": args.val,
        "candidate_count": len(candidates), "unique_questions": len({normalized_question(r["question"]) for r in rows}),
        "train": 0, "val": 0, "complete": False, "old_labels_used": False,
        "evidence_count_per_example": 20, "truncated_documents": 0,
        "train_val_story_overlap": [], "train_val_source_family_overlap": [],
        "train_val_document_overlap": [], "train_val_question_overlap": [],
        "split_strategy": "all_source_family_and_question_connected_components",
        "source_index": str(args.documents), "source_index_sha256": file_sha(args.documents),
        "source_files_sha256": {str(s): file_sha(s) for s in args.source},
        "requests_sha256": file_sha(out / "requests.jsonl"), "rejected": dict(rejects),
        "component_sizes": sorted(group_sizes.values()),
        "families": {s: sorted({f for r in rr for f in r["source_families"]}) for s, rr in splits.items()},
        "single_family_scope": args.single_family,
        "round_policy": getattr(args, "round_policy", "phase-to-two"),
        "round_counts": dict(Counter(r["round"] for r in rows)),
        "source_round_counts": dict(Counter(r["source_round"] for r in rows)),
        "excluded_families": args.exclude_family, "max_context_tokens": args.max_context_tokens,
        "protected_earlier_holdout_families": getattr(args, "holdout_family", []),
        "max_output_tokens_reserved": args.max_output_tokens,
        "not_human_gold": True, "not_unseen_benchmark": True,
        "historical_questions_relabelled": True, "not_full_chain_retrieval_evaluation": True,
        "answer_validation_in_production_modified": False, "production_modified": False,
        "review_required_before_training": True,
    }
    atomic_json(out / "manifest.json", manifest)
    print(compact(manifest), flush=True)


@dataclass
class Settings:
    model: str = "glm-5.3"
    endpoint: str = ENDPOINT
    reasoning_effort: str = "high"
    max_output_tokens: int = 8192
    timeout: float = 180
    workers: int = 2
    attempts: int = 3
    max_api_calls: int = 400
    max_total_tokens: int = 6_000_000
    review_fraction: float = 0.1
    seed: int = 20261013
    request_interval: float = 0.5
    max_consecutive_failures: int = 5
    backoff_seconds: float = 2

    def validate(self):
        url = urllib.parse.urlsplit(self.endpoint)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("endpoint credentials/query not allowed")
        if self.endpoint not in (ENDPOINT, CODING_PLAN_ENDPOINT):
            raise ValueError("only official standard or Coding Plan GLM endpoints are allowed")
        if not 1 <= self.workers <= 8 or not 1 <= self.attempts <= 5:
            raise ValueError("invalid concurrency/retry bound")
        if not 0 <= self.review_fraction <= 1 or self.reasoning_effort not in ("low", "high", "max"):
            raise ValueError("invalid review/thinking settings")
        if min(self.max_api_calls, self.max_total_tokens, self.max_output_tokens) <= 0:
            raise ValueError("positive budget required")
        if self.max_consecutive_failures <= 0:
            raise ValueError("positive circuit breaker bound required")
        if self.timeout <= 0 or self.request_interval < 0 or self.backoff_seconds < 0:
            raise ValueError("invalid timing bound")
        return self

    def cache_settings(self):
        return {"version": VERSION, "endpoint": self.endpoint, "model": self.model,
                "reasoning_effort": self.reasoning_effort, "max_output_tokens": self.max_output_tokens}


class APIError(Exception):
    def __init__(self, kind, http_status=None, provider_code=None, retryable=False, terminal=False, retry_after=0):
        super().__init__(kind)
        self.kind, self.http_status, self.provider_code = kind, http_status, provider_code
        self.retryable, self.terminal, self.retry_after = retryable, terminal, retry_after

    def record(self):
        return {"kind": self.kind, "http_status": self.http_status,
                "provider_code": self.provider_code, "retryable": self.retryable, "terminal": self.terminal}


class BudgetReached(Exception):
    pass


def transport(settings, messages):
    key = os.environ.get("BIGMODEL_API_KEY", "")
    if not key:
        raise APIError("missing_api_key", terminal=True)
    body = {"model": settings.model, "messages": messages, "temperature": 0, "stream": False,
            "max_tokens": settings.max_output_tokens, "thinking": {"type": "enabled"},
            "reasoning_effort": settings.reasoning_effort, "response_format": {"type": "json_object"}}
    request = urllib.request.Request(settings.endpoint, compact(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")))
    try:
        with opener.open(request, timeout=settings.timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            provider_code = str(json.loads(exc.read()).get("error", {}).get("code", "unknown"))
        except (ValueError, AttributeError, TypeError):
            provider_code = "unknown"
        # Never persist raw error bodies, request headers, keys or reasoning.
        if provider_code == "1113":
            raise APIError("insufficient_balance", exc.code, provider_code, terminal=True) from None
        if exc.code == 400 and provider_code == "1301":
            raise APIError("content_filtered", exc.code, provider_code) from None
        if exc.code in (400, 401, 402, 403, 404):
            raise APIError("authentication_billing_or_configuration", exc.code, provider_code, terminal=True) from None
        retryable = exc.code == 429 or 500 <= exc.code <= 599
        try:
            retry_after = min(30, max(0, float(exc.headers.get("Retry-After", "0"))))
        except (ValueError, TypeError):
            retry_after = 0
        raise APIError("rate_limit" if exc.code == 429 else "http_error",
                       exc.code, provider_code, retryable=retryable, retry_after=retry_after) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
        raise APIError("network_or_timeout", retryable=True) from None
    except (ValueError, UnicodeError):
        raise APIError("invalid_response_envelope", retryable=True) from None


class Ledger:
    """SQLite transactions persist budgets BEFORE any outbound API call."""
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS calls(
                id INTEGER PRIMARY KEY, task_id TEXT, phase TEXT, cache_key TEXT,
                created REAL, state TEXT, reserved_tokens INTEGER, charged_tokens INTEGER, receipt TEXT);
            CREATE TABLE IF NOT EXISTS cache(cache_key TEXT PRIMARY KEY, receipt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(task_id TEXT PRIMARY KEY, fingerprint TEXT, state TEXT, result TEXT);
        """)
        # A crashed in-flight request may have been billed; retain its reserve.
        self.db.execute("UPDATE calls SET state='interrupted_unknown_billing' WHERE state='inflight'")
        self.db.commit()

    def close(self):
        self.db.close()

    def stats(self):
        with self.lock:
            calls, tokens = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(COALESCE(charged_tokens,reserved_tokens)),0) FROM calls").fetchone()
            states = dict(self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state").fetchall())
        return {"api_calls": calls, "accounted_tokens": tokens, "job_states": states}

    def reserve(self, row_id, phase, cache_key, estimated_tokens, settings):
        with self.lock:
            stats = self.stats()
            if stats["api_calls"] >= settings.max_api_calls or stats["accounted_tokens"] + estimated_tokens > settings.max_total_tokens:
                raise BudgetReached()
            cursor = self.db.execute(
                "INSERT INTO calls(task_id,phase,cache_key,created,state,reserved_tokens) VALUES(?,?,?,?,?,?)",
                (row_id, phase, cache_key, time.time(), "inflight", estimated_tokens))
            self.db.commit()
            return cursor.lastrowid

    def finish(self, call_id, receipt, cache_key=None):
        usage = receipt.get("usage") or {}
        total = usage.get("total_tokens")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            total = None
        with self.lock:
            self.db.execute("UPDATE calls SET state=?,charged_tokens=?,receipt=? WHERE id=?",
                            ("response" if receipt.get("response_id") else "error", total, compact(receipt), call_id))
            if cache_key:
                self.db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (cache_key, compact(receipt)))
            self.db.commit()

    def cached(self, key):
        with self.lock:
            row = self.db.execute("SELECT receipt FROM cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def job(self, task_id):
        with self.lock:
            value = self.db.execute("SELECT fingerprint,state,result FROM jobs WHERE task_id=?", (task_id,)).fetchone()
        return (value[0], value[1], json.loads(value[2])) if value else None

    def save_job(self, task_id, fingerprint, state, result):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?)",
                            (task_id, fingerprint, state, compact(result)))
            self.db.commit()

    def receipts(self):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute(
                "SELECT receipt FROM calls WHERE receipt IS NOT NULL ORDER BY id").fetchall()]


def review_messages(row, target):
    evidence = "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in row["documents"])
    return [{"role": "system", "content": "只基于给定完整证据复核候选标签，不用外部记忆，不要求或输出思维链。只输出JSON。"},
            {"role": "user", "content": f"问题：{row['question']}\n轮次：{row['round']}\n证据：\n" + evidence +
             "\n候选标签：" + compact(target) +
             '\n请核对每条事实是否被其引用支持、动作是否合适。仅输出'
             '{"label_supported":true或false,"decision_appropriate":true或false,'
             '"issues":["简短问题说明"]}。无证据的因果、过强断言、引用错位或错误拒答应指出。'}]


def check_review(value):
    if (not isinstance(value, dict) or set(value) != {"label_supported", "decision_appropriate", "issues"} or
            any(type(value[k]) is not bool for k in ("label_supported", "decision_appropriate")) or
            not isinstance(value["issues"], list) or any(not isinstance(v, str) for v in value["issues"])):
        raise ValueError("review_schema")
    return value


def verify_run_settings(out, settings):
    settings_path = Path(out) / "run_settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        # Retry count is an operational bound, not part of the teacher prompt
        # or cache fingerprint. Increasing it must not relabel accepted jobs.
        keys = set(vars(settings)) - {"workers", "timeout", "request_interval", "attempts",
                                     "max_api_calls", "max_total_tokens"}
        if any(previous.get(k) != vars(settings)[k] for k in keys):
            raise ValueError("annotation settings changed; use a new output directory")


class Runner:
    def __init__(self, out, rows, settings, request_fn=transport):
        self.out, self.rows, self.settings = Path(out), rows, settings.validate()
        self.request_fn = request_fn
        self.ledger = Ledger(self.out / "ledger.sqlite3")
        self.stop = threading.Event()
        self.reason = None
        self.reason_lock, self.rate_lock = threading.Lock(), threading.Lock()
        self.last_request = 0.0

    def fingerprint(self, row):
        return sha(compact({"prompt": row["teacher_prompt_sha256"],
                            "settings": self.settings.cache_settings(),
                            "review_fraction": self.settings.review_fraction, "review_seed": self.settings.seed}))

    def halt(self, reason):
        with self.reason_lock:
            if self.reason is None:
                self.reason = reason
        self.stop.set()

    def progress(self, stage):
        stats = self.ledger.stats()
        states = stats["job_states"]
        accepted = states.get("accepted", 0)
        quarantined = states.get("quarantined", 0)
        return dict(stats, total_tasks=len(self.rows), accepted=accepted, quarantined=quarantined,
                    remaining_tasks=len(self.rows) - accepted - quarantined,
                    unstarted_tasks=len(self.rows) - sum(states.values()),
                    stop_reason=self.reason, stage=stage,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    teacher_model=self.settings.model, teacher_endpoint=self.settings.endpoint)

    def call(self, row, phase, messages, validator):
        key = sha(compact({"phase": phase, "messages": messages, "settings": self.settings.cache_settings()}))
        cached = self.ledger.cached(key)
        if cached:
            validator(cached["parsed"])
            return cached
        for attempt in range(1, self.settings.attempts + 1):
            if self.stop.is_set():
                raise BudgetReached()
            # Conservative UTF-8 byte bound, not Qwen-token counts pretending
            # to be exact GLM billing. Actual usage settles the reservation.
            prompt_tokens = len(compact(messages).encode("utf-8")) + 128 * len(messages)
            call_id = self.ledger.reserve(row["id"], phase, key,
                prompt_tokens + self.settings.max_output_tokens, self.settings)
            with self.rate_lock:
                delay = self.settings.request_interval - (time.monotonic() - self.last_request)
                if delay > 0:
                    time.sleep(delay)
                self.last_request = time.monotonic()
            started = time.monotonic()
            receipt = {"task_id": row["id"], "phase": phase, "call_id": call_id, "attempt": attempt,
                       "cache_key": key, "teacher_prompt_sha256": row["teacher_prompt_sha256"],
                       "requested_model": self.settings.model, "endpoint": self.settings.endpoint,
                       "accepted": False}
            error = None
            try:
                response = self.request_fn(self.settings, messages)
                choice = response["choices"][0]
                content = choice["message"].get("content") or ""
                receipt.update(response_id=response.get("id"), returned_model=response.get("model"),
                               usage=response.get("usage"), finish_reason=choice.get("finish_reason"),
                               content=content)
                if choice.get("finish_reason") != "stop":
                    raise APIError("incomplete_output")
                try:
                    value = json.loads(content)
                except json.JSONDecodeError:
                    receipt["validation_error"] = "invalid_json"
                    raise APIError("invalid_json_or_protocol", retryable=True) from None
                try:
                    validator(value)
                except ValueError as exc:
                    # Our protocol validators emit fixed identifiers, not
                    # raw output or credentials. Never normalize the label.
                    detail = str(exc)
                    receipt["validation_error"] = detail if re.fullmatch(r"[a-z_]+", detail) else "invalid_protocol"
                    raise APIError("invalid_json_or_protocol", retryable=True) from None
                receipt.update(accepted=True, parsed=value)
            except APIError as exc:
                error = exc
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                error = APIError("invalid_json_or_protocol", retryable=True)
            receipt["elapsed_seconds"] = time.monotonic() - started
            if error is None:
                self.ledger.finish(call_id, receipt, key)
                return receipt
            receipt["error"] = error.record()
            self.ledger.finish(call_id, receipt)
            if error.terminal:
                self.halt(error.kind)
            if not error.retryable or attempt == self.settings.attempts:
                raise error
            delay = max(error.retry_after, min(30, self.settings.backoff_seconds * 2**(attempt - 1)))
            if self.stop.wait(delay):
                raise BudgetReached()
        raise AssertionError("unreachable")

    def process(self, row):
        check_row(row)
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        if prior and prior[0] != fingerprint:
            raise ValueError("job fingerprint changed; use a new output directory")
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        try:
            label = self.call(row, "annotate", row["teacher_messages"], lambda x: check_target(x, row))
            selected = int(sha(f"{self.settings.seed}:{row['id']}")[:16], 16) / 2**64 < self.settings.review_fraction
            review = None
            if selected:
                review = self.call(row, "review", review_messages(row, label["parsed"]), check_review)
            accepted = not review or (review["parsed"]["label_supported"] and review["parsed"]["decision_appropriate"])
            result = {"label": label, "review": review, "semantic_review_sampled": selected,
                      "semantic_review_is_independent_human_gold": False}
            state = "accepted" if accepted else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, result)
            return state
        except BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending", {"reason": self.reason or "budget_or_stop"})
            self.halt(self.reason or "budget_limit")
            return "pending"
        except APIError as error:
            state = "pending" if error.terminal or error.retryable else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, {"error": error.record()})
            return "failed" if error.retryable else state

    def run(self, limit=0):
        settings_path = self.out / "run_settings.json"
        settings_value = vars(self.settings)
        verify_run_settings(self.out, self.settings)
        atomic_json(settings_path, settings_value)
        for row in self.rows:
            check_row(row)
            prior = self.ledger.job(row["id"])
            if prior and prior[0] != self.fingerprint(row):
                raise ValueError("job fingerprint changed")
        pending = [r for r in self.rows if not self.ledger.job(r["id"]) or
                   self.ledger.job(r["id"])[1] not in ("accepted", "quarantined")]
        if limit:
            pending = pending[:limit]
        iterator, futures, failures = iter(pending), {}, 0
        with ThreadPoolExecutor(max_workers=self.settings.workers) as pool:
            def submit():
                if self.stop.is_set():
                    return
                row = next(iterator, None)
                if row:
                    futures[pool.submit(self.process, row)] = row["id"]
            for _ in range(self.settings.workers):
                submit()
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    task_id = futures.pop(future)
                    state = future.result()
                    failures = failures + 1 if state == "failed" else 0
                    if failures >= self.settings.max_consecutive_failures:
                        self.halt("transient_failure_circuit_breaker")
                    snapshot = dict(self.progress("blocked" if self.reason else "running"),
                                    last_task_id=task_id, last_state=state)
                    atomic_json(self.out / "status.json", snapshot)
                    print(compact(snapshot), flush=True)
                    submit()
        snapshot = self.progress("blocked" if self.reason else "batch_finished")
        atomic_json(self.out / "status.json", snapshot)
        return snapshot

    def export(self):
        splits, quarantine = {"train": [], "val": []}, []
        for row in self.rows:
            job = self.ledger.job(row["id"])
            if not job or job[1] != "accepted":
                if job:
                    quarantine.append({"id": row["id"], "state": job[1], "result": job[2]})
                continue
            if job[0] != self.fingerprint(row):
                raise ValueError("export fingerprint mismatch")
            label = job[2]["label"]
            check_target(label["parsed"], row)
            exported = {k: v for k, v in row.items() if k != "teacher_messages"}
            exported.update(target=compact(label["parsed"]), source_label="fresh_GLM_API_full20",
                            user_prompt=row["teacher_messages"][1]["content"],
                            label_prompt_sha256=row["teacher_prompt_sha256"],
                            annotation_response_id=label.get("response_id"),
                            annotation_model=label.get("returned_model"),
                            semantic_review_sampled=job[2]["semantic_review_sampled"],
                            semantic_review=job[2]["review"]["parsed"] if job[2]["review"] else None)
            splits[row["split"]].append(exported)
        assert_split(splits)
        for split, rows in splits.items():
            path = self.out / f"{split}.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(compact(r) + "\n" for r in rows))
            os.replace(temporary, path)
            sharegpt = [{"id": r["id"], "system": SYSTEM, "conversations": [
                {"from": "human", "value": r["user_prompt"]},
                {"from": "gpt", "value": r["target"]}]} for r in rows]
            atomic_json(self.out / f"{split}.json", sharegpt)
        atomic_json(self.out / "quarantine.json", quarantine)
        # Sanitized receipts; no authorization header or hidden reasoning stored.
        path = self.out / "api_receipts.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(compact(r) + "\n" for r in self.ledger.receipts()))
        os.replace(temporary, path)
        manifest = json.loads((self.out / "manifest.json").read_text())
        status_path = self.out / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"stage": "prepared"}
        counts = {s: len(rr) for s, rr in splits.items()}
        manifest.update(counts, new_api_calls=self.ledger.stats()["api_calls"],
                        labelled_examples=sum(counts.values()),
                        labels_provider=("bigmodel_coding_plan_api" if self.settings.endpoint == CODING_PLAN_ENDPOINT
                                         else "bigmodel_standard_api"),
                        teacher_model=self.settings.model, teacher_endpoint=self.settings.endpoint,
                        review_fraction=self.settings.review_fraction,
                        complete=sum(counts.values()) == len(self.rows),
                        data_sha256={s: file_sha(self.out / f"{s}.jsonl") for s in splits},
                        action_counts={s: dict(Counter(json.loads(r["target"])["next_action"] for r in rr))
                                       for s, rr in splits.items()},
                        run_status=self.reason or status.get("stop_reason") or status.get("stage", "prepared"),
                        no_QA_accuracy_claim=True)
        atomic_json(self.out / "manifest.json", manifest)
        return manifest


@contextmanager
def exclusive_run(out):
    with (Path(out) / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another annotation/export process holds this dataset") from None
        yield


def load_requests(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["version"] != VERSION or file_sha(out / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("immutable input changed")
    rows = list(json_rows(out / "requests.jsonl"))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate task id")
    for row in rows:
        check_row(row)
    assert_split({s: [r for r in rows if r["split"] == s] for s in ("train", "val")})
    return rows


def fork_inputs(source, out):
    """Reuse verified questions/evidence only; never copy labels, state or budgets."""
    source, out = Path(source).resolve(), Path(out).resolve()
    rows = load_requests(source)
    manifest = json.loads((source / "manifest.json").read_text())
    # A completed/blocked manifest also contains export and runtime metadata.
    for key in ("new_api_calls", "labelled_examples", "labels_provider", "teacher_model",
                "teacher_endpoint", "review_fraction", "data_sha256", "action_counts",
                "run_status", "no_QA_accuracy_claim"):
        manifest.pop(key, None)
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(),
                    train=0, val=0, complete=False, labelled_examples=0,
                    planned_train=sum(r["split"] == "train" for r in rows),
                    planned_val=sum(r["split"] == "val" for r in rows),
                    forked_from=str(source), forked_from_manifest_sha256=file_sha(source / "manifest.json"),
                    labels_copied=False, ledger_copied=False, production_modified=False)
    out.mkdir()  # Refuse existing directories, including the source itself.
    shutil.copyfile(source / "requests.jsonl", out / "requests.jsonl")
    if file_sha(out / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("source input changed during fork")
    atomic_json(out / "manifest.json", manifest)
    load_requests(out)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", action="append", required=True)
    prep.add_argument("--documents", required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--train", type=int, default=300)
    prep.add_argument("--val", type=int, default=50)
    prep.add_argument("--seed", type=int, default=20261013)
    prep.add_argument("--exclude-family", action="append", default=["act10d5", "act10mini"])
    prep.add_argument("--single-family", action="store_true")
    prep.add_argument("--holdout-family", action="append", default=[])
    prep.add_argument("--round-policy", choices=["phase-to-two", "preserve"], default="phase-to-two")
    prep.add_argument("--allow-fewer", action="store_true")
    prep.add_argument("--max-output-tokens", type=int, default=8192)
    prep.add_argument("--max-context-tokens", type=int, default=32768)
    run = commands.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--limit", type=int, default=0)
    status = commands.add_parser("status")
    status.add_argument("--out", required=True)
    export = commands.add_parser("export")
    export.add_argument("--out", required=True)
    export.add_argument("--config", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--source", action="append", required=True)
    fork = commands.add_parser("fork-inputs")
    fork.add_argument("--source", required=True)
    fork.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "fork-inputs":
        print(compact(fork_inputs(args.source, args.out)), flush=True)
        return
    if args.command == "inspect":
        for source in args.source:
            rounds, evidence_sizes, parsed, rejected = Counter(), Counter(), 0, 0
            for raw in json_rows(source):
                rounds[str(raw.get("round", "missing"))] += 1
                evidence_sizes[len(raw.get("evidence", raw.get("documents", [])))] += 1
                try:
                    question, ids, round_value = source_question(raw)
                    parsed += bool(question and ids)
                except ValueError:
                    rejected += 1
            print(compact({"source": source, "raw_round_counts": dict(rounds),
                           "evidence_count_distribution": dict(evidence_sizes),
                           "parsed_question_evidence_rows": parsed, "unsupported_rounds": rejected}))
        return
    if args.command == "prepare":
        if args.train <= 0 or args.val <= 0:
            raise ValueError("positive train and validation sizes required")
        prepare(args)
        return
    out = Path(args.out)
    if args.command == "status":
        # Read-only; never alter in-flight calls to inspect progress.
        if (out / "status.json").exists():
            print((out / "status.json").read_text())
        else:
            print(compact({"stage": "prepared", "manifest": json.loads((out / "manifest.json").read_text())}))
        return
    settings = Settings(**json.loads(Path(args.config).read_text())).validate()
    with exclusive_run(out):
        rows = load_requests(out)
        manifest = json.loads((out / "manifest.json").read_text())
        if settings.max_output_tokens > manifest["max_output_tokens_reserved"]:
            raise ValueError("output budget exceeds prepared context reservation")
        runner = Runner(out, rows, settings)
        try:
            if args.command == "run":
                if not os.environ.get("BIGMODEL_API_KEY"):
                    runner.halt("missing_api_key")
                    atomic_json(out / "status.json", dict(runner.ledger.stats(), stage="blocked",
                                                          stop_reason="missing_api_key"))
                else:
                    runner.run(args.limit)
            print(compact(runner.export()), flush=True)
            if runner.reason:
                raise SystemExit(3 if runner.reason == "budget_limit" else 2)
        finally:
            runner.ledger.close()


if __name__ == "__main__":
    main()
