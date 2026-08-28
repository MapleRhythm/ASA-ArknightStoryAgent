#!/usr/bin/env python3
"""GLM evidence-only semantic reward for Exx rollout groups.

The judge never receives gold actions, reference facts, required evidence, or
hidden answers.  It sees only the user question, round, complete visible
evidence, and candidate rollouts.  Deterministic protocol checks remain the
responsibility of the GRPO training script.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "asa_glm_exx_evidence_judge_v1"
SUPPORT_VALUES = {"entailed": 1.0, "partial": 0.25, "unsupported": 0.0, "contradicted": -1.0}
APPROPRIATENESS_VALUES = {"appropriate": 1.0, "inappropriate": -1.0, "uncertain": 0.0}
COVERAGE_VALUES = {"complete": 1.0, "partial": 0.25, "none": 0.0, "not_applicable": 0.0}
EVIDENCE_HEADER_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
FOLLOW_UP_REQUIRED = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
FOLLOW_UP_OPTIONAL = {"dialogue_context"}


class SemanticJudgeError(RuntimeError):
    pass


class TerminalSemanticJudgeError(SemanticJudgeError):
    pass


def build_ssl_context(ca_bundle: str | Path | None = None) -> ssl.SSLContext:
    """Build a reproducible HTTPS context even in relocated Python envs.

    Some training environments retain an OpenSSL default path from the
    interpreter's original Conda prefix.  Prefer an explicit bundle, then the
    environment override, and finally the standard Linux system bundle before
    falling back to Python's compiled defaults.
    """
    candidates = [
        str(ca_bundle or "").strip(),
        os.environ.get("SSL_CERT_FILE", "").strip(),
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    selected = next((path for path in candidates if path and Path(path).is_file()), None)
    if ca_bundle and selected != str(ca_bundle):
        raise ValueError(f"GLM CA bundle does not exist: {ca_bundle}")
    return ssl.create_default_context(cafile=selected) if selected else ssl.create_default_context()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: Any) -> dict[str, Any] | None:
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


def completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content") or "")
    if isinstance(completion, Mapping):
        return str(completion.get("content") or "")
    return str(completion or "")


def extract_judge_context(user_prompt: str) -> dict[str, str]:
    question = re.search(r"^question:\s*(.+)$", user_prompt, re.MULTILINE)
    round_match = re.search(r"^round:\s*(.+)$", user_prompt, re.MULTILINE)
    evidence = re.search(
        r"^evidence:\s*\n(.*?)\noutput_schema:\s*[^\n]+(?:\n|$)",
        user_prompt,
        re.MULTILINE | re.DOTALL,
    )
    if not question or not round_match or not evidence:
        raise SemanticJudgeError("cannot_parse_exx_prompt")
    evidence_text = evidence.group(1).strip()
    if not EVIDENCE_HEADER_RE.search(evidence_text):
        raise SemanticJudgeError("judge_context_has_no_evidence")
    return {
        "question": question.group(1).strip(),
        "round": round_match.group(1).strip(),
        "evidence": evidence_text,
    }


def payload_is_judge_eligible(payload: dict[str, Any], evidence_text: str) -> bool:
    """Gate semantic credit behind the deterministic Exx protocol."""
    visible_ids = set(EVIDENCE_HEADER_RE.findall(evidence_text))
    action = payload.get("next_action")
    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            return False
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return False
        seen: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                return False
            text = fact.get("fact")
            ids = fact.get("evidence_ids")
            normalized = re.sub(r"[^\w\u3400-\u9fff]+", "", str(text or "").lower())
            if not isinstance(text, str) or not text.strip() or not normalized or normalized in seen:
                return False
            seen.add(normalized)
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or len({str(item) for item in ids}) != len(ids)
                or any(str(item) not in visible_ids for item in ids)
            ):
                return False
        return True
    if action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            return False
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict):
            return False
        keys = set(follow_up)
        return (
            FOLLOW_UP_REQUIRED.issubset(keys)
            and not keys - FOLLOW_UP_REQUIRED - FOLLOW_UP_OPTIONAL
            and isinstance(follow_up.get("question"), str)
            and bool(follow_up.get("question", "").strip())
            and all(isinstance(follow_up.get(key), str) for key in ("query_type", "expected_answer_type"))
            and all(
                isinstance(follow_up.get(key), list)
                and all(isinstance(item, str) for item in follow_up[key])
                for key in ("entities", "keywords")
            )
            and (
                "dialogue_context" not in follow_up
                or isinstance(follow_up["dialogue_context"], str)
            )
        )
    return (
        action == "abstain"
        and set(payload) == {"next_action", "reason"}
        and isinstance(payload.get("reason"), str)
        and bool(payload.get("reason", "").strip())
    )


def build_messages(
    context: dict[str, str], indexed_payloads: Sequence[tuple[int, dict[str, Any]]]
) -> list[dict[str, str]]:
    rollouts = [{"rollout_index": index, "output": payload} for index, payload in indexed_payloads]
    output_example = {
        "protocol": PROTOCOL_VERSION,
        "rollouts": [
            {
                "rollout_index": 0,
                "action_appropriateness": "appropriate|inappropriate|uncertain",
                "facts": [
                    {
                        "fact_index": 0,
                        "support": "entailed|partial|unsupported|contradicted",
                        "checked_evidence_ids": ["E1"],
                        "citation_complete": True,
                    }
                ],
                "coverage": "complete|partial|none|not_applicable",
                "critical_unsupported_claims": 0,
            }
        ],
    }
    user = "\n".join(
        (
            "你是独立的RAG证据裁判。只依据下面当前可见证据评价候选输出，不得使用游戏常识或候选之外的信息。",
            "你不能改写候选输出，不能生成参考答案，也不能猜测隐藏gold。",
            f"问题：{context['question']}",
            f"轮次：{context['round']}",
            "当前可见证据（完整正文）：",
            context["evidence"],
            "候选rollout：",
            compact_json(rollouts),
            "只输出一个JSON对象，结构如下：",
            compact_json(output_example),
            "判定规则：",
            "1. 对answer_directly逐fact审核。support只能依据该fact自己的evidence_ids正文；主题相关但未陈述该断言应为unsupported，人物/时间/因果相反应为contradicted。",
            "2. checked_evidence_ids必须与候选fact给出的evidence_ids完全相同；不得补充别的E编号。",
            "3. citation_complete表示所列证据是否足以支持该fact的完整断言；缺少关键因果、身份、动机或时间条件时为false。",
            "4. coverage检查全部受支持facts合起来是否完整回答问题核心信息需求；只回答一部分为partial。",
            "5. action_appropriateness根据当前全部证据和轮次判断动作是否合适。证据足够却retrieve/abstain，或证据不足却answer，均为inappropriate。",
            "6. retrieve_more或abstain的facts必须为空且coverage为not_applicable，只判断动作是否合适。",
            "7. 无法从当前证据可靠判定时使用uncertain，不得凭常识强行判正。",
            "8. 每个输入rollout恰好返回一项，索引及fact_index不得遗漏、重复或新增。不要输出解释、引文或思维过程。",
        )
    )
    return [
        {"role": "system", "content": "你是严格的RAG证据裁判，只输出合法JSON。"},
        {"role": "user", "content": user},
    ]


def validate_judgement(
    value: dict[str, Any], indexed_payloads: Sequence[tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    if value.get("protocol") != PROTOCOL_VERSION or set(value) != {"protocol", "rollouts"}:
        raise SemanticJudgeError("invalid_judge_top_schema")
    rows = value.get("rollouts")
    if not isinstance(rows, list):
        raise SemanticJudgeError("judge_rollouts_not_list")
    expected = {index: payload for index, payload in indexed_payloads}
    actual: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "rollout_index",
            "action_appropriateness",
            "facts",
            "coverage",
            "critical_unsupported_claims",
        }:
            raise SemanticJudgeError("invalid_judge_rollout_schema")
        index = row.get("rollout_index")
        if not isinstance(index, int) or index not in expected or index in actual:
            raise SemanticJudgeError("invalid_or_duplicate_rollout_index")
        if row.get("action_appropriateness") not in APPROPRIATENESS_VALUES:
            raise SemanticJudgeError("invalid_action_appropriateness")
        if row.get("coverage") not in COVERAGE_VALUES:
            raise SemanticJudgeError("invalid_coverage")
        unsupported = row.get("critical_unsupported_claims")
        if not isinstance(unsupported, int) or isinstance(unsupported, bool) or unsupported < 0:
            raise SemanticJudgeError("invalid_critical_unsupported_claims")
        facts = row.get("facts")
        if not isinstance(facts, list):
            raise SemanticJudgeError("judge_facts_not_list")
        payload = expected[index]
        predicted_facts = payload.get("supported_facts") if payload.get("next_action") == "answer_directly" else []
        predicted_facts = predicted_facts if isinstance(predicted_facts, list) else []
        if len(facts) != len(predicted_facts):
            raise SemanticJudgeError("judge_fact_count_mismatch")
        seen_fact_indices: set[int] = set()
        for fact_row in facts:
            if not isinstance(fact_row, dict) or set(fact_row) != {
                "fact_index",
                "support",
                "checked_evidence_ids",
                "citation_complete",
            }:
                raise SemanticJudgeError("invalid_judge_fact_schema")
            fact_index = fact_row.get("fact_index")
            if (
                not isinstance(fact_index, int)
                or fact_index < 0
                or fact_index >= len(predicted_facts)
                or fact_index in seen_fact_indices
            ):
                raise SemanticJudgeError("invalid_or_duplicate_fact_index")
            seen_fact_indices.add(fact_index)
            if fact_row.get("support") not in SUPPORT_VALUES:
                raise SemanticJudgeError("invalid_fact_support")
            checked = fact_row.get("checked_evidence_ids")
            expected_ids = predicted_facts[fact_index].get("evidence_ids")
            if not isinstance(checked, list) or [str(item) for item in checked] != [
                str(item) for item in expected_ids or []
            ]:
                raise SemanticJudgeError("checked_evidence_ids_mismatch")
            if not isinstance(fact_row.get("citation_complete"), bool):
                raise SemanticJudgeError("invalid_citation_complete")
        if payload.get("next_action") != "answer_directly" and row.get("coverage") != "not_applicable":
            raise SemanticJudgeError("nonanswer_coverage_must_be_not_applicable")
        actual[index] = row
    if set(actual) != set(expected):
        raise SemanticJudgeError("judge_rollout_indices_mismatch")
    return {"protocol": PROTOCOL_VERSION, "rollouts": [actual[index] for index in sorted(actual)]}


def semantic_score(row: dict[str, Any], payload: dict[str, Any]) -> float:
    action_score = APPROPRIATENESS_VALUES[row["action_appropriateness"]]
    if payload.get("next_action") != "answer_directly":
        return action_score
    facts = row["facts"]
    if not facts:
        return -1.0
    entailment = sum(SUPPORT_VALUES[item["support"]] for item in facts) / len(facts)
    citation = sum(bool(item["citation_complete"]) for item in facts) / len(facts)
    coverage = COVERAGE_VALUES[row["coverage"]]
    score = 0.5 * entailment + 0.2 * citation + 0.2 * coverage + 0.1 * action_score
    score -= min(0.5, 0.25 * row["critical_unsupported_claims"])
    if any(item["support"] == "contradicted" for item in facts):
        score = min(score, -0.5)
    return max(-1.0, min(1.0, score))


def judgement_counts(judgement: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in judgement["rollouts"]:
        counts[f"action:{row['action_appropriateness']}"] += 1
        counts[f"coverage:{row['coverage']}"] += 1
        counts["critical_unsupported_claims"] += row["critical_unsupported_claims"]
        for fact in row["facts"]:
            counts[f"support:{fact['support']}"] += 1
            counts["citation_complete"] += int(fact["citation_complete"])
            counts["facts"] += 1
    return dict(sorted(counts.items()))


def build_request_body(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = reasoning_effort
    return body


class GlmEvidenceJudge:
    """Synchronous, cached GLM judge suitable for a GRPO reward closure."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        cache_path: Path,
        failures_path: Path,
        timeout: float = 240.0,
        max_tokens: int = 8192,
        max_attempts: int = 3,
        reasoning_effort: str = "high",
        workers: int = 1,
        max_consecutive_failures: int = 3,
        ca_bundle: str | Path | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("missing GLM API key")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.cache_path = cache_path
        self.failures_path = failures_path
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.reasoning_effort = reasoning_effort
        self.workers = max(1, workers)
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.ca_bundle = str(ca_bundle or "").strip() or None
        self.ssl_context = build_ssl_context(self.ca_bundle)
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        self._consecutive_failures = 0
        self.stats: Counter[str] = Counter()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("cache_key") and isinstance(row.get("scores"), list):
                    cache[str(row["cache_key"])] = row
        return cache

    def _append(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(compact_json(value) + "\n")
                handle.flush()

    def _cache_key(self, context: dict[str, str], indexed: Sequence[tuple[int, dict[str, Any]]]) -> str:
        value = {
            "protocol": PROTOCOL_VERSION,
            "endpoint": self.endpoint,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "context": context,
            "rollouts": [{"index": index, "payload": payload} for index, payload in indexed],
        }
        return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()

    def _request(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        body = build_request_body(
            messages,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=self.ssl_context),
        )
        try:
            with opener.open(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code in {400, 401, 402, 403}:
                raise TerminalSemanticJudgeError(f"http_{exc.code}:{detail}") from exc
            raise SemanticJudgeError(f"http_{exc.code}:{detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise SemanticJudgeError(f"network_error:{exc}") from exc
        try:
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SemanticJudgeError("api_response_missing_content") from exc
        return str(content), {
            "id": response_body.get("id"),
            "model": response_body.get("model"),
            "usage": response_body.get("usage"),
            "finish_reason": response_body.get("choices", [{}])[0].get("finish_reason"),
        }

    def score_group(self, user_prompt: str, completion_values: Sequence[Any]) -> list[float]:
        scores = [0.0] * len(completion_values)
        context = extract_judge_context(user_prompt)
        indexed: list[tuple[int, dict[str, Any]]] = []
        for index, completion in enumerate(completion_values):
            payload = parse_json_object(completion_text(completion))
            if payload is not None and payload_is_judge_eligible(payload, context["evidence"]):
                indexed.append((index, payload))
        if not indexed:
            self.stats["skip:no_parseable_rollout"] += 1
            return scores
        cache_key = self._cache_key(context, indexed)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.stats["cache_hit"] += 1
            cached_scores = cached["scores"]
            for (index, _), value in zip(indexed, cached_scores, strict=True):
                scores[index] = float(value)
            return scores

        messages = build_messages(context, indexed)
        last_error = ""
        last_response: dict[str, Any] | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw, api_meta = self._request(messages)
                last_response = {
                    "api": api_meta,
                    # The assistant content should contain only the requested
                    # judgement JSON.  Persist a bounded excerpt so malformed
                    # or length-truncated responses remain diagnosable without
                    # storing provider reasoning content.
                    "content_chars": len(raw),
                    "content_excerpt": raw[:4000],
                }
                parsed = parse_json_object(raw)
                if parsed is None:
                    raise SemanticJudgeError("judge_response_invalid_json")
                judgement = validate_judgement(parsed, indexed)
                semantic_scores = [
                    semantic_score(row, payload)
                    for row, (_, payload) in zip(judgement["rollouts"], indexed, strict=True)
                ]
                record = {
                    "cache_key": cache_key,
                    "protocol": PROTOCOL_VERSION,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "context": context,
                    "rollouts": [
                        {"rollout_index": index, "output": payload} for index, payload in indexed
                    ],
                    "rollout_indices": [index for index, _ in indexed],
                    "scores": semantic_scores,
                    "counts": judgement_counts(judgement),
                    "judgement": judgement,
                    "api": api_meta,
                    "attempt": attempt,
                }
                self._append(self.cache_path, record)
                self._cache[cache_key] = record
                self.stats["api_success"] += 1
                self._consecutive_failures = 0
                for (index, _), value in zip(indexed, semantic_scores, strict=True):
                    scores[index] = value
                return scores
            except TerminalSemanticJudgeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}:{exc}"
                if attempt < self.max_attempts:
                    time.sleep(min(2**attempt, 8))
        self.stats["api_failure"] += 1
        self._consecutive_failures += 1
        self._append(
            self.failures_path,
            {
                "cache_key": cache_key,
                "protocol": PROTOCOL_VERSION,
                "model": self.model,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "rollout_indices": [index for index, _ in indexed],
                "error": last_error[:2000],
                "last_response": last_response,
            },
        )
        if self._consecutive_failures >= self.max_consecutive_failures:
            raise SemanticJudgeError(
                f"GLM judge failed {self._consecutive_failures} consecutive groups; last={last_error}"
            )
        # A single transient failure supplies no semantic advantage. It must
        # never turn an API outage into a negative reward for the policy.
        return scores

    def score_batch(
        self,
        completions: Sequence[Any],
        judge_context: Sequence[str],
    ) -> list[float]:
        if len(completions) != len(judge_context):
            raise ValueError("completion/context length mismatch")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, context in enumerate(judge_context):
            grouped[str(context)].append(index)

        result = [0.0] * len(completions)

        def score_one(item: tuple[str, list[int]]) -> tuple[list[int], list[float]]:
            context, indices = item
            values = [completions[index] for index in indices]
            return indices, self.score_group(context, values)

        items = list(grouped.items())
        if self.workers == 1 or len(items) == 1:
            scored = map(score_one, items)
        else:
            executor = ThreadPoolExecutor(max_workers=min(self.workers, len(items)))
            scored = executor.map(score_one, items)
        try:
            for indices, values in scored:
                for index, value in zip(indices, values, strict=True):
                    result[index] = value
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True)
        return result


def make_glm_semantic_reward(judge: GlmEvidenceJudge):
    def glm_semantic_reward(
        completions: Sequence[Any], judge_context: Sequence[str], **_: Any
    ) -> list[float]:
        return judge.score_batch(completions, judge_context)

    glm_semantic_reward.__name__ = "glm_semantic_reward"
    return glm_semantic_reward
