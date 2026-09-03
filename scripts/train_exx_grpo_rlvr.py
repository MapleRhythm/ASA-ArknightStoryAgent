#!/usr/bin/env python3
"""Train grounded_action_exx_v1 with fixed-evidence GRPO/RLVR rewards.

Historical profiles optimize deterministic protocol and hidden-teacher proxy
targets.  ``glm-semantic-gated`` instead makes evidence-only GLM judgement the
dominant signal: hidden gold actions and reference facts remain available for
auditing, but do not contribute reward in that profile.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
ACTIONS = ("answer_directly", "retrieve_more", "abstain")
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
FOLLOW_UP_REQUIRED = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
FOLLOW_UP_OPTIONAL = {"dialogue_context"}
REWARD_PROFILES = (
    "legacy",
    "semantic-gated",
    "protocol-gated-rules",
    "glm-semantic-gated",
    "glm-precision-gated",
    "glm-precision-structural",
    "glm-precision-structural-v2",
)
SEMANTIC_JUDGE_PROTOCOL = "asa_glm_exx_evidence_judge_v3"
GLM_SEMANTIC_DEFAULT_WEIGHT = 3.0
GLM_SEMANTIC_MIN_WEIGHT = 2.0
NEAR_DUPLICATE_THRESHOLD = 0.78


@dataclass(frozen=True)
class TrainingExample:
    row_id: str
    prompt: list[dict[str, str]]
    prompt_tokens: int
    visible_ids: list[str]
    gold_action: str
    gold_fact_bindings: list[dict[str, Any]]


def parse_json_object(text: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(str(text or "").strip())
    except (json.JSONDecodeError, TypeError, ValueError):
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


def normalized_fact_text(text: Any) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", str(text or "").lower())


def validate_payload(payload: dict[str, Any] | None, visible_ids: set[str]) -> list[str]:
    """Return strict grounded_action_exx_v1 structural problems."""
    if payload is None:
        return ["invalid_json"]
    problems: list[str] = []
    legacy = FORBIDDEN.intersection(payload)
    if legacy:
        problems.append("legacy_fields:" + ",".join(sorted(legacy)))
    action = str(payload.get("next_action") or "")
    if action not in ACTIONS:
        return [*problems, "invalid_action"]

    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            problems.append("answer_top_schema")
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return [*problems, "invalid_fact_count"]
        seen_facts: set[str] = set()
        for index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                problems.append(f"fact_{index}_schema")
                continue
            text = fact.get("fact")
            evidence_ids = fact.get("evidence_ids")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"fact_{index}_empty")
            fact_key = normalized_fact_text(text)
            if fact_key and fact_key in seen_facts:
                problems.append(f"fact_{index}_duplicate")
            elif fact_key:
                seen_facts.add(fact_key)
            if (
                not isinstance(evidence_ids, list)
                or not 1 <= len(evidence_ids) <= 2
                or len({str(item) for item in evidence_ids}) != len(evidence_ids)
            ):
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible_ids for item in evidence_ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            problems.append("retrieve_top_schema")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict):
            problems.append("missing_follow_up")
        else:
            keys = set(follow_up)
            if not FOLLOW_UP_REQUIRED.issubset(keys) or keys - FOLLOW_UP_REQUIRED - FOLLOW_UP_OPTIONAL:
                problems.append("follow_up_schema")
            if not isinstance(follow_up.get("question"), str) or not follow_up.get("question", "").strip():
                problems.append("follow_up_question")
            for key in ("query_type", "expected_answer_type"):
                if not isinstance(follow_up.get(key), str):
                    problems.append(f"follow_up_{key}")
            for key in ("entities", "keywords"):
                value = follow_up.get(key)
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    problems.append(f"follow_up_{key}")
            if "dialogue_context" in follow_up and not isinstance(follow_up["dialogue_context"], str):
                problems.append("follow_up_dialogue_context")
    else:
        if set(payload) != {"next_action", "reason"}:
            problems.append("abstain_top_schema")
        if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
            problems.append("abstain_reason")
    return problems


def cited_evidence_ids(payload: dict[str, Any] | None) -> set[str]:
    if not payload or payload.get("next_action") != "answer_directly":
        return set()
    result: set[str] = set()
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return result
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("evidence_ids"), list):
            continue
        result.update(str(item) for item in fact["evidence_ids"])
    return result


def supported_fact_texts(payload: dict[str, Any] | None) -> list[str]:
    if not payload or payload.get("next_action") != "answer_directly":
        return []
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return []
    return [
        str(fact.get("fact") or "").strip()
        for fact in facts
        if isinstance(fact, dict) and str(fact.get("fact") or "").strip()
    ]


def supported_fact_bindings(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return hidden teacher claims with their claim-local citation IDs."""
    if not payload or payload.get("next_action") != "answer_directly":
        return []
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return []
    bindings: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("fact") or "").strip()
        evidence_ids = fact.get("evidence_ids")
        if text and isinstance(evidence_ids, list):
            bindings.append(
                {"fact": text, "evidence_ids": sorted({str(item) for item in evidence_ids})}
            )
    return bindings


def normalized_char_ngrams(texts: Sequence[str], n: int) -> Counter[str]:
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", "", "。".join(texts).lower())
    if not normalized:
        return Counter()
    if len(normalized) < n:
        return Counter({normalized: 1})
    return Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def reference_fact_similarity(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Deterministic character 1-3 gram F1 against hidden teacher facts."""
    scores: list[float] = []
    for n in (1, 2, 3):
        predicted_ngrams = normalized_char_ngrams(predicted, n)
        expected_ngrams = normalized_char_ngrams(expected, n)
        predicted_total = sum(predicted_ngrams.values())
        expected_total = sum(expected_ngrams.values())
        if not predicted_total or not expected_total:
            scores.append(0.0)
            continue
        overlap = sum((predicted_ngrams & expected_ngrams).values())
        precision = overlap / predicted_total
        recall = overlap / expected_total
        scores.append(2 * precision * recall / (precision + recall) if overlap else 0.0)
    return sum(scores) / len(scores)


def fact_pair_similarity(left: str, right: str) -> float:
    """Conservative lexical similarity for detecting near-duplicate claims.

    This is deliberately not an entailment proxy.  It only detects highly
    overlapping 1-3 character n-grams, which catches reward-padding variants
    while leaving semantic correctness to the evidence judge.
    """
    scores: list[float] = []
    for n in (1, 2, 3):
        left_ngrams = normalized_char_ngrams([left], n)
        right_ngrams = normalized_char_ngrams([right], n)
        left_total = sum(left_ngrams.values())
        right_total = sum(right_ngrams.values())
        if not left_total or not right_total:
            scores.append(0.0)
            continue
        overlap = sum((left_ngrams & right_ngrams).values())
        scores.append(2 * overlap / (left_total + right_total))
    return sum(scores) / len(scores)


def maximum_bipartite_score(scores: Sequence[Sequence[float]]) -> float:
    """Return the exact maximum one-to-one matching score for a small matrix."""
    if not scores or not scores[0]:
        return 0.0
    width = len(scores[0])
    if any(len(row) != width for row in scores):
        raise ValueError("ragged score matrix")
    # supported_facts is capped at eight, so exact bitmask DP is both cheaper
    # and less error-prone than a greedy approximation or another dependency.
    best_by_used_columns = {0: 0.0}
    for row in scores:
        updated = dict(best_by_used_columns)
        for used_columns, current in best_by_used_columns.items():
            for column, value in enumerate(row):
                column_bit = 1 << column
                if used_columns & column_bit:
                    continue
                next_columns = used_columns | column_bit
                updated[next_columns] = max(
                    updated.get(next_columns, float("-inf")), current + float(value)
                )
        best_by_used_columns = updated
    return max(best_by_used_columns.values(), default=0.0)


def json_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    return [1.0 if parse_json_object(completion_text(item)) is not None else 0.0 for item in completions]


def schema_reward(
    completions: Sequence[Any], visible_ids: Sequence[Sequence[str]], **_: Any
) -> list[float]:
    return [
        1.0
        if not validate_payload(parse_json_object(completion_text(item)), set(visible))
        else 0.0
        for item, visible in zip(completions, visible_ids, strict=True)
    ]


def protocol_penalty(
    completions: Sequence[Any], visible_ids: Sequence[Sequence[str]], **_: Any
) -> list[float]:
    """Make strict protocol validity a gate instead of a saturating bonus.

    JSON and schema rewards are highly correlated and quickly saturate.  In
    the semantic-gated profile, a valid payload gets no positive reward merely
    for formatting while an invalid payload is penalized.  Positive action or
    factual rewards are independently masked by :func:`protocol_gated`.
    """
    return [
        0.0
        if not validate_payload(parse_json_object(completion_text(item)), set(visible))
        else -1.0
        for item, visible in zip(completions, visible_ids, strict=True)
    ]


def protocol_violation_penalty(
    completions: Sequence[Any], visible_ids: Sequence[Sequence[str]], **_: Any
) -> list[float]:
    """Give malformed outputs a graded training signal.

    A binary protocol penalty treats a truncated JSON object, one unknown E-ID,
    and eight duplicated/unknown facts as the same failure.  That makes it
    difficult for GRPO to learn which local error to remove.  Keep valid
    outputs at zero, but make the penalty grow with the number of structural
    violations.  Invalid JSON receives a fixed stronger penalty because no
    field-level diagnosis is possible.
    """
    penalties: list[float] = []
    for item, visible in zip(completions, visible_ids, strict=True):
        problems = validate_payload(parse_json_object(completion_text(item)), set(visible))
        if not problems:
            penalties.append(0.0)
        elif "invalid_json" in problems:
            penalties.append(-1.5)
        else:
            penalties.append(-min(1.5, 0.25 * len(problems)))
    return penalties


def concise_fact_penalty(completions: Sequence[Any], **_: Any) -> list[float]:
    """Discourage padding an answer with unnecessary facts.

    The protocol permits up to eight facts for genuinely multi-part questions,
    but the canonical prompt says the usual answer should be 1--4 facts.  This
    soft penalty also applies to structurally invalid answer payloads when their
    fact list is parseable.  This is intentional: a malformed answer must not
    bypass the anti-padding signal by first introducing a duplicate or invalid
    E-ID.  It remains weaker than the evidence-only semantic reward, so complete
    multi-fact answers can still win when GLM judges their coverage as complete.
    """
    penalties: list[float] = []
    for item in completions:
        payload = parse_json_object(completion_text(item))
        if not payload or payload.get("next_action") != "answer_directly":
            penalties.append(0.0)
            continue
        facts = payload.get("supported_facts")
        count = len(facts) if isinstance(facts, list) else 0
        penalties.append(-min(1.0, max(0, count - 4) / 4.0))
    return penalties


def protocol_gated(reward_func: Any) -> Any:
    """Return a reward function whose positive credit requires valid schema."""

    def gated_reward(
        completions: Sequence[Any], visible_ids: Sequence[Sequence[str]], **kwargs: Any
    ) -> list[float]:
        raw = reward_func(completions=completions, visible_ids=visible_ids, **kwargs)
        if len(raw) != len(completions):
            raise ValueError(f"{reward_func.__name__} returned the wrong reward count")
        return [
            float(value)
            if not validate_payload(parse_json_object(completion_text(item)), set(visible))
            else 0.0
            for item, visible, value in zip(completions, visible_ids, raw, strict=True)
        ]

    gated_reward.__name__ = f"gated_{reward_func.__name__}"
    gated_reward.__doc__ = f"Protocol-gated form of {reward_func.__name__}."
    return gated_reward


def action_reward(
    completions: Sequence[Any], gold_action: Sequence[str], **_: Any
) -> list[float]:
    result = []
    for item, gold in zip(completions, gold_action, strict=True):
        payload = parse_json_object(completion_text(item))
        result.append(1.0 if payload and payload.get("next_action") == gold else 0.0)
    return result


def claim_citation_reward(
    completions: Sequence[Any],
    gold_action: Sequence[str],
    gold_fact_bindings: Sequence[Sequence[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    """Match each predicted claim jointly on text and its local citations.

    Flattening every E-ID in the answer lets a policy attach the right global
    set to the wrong claims.  This maximum bipartite-style score rewards a
    predicted claim only when both its content and its own citation IDs agree
    with one hidden teacher claim.
    """
    rewards: list[float] = []
    for item, gold, expected_bindings in zip(
        completions, gold_action, gold_fact_bindings, strict=True
    ):
        payload = parse_json_object(completion_text(item))
        if gold != "answer_directly":
            rewards.append(0.0)
            continue
        predicted_bindings = supported_fact_bindings(payload)
        if not predicted_bindings or not expected_bindings:
            rewards.append(0.0)
            continue
        scores: list[list[float]] = []
        for predicted in predicted_bindings:
            row_scores: list[float] = []
            for expected in expected_bindings:
                fact_score = reference_fact_similarity(
                    [str(predicted["fact"])], [str(expected["fact"])]
                )
                predicted_ids = set(predicted["evidence_ids"])
                expected_ids = set(expected["evidence_ids"])
                union = predicted_ids | expected_ids
                citation_score = len(predicted_ids & expected_ids) / len(union) if union else 0.0
                row_scores.append(fact_score * citation_score)
            scores.append(row_scores)
        matched_score = maximum_bipartite_score(scores)
        rewards.append(matched_score / max(len(predicted_bindings), len(expected_bindings)))
    return rewards


def reference_fact_reward(
    completions: Sequence[Any],
    gold_action: Sequence[str],
    gold_fact_bindings: Sequence[Sequence[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    """Reference-based factual-content reward, hidden from the policy prompt."""
    rewards: list[float] = []
    for item, gold, expected in zip(completions, gold_action, gold_fact_bindings, strict=True):
        if gold != "answer_directly":
            rewards.append(0.0)
            continue
        payload = parse_json_object(completion_text(item))
        expected_texts = [str(binding.get("fact") or "") for binding in expected]
        rewards.append(reference_fact_similarity(supported_fact_texts(payload), expected_texts))
    return rewards


def duplicate_fact_penalty(
    completions: Sequence[Any], gold_action: Sequence[str], **_: Any
) -> list[float]:
    penalties: list[float] = []
    for item, gold in zip(completions, gold_action, strict=True):
        if gold != "answer_directly":
            penalties.append(0.0)
            continue
        facts = supported_fact_texts(parse_json_object(completion_text(item)))
        normalized = [normalized_fact_text(fact) for fact in facts]
        duplicate_count = len(normalized) - len(set(normalized))
        penalties.append(-min(1.0, duplicate_count / max(1, len(normalized))))
    return penalties


def near_duplicate_fact_penalty(completions: Sequence[Any], **_: Any) -> list[float]:
    """Penalize semantically redundant-looking padding without using gold.

    Each fact after the first is compared with earlier facts.  Similarity below
    the conservative threshold is free; increasingly close paraphrases are
    penalized continuously.  Exact duplicates are already protocol-invalid,
    so this primarily handles small wording changes used to pad an answer.
    """
    penalties: list[float] = []
    for item in completions:
        facts = supported_fact_texts(parse_json_object(completion_text(item)))
        if len(facts) < 2:
            penalties.append(0.0)
            continue
        redundant_mass = 0.0
        for index in range(1, len(facts)):
            closest = max(fact_pair_similarity(facts[index], prior) for prior in facts[:index])
            redundant_mass += max(
                0.0,
                (closest - NEAR_DUPLICATE_THRESHOLD) / (1.0 - NEAR_DUPLICATE_THRESHOLD),
            )
        penalties.append(-min(1.0, redundant_mass / (len(facts) - 1)))
    return penalties


def premature_answer_penalty(
    completions: Sequence[Any], gold_action: Sequence[str], **_: Any
) -> list[float]:
    penalties = []
    for item, gold in zip(completions, gold_action, strict=True):
        payload = parse_json_object(completion_text(item))
        action = str((payload or {}).get("next_action") or "")
        penalties.append(-1.0 if gold != "answer_directly" and action == "answer_directly" else 0.0)
    return penalties


def build_rule_reward_stack(profile: str) -> tuple[list[Any], list[float]]:
    """Build a named, reproducible rule-reward profile.

    ``legacy`` and ``semantic-gated`` exactly preserve completed experiments.
    ``protocol-gated-rules`` is an explicit name for the latter.  The new
    ``glm-semantic-gated`` profile contains no hidden-gold positive reward:
    evidence-only GLM judgement is appended by :func:`main` and dominates the
    strict protocol and near-duplicate safeguards returned here.
    """
    if profile == "legacy":
        return (
            [
                json_reward,
                schema_reward,
                action_reward,
                claim_citation_reward,
                reference_fact_reward,
                duplicate_fact_penalty,
                premature_answer_penalty,
            ],
            [1.0, 1.0, 1.5, 1.5, 1.0, 0.5, 0.5],
        )
    if profile in {"semantic-gated", "protocol-gated-rules"}:
        return (
            [
                protocol_penalty,
                protocol_gated(action_reward),
                protocol_gated(claim_citation_reward),
                protocol_gated(reference_fact_reward),
                duplicate_fact_penalty,
                premature_answer_penalty,
            ],
            [1.0, 0.75, 1.0, 1.0, 0.75, 1.0],
        )
    if profile in {"glm-semantic-gated", "glm-precision-gated"}:
        return (
            [
                protocol_penalty,
                protocol_gated(near_duplicate_fact_penalty),
            ],
            # Invalid protocol must be worse than the lowest possible GLM
            # semantic reward.  Near-duplicate padding is a material but
            # secondary penalty; evidence entailment remains the main signal.
            [2.0, 1.5],
        )
    if profile == "glm-precision-structural":
        return (
            [
                protocol_penalty,
                protocol_violation_penalty,
                protocol_gated(near_duplicate_fact_penalty),
                protocol_gated(concise_fact_penalty),
            ],
            # Keep the evidence-only GLM judge dominant while adding graded
            # structural feedback for malformed JSON, bad E-IDs, and padding.
            [1.5, 1.5, 1.5, 0.75],
        )
    if profile == "glm-precision-structural-v2":
        # Unlike the historical structural profile, anti-padding rewards are
        # deliberately *not* schema-gated.  Exact duplicate facts make a
        # payload invalid, so gating these components would zero the only
        # targeted feedback for the failure mode we want to remove.
        return (
            [
                protocol_penalty,
                protocol_violation_penalty,
                near_duplicate_fact_penalty,
                concise_fact_penalty,
            ],
            [1.5, 1.5, 2.0, 1.0],
        )
    raise ValueError(f"unknown reward profile: {profile}")


def tokenized_length(tokenizer: Any, prompt: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], Sequence) and not isinstance(input_ids[0], (str, bytes)):
        if len(input_ids) != 1:
            raise ValueError("expected one tokenized prompt")
        input_ids = input_ids[0]
    return len(input_ids)


def read_training_examples(
    path: Path, tokenizer: Any, *, max_prompt_tokens: int
) -> tuple[list[TrainingExample], Counter[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array: {path}")
    examples: list[TrainingExample] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row.get("task_type") != "grounded_action_generation":
            counts["skip_non_grounded"] += 1
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            counts["skip_conversations"] += 1
            continue
        user = str(conversations[0].get("value") or "")
        gold = parse_json_object(conversations[-1].get("value"))
        action = str((gold or {}).get("next_action") or "")
        visible = EVIDENCE_RE.findall(user)
        if action not in ACTIONS or not visible or validate_payload(gold, set(visible)):
            counts["skip_invalid_gold"] += 1
            continue
        prompt = [
            {"role": "system", "content": str(row.get("system") or "")},
            {"role": "user", "content": user},
        ]
        n_tokens = tokenized_length(tokenizer, prompt)
        if n_tokens > max_prompt_tokens:
            counts["skip_prompt_too_long"] += 1
            continue
        examples.append(
            TrainingExample(
                row_id=str(row.get("id") or row.get("task_id") or index),
                prompt=prompt,
                prompt_tokens=n_tokens,
                visible_ids=visible,
                gold_action=action,
                gold_fact_bindings=supported_fact_bindings(gold),
            )
        )
        counts[f"keep:{action}"] += 1
    return examples, counts


def select_examples(
    examples: list[TrainingExample], *, max_rows: int, selection_order: str, seed: int
) -> list[TrainingExample]:
    if not max_rows or max_rows >= len(examples):
        selected = list(examples)
        random.Random(seed).shuffle(selected)
        return selected
    if selection_order == "random":
        selected = list(examples)
        random.Random(seed).shuffle(selected)
        return selected[:max_rows]

    if selection_order == "stratified-action-length":
        groups: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in examples:
            groups[example.gold_action].append(example)
        for values in groups.values():
            values.sort(key=lambda item: (item.prompt_tokens, item.row_id))

        # Allocate evenly across actions, then spread each allocation from the
        # shortest through the longest context.  Smoke tests therefore cover
        # both state-machine branches and the long-evidence failure mode.
        allocation = Counter(ACTIONS[index % len(ACTIONS)] for index in range(max_rows))
        sampled: dict[str, list[TrainingExample]] = {}
        for action in ACTIONS:
            values = groups[action]
            count = min(allocation[action], len(values))
            if count == 0:
                sampled[action] = []
            elif count == 1:
                sampled[action] = [values[-1]]
            else:
                indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
                sampled[action] = [values[index] for index in indices]
        selected: list[TrainingExample] = []
        while len(selected) < max_rows and any(sampled.values()):
            for action in ACTIONS:
                if sampled[action] and len(selected) < max_rows:
                    selected.append(sampled[action].pop(0))
        if len(selected) < max_rows:
            selected_ids = {item.row_id for item in selected}
            remainder = [item for item in examples if item.row_id not in selected_ids]
            random.Random(seed).shuffle(remainder)
            selected.extend(remainder[: max_rows - len(selected)])
        return selected

    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[example.gold_action].append(example)
    for values in groups.values():
        values.sort(
            key=lambda item: (item.prompt_tokens, item.row_id),
            reverse=selection_order == "stratified-longest",
        )
    selected: list[TrainingExample] = []
    while len(selected) < max_rows and any(groups.values()):
        for action in ACTIONS:
            if groups[action] and len(selected) < max_rows:
                selected.append(groups[action].pop(0))
    return selected


def adapter_key_coverage(model: Any, adapter_path: Path) -> tuple[int, int, list[str]]:
    from safetensors import safe_open

    adapter_file = adapter_path / "adapter_model.safetensors"
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        expected = set(handle.keys())
    actual = {
        name.replace(".default.", ".")
        for name, _ in model.named_parameters()
        if ".lora_" in name and ".default." in name
    }
    missing = sorted(expected - actual)
    return len(expected) - len(missing), len(expected), missing


def build_training_diagnostic(logs: Mapping[str, Any], max_grad_norm: float) -> dict[str, Any]:
    """Normalize one Trainer log event and expose clipping diagnostics."""
    diagnostic = {str(key): value for key, value in logs.items()}
    try:
        pre_clip = float(logs["grad_norm"])
    except (KeyError, TypeError, ValueError):
        return diagnostic
    post_clip = min(pre_clip, max_grad_norm)
    diagnostic.update(
        {
            "grad_norm_pre_clip": pre_clip,
            "grad_norm_post_clip_bound": post_clip,
            "grad_clip_coefficient": min(1.0, max_grad_norm / max(pre_clip, 1e-12)),
            "grad_was_clipped": pre_clip > max_grad_norm,
        }
    )
    return diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--selection-order",
        choices=(
            "random",
            "stratified-shortest",
            "stratified-longest",
            "stratified-action-length",
        ),
        default="random",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=10000)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="legacy")
    parser.add_argument("--glm-semantic-reward", action="store_true")
    parser.add_argument("--glm-api-key-env", default="BIGMODEL_API_KEY")
    parser.add_argument(
        "--glm-endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    )
    parser.add_argument("--glm-model", default="glm-5.3")
    parser.add_argument("--glm-reasoning-effort", default="medium")
    parser.add_argument("--glm-timeout", type=float, default=180.0)
    parser.add_argument("--glm-max-tokens", type=int, default=4096)
    parser.add_argument("--glm-max-attempts", type=int, default=1)
    parser.add_argument("--glm-workers", type=int, default=1)
    parser.add_argument(
        "--glm-max-consecutive-failures",
        type=int,
        default=3,
        help=(
            "abort after this many consecutive transient judge failures; "
            "set to 0 to keep failed groups neutral and never trip the circuit breaker"
        ),
    )
    parser.add_argument("--glm-ca-bundle", type=Path)
    parser.add_argument("--glm-reward-weight", type=float)
    parser.add_argument("--glm-cache", type=Path)
    parser.add_argument("--glm-failures", type=Path)
    return parser


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.glm_max_consecutive_failures < 0:
        raise ValueError("glm-max-consecutive-failures must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.reward_profile in {
        "glm-semantic-gated",
        "glm-precision-gated",
        "glm-precision-structural",
        "glm-precision-structural-v2",
    } and not args.glm_semantic_reward:
        raise ValueError(f"{args.reward_profile} requires --glm-semantic-reward")
    if args.glm_semantic_reward and not os.environ.get(args.glm_api_key_env, "").strip():
        raise ValueError(f"missing GLM API key environment variable: {args.glm_api_key_env}")
    if args.glm_reward_weight is None:
        args.glm_reward_weight = (
            GLM_SEMANTIC_DEFAULT_WEIGHT
            if args.reward_profile
            in {
                "glm-semantic-gated",
                "glm-precision-gated",
                "glm-precision-structural",
                "glm-precision-structural-v2",
            }
            else 1.0
        )
    if args.glm_reward_weight <= 0:
        raise ValueError("glm-reward-weight must be positive")
    if (
        args.reward_profile
        in {
            "glm-semantic-gated",
            "glm-precision-gated",
            "glm-precision-structural",
            "glm-precision-structural-v2",
        }
        and args.glm_reward_weight < GLM_SEMANTIC_MIN_WEIGHT
    ):
        raise ValueError(
            f"{args.reward_profile} requires --glm-reward-weight >= {GLM_SEMANTIC_MIN_WEIGHT}"
        )
    generation_batch_size = args.batch_size * args.gradient_accumulation_steps
    if generation_batch_size % args.num_generations:
        raise ValueError(
            "batch_size * gradient_accumulation_steps must be divisible by num_generations"
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    validate_runtime_args(args)
    generation_batch_size = args.batch_size * args.gradient_accumulation_steps

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    examples, input_counts = read_training_examples(
        args.train_file, tokenizer, max_prompt_tokens=args.max_prompt_tokens
    )
    selected = select_examples(
        examples,
        max_rows=args.max_rows,
        selection_order=args.selection_order,
        seed=args.seed,
    )
    if len(selected) < args.num_generations:
        raise ValueError(f"too few selected examples: {len(selected)}")
    selected_counts = Counter(item.gold_action for item in selected)
    print(
        json.dumps(
            {
                "input_counts": dict(input_counts),
                "selected_rows": len(selected),
                "selected_actions": dict(selected_counts),
                "prompt_tokens": {
                    "min": min(item.prompt_tokens for item in selected),
                    "max": max(item.prompt_tokens for item in selected),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dataset = Dataset.from_list(
        [
            {
                "row_id": item.row_id,
                "prompt": item.prompt,
                "visible_ids": item.visible_ids,
                "gold_action": item.gold_action,
                "gold_fact_bindings": item.gold_fact_bindings,
                "judge_context": item.prompt[-1]["content"],
            }
            for item in selected
        ]
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(args.sft_adapter),
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    loaded, total, missing = adapter_key_coverage(model, args.sft_adapter)
    print(f"adapter key coverage: {loaded}/{total}", flush=True)
    if missing:
        raise RuntimeError(f"adapter/model mismatch; first missing keys: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_funcs, reward_weights = build_rule_reward_stack(args.reward_profile)
    if args.glm_semantic_reward:
        # The callable itself is appended after the judge is constructed.
        reward_component_names = [func.__name__ for func in reward_funcs] + [
            "glm_semantic_reward"
        ]
        effective_reward_weights = [*reward_weights, args.glm_reward_weight]
    else:
        reward_component_names = [func.__name__ for func in reward_funcs]
        effective_reward_weights = list(reward_weights)

    manifest = {
        "protocol": "grounded_action_exx_v1",
        "method": "GRPO with verifiable rewards (fixed evidence)",
        "limitations": [
            "reference fact overlap is not semantic entailment verification",
            "does not execute interactive retrieval actions",
        ],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_counts": dict(input_counts),
        "selected_actions": dict(selected_counts),
        "selected_ids": [item.row_id for item in selected],
        "adapter_key_coverage": {"loaded": loaded, "total": total},
        "reward_profile": args.reward_profile,
        "reward_stack": [
            {"name": name, "weight": weight}
            for name, weight in zip(
                reward_component_names, effective_reward_weights, strict=True
            )
        ],
        "glm_semantic_judge": {
            "enabled": args.glm_semantic_reward,
            "protocol": SEMANTIC_JUDGE_PROTOCOL if args.glm_semantic_reward else None,
            "gold_or_reference_visible": False,
            "score_profile": (
                "precision-v1"
                if args.reward_profile
                in {
                    "glm-precision-gated",
                    "glm-precision-structural",
                    "glm-precision-structural-v2",
                }
                else "balanced-v3"
            ),
            "group_quality_threshold": (
                0.75
                if args.reward_profile
                in {
                    "glm-precision-gated",
                    "glm-precision-structural",
                    "glm-precision-structural-v2",
                }
                else None
            ),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.glm_semantic_reward:
        from glm_exx_semantic_reward import GlmEvidenceJudge, make_glm_semantic_reward

        # Keep API checkpoints next to, rather than inside, the model output.
        # A partial cache must remain reusable even when a failed model run is
        # intentionally restarted into a fresh versioned directory.
        cache_path = args.glm_cache or args.output_dir.with_name(
            f"{args.output_dir.name}.glm_semantic_cache.jsonl"
        )
        failures_path = args.glm_failures or args.output_dir.with_name(
            f"{args.output_dir.name}.glm_semantic_failures.jsonl"
        )
        judge = GlmEvidenceJudge(
            endpoint=args.glm_endpoint,
            api_key=os.environ[args.glm_api_key_env].strip(),
            model=args.glm_model,
            cache_path=cache_path,
            failures_path=failures_path,
            timeout=args.glm_timeout,
            max_tokens=args.glm_max_tokens,
            max_attempts=args.glm_max_attempts,
            reasoning_effort=args.glm_reasoning_effort,
            workers=args.glm_workers,
            max_consecutive_failures=args.glm_max_consecutive_failures,
            ca_bundle=args.glm_ca_bundle,
            strict_json=args.reward_profile
            in {
                "glm-semantic-gated",
                "glm-precision-gated",
                "glm-precision-structural",
                "glm-precision-structural-v2",
            },
            score_profile=(
                "precision-v1"
                if args.reward_profile
                in {
                    "glm-precision-gated",
                    "glm-precision-structural",
                    "glm-precision-structural-v2",
                }
                else "balanced-v3"
            ),
        )
        reward_funcs.append(
            make_glm_semantic_reward(
                judge,
                gate_invalid=args.reward_profile
                in {
                    "glm-semantic-gated",
                    "glm-precision-gated",
                    "glm-precision-structural",
                    "glm-precision-structural-v2",
                },
                group_quality_threshold=(
                    0.75
                    if args.reward_profile
                    in {
                        "glm-precision-gated",
                        "glm-precision-structural",
                        "glm-precision-structural-v2",
                    }
                    else None
                ),
            )
        )
        reward_weights.append(args.glm_reward_weight)

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        chat_template_kwargs={"enable_thinking": False},
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        torch_empty_cache_steps=1,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps,
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        mask_truncated_completions=True,
        reward_weights=reward_weights,
    )

    diagnostics_path = args.output_dir / "training_diagnostics.jsonl"

    class TrainingDiagnosticsCallback(TrainerCallback):
        def on_log(self, _args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> Any:
            if not isinstance(logs, Mapping):
                return control
            row = build_training_diagnostic(logs, args.max_grad_norm)
            row.setdefault("step", state.global_step)
            with diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            return control

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        callbacks=[TrainingDiagnosticsCallback()],
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SAVED {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
