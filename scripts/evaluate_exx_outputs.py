#!/usr/bin/env python3
"""Evaluate raw or normalized grounded_action_exx_v1 model outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
ACTIONS = ("answer_directly", "retrieve_more", "abstain")
FOLLOW_UP_REQUIRED = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
FOLLOW_UP_OPTIONAL = {"dialogue_context"}


def cited_evidence_ids(payload: dict[str, Any] | None) -> set[str]:
    if not payload or payload.get("next_action") != "answer_directly":
        return set()
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return set()
    return {
        str(evidence_id)
        for fact in facts
        if isinstance(fact, dict) and isinstance(fact.get("evidence_ids"), list)
        for evidence_id in fact["evidence_ids"]
    }


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


def normalized_fact_text(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", str(text or "").lower())


def supported_fact_bindings(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload or payload.get("next_action") != "answer_directly":
        return []
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return []
    return [
        {
            "fact": str(fact.get("fact") or "").strip(),
            "evidence_ids": {str(item) for item in fact.get("evidence_ids") or []},
        }
        for fact in facts
        if isinstance(fact, dict) and str(fact.get("fact") or "").strip()
    ]


def normalized_char_ngrams(texts: Sequence[str], n: int) -> Counter[str]:
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", "", "。".join(texts).lower())
    if not normalized:
        return Counter()
    if len(normalized) < n:
        return Counter({normalized: 1})
    return Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def reference_fact_similarity(predicted: Sequence[str], expected: Sequence[str]) -> float:
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


def maximum_bipartite_score(scores: Sequence[Sequence[float]]) -> float:
    if not scores or not scores[0]:
        return 0.0
    width = len(scores[0])
    if any(len(row) != width for row in scores):
        raise ValueError("ragged score matrix")
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


def claim_citation_alignment(
    predicted: dict[str, Any] | None, expected: dict[str, Any] | None
) -> float:
    predicted_bindings = supported_fact_bindings(predicted)
    expected_bindings = supported_fact_bindings(expected)
    if not predicted_bindings or not expected_bindings:
        return 0.0
    scores: list[list[float]] = []
    for predicted_binding in predicted_bindings:
        row_scores: list[float] = []
        for expected_binding in expected_bindings:
            fact_score = reference_fact_similarity(
                [predicted_binding["fact"]], [expected_binding["fact"]]
            )
            predicted_ids = predicted_binding["evidence_ids"]
            expected_ids = expected_binding["evidence_ids"]
            union = predicted_ids | expected_ids
            citation_score = len(predicted_ids & expected_ids) / len(union) if union else 0.0
            row_scores.append(fact_score * citation_score)
        scores.append(row_scores)
    score = maximum_bipartite_score(scores)
    return score / max(len(predicted_bindings), len(expected_bindings))


def question_from_prompt(prompt: str) -> str:
    match = re.search(r"^(?:question|问题)[:：]\s*(.+)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", "", question).strip("？?。！!，,；;：:\"'")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def assistant_value(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[-1].get("value") or "")
    return str(row.get("raw_output") or row.get("output") or "")


def prompt_value(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[0].get("value") or "")
    return str(row.get("prompt") or "")


def payload_from_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    direct = row.get("payload")
    if isinstance(direct, dict):
        return direct, True
    text = assistant_value(row).strip()
    try:
        value = json.loads(text)
        return (value if isinstance(value, dict) else None), isinstance(value, dict)
    except json.JSONDecodeError:
        return None, False


def validate_payload(payload: dict[str, Any] | None, visible: set[str]) -> list[str]:
    if payload is None:
        return ["invalid_json"]
    problems: list[str] = []
    forbidden = FORBIDDEN.intersection(payload)
    if forbidden:
        problems.append("legacy_fields:" + ",".join(sorted(forbidden)))
    action = str(payload.get("next_action") or "")
    if action not in ACTIONS:
        problems.append("invalid_action")
        return problems
    expected_top = {"next_action", "supported_facts"} if action == "answer_directly" else (
        {"next_action", "follow_up_hypothesis"} if action == "retrieve_more" else {"next_action", "reason"}
    )
    if set(payload) != expected_top:
        problems.append("top_schema")
    if action == "answer_directly":
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            problems.append("invalid_fact_count")
            return problems
        seen_facts: set[str] = set()
        for index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                problems.append(f"fact_{index}_schema")
                continue
            ids = fact.get("evidence_ids")
            if not isinstance(fact.get("fact"), str) or not fact.get("fact", "").strip():
                problems.append(f"fact_{index}_empty")
            fact_key = normalized_fact_text(str(fact.get("fact") or ""))
            if fact_key and fact_key in seen_facts:
                problems.append(f"fact_{index}_duplicate")
            elif fact_key:
                seen_facts.add(fact_key)
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or len({str(item) for item in ids}) != len(ids)
            ):
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible for item in ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more":
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
    elif not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
        problems.append("abstain_reason")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_rows(args.predictions)
    gold_rows = read_rows(args.gold) if args.gold else []
    gold_by_id = {str(row.get("id") or row.get("task_id") or i): row for i, row in enumerate(gold_rows)}
    counts = Counter()
    action_matrix = Counter()
    scalar_sums: Counter[str] = Counter()
    records = []
    for index, row in enumerate(predictions):
        key = str(row.get("id") or row.get("task_id") or index)
        prompt = prompt_value(row)
        question = question_from_prompt(prompt)
        visible = set(EVIDENCE_RE.findall(prompt))
        payload, strict = payload_from_row(row)
        problems = validate_payload(payload, visible)
        counts["total"] += 1
        counts["strict_json"] += int(strict)
        counts["schema_valid"] += int(not problems)
        action = str((payload or {}).get("next_action") or "invalid")
        counts[f"action:{action}"] += 1
        counts["legacy_field_rows"] += int(any(item.startswith("legacy_fields:") for item in problems))
        counts["invalid_e_id_rows"] += int(any("unknown_id" in item for item in problems))
        counts["duplicate_fact_rows"] += int(any("duplicate" in item for item in problems))
        gold = gold_by_id.get(key)
        gold_payload, _ = payload_from_row(gold) if gold else (None, False)
        gold_action = str((gold_payload or {}).get("next_action") or "")
        if gold_action:
            action_matrix[f"{gold_action}->{action}"] += 1
            counts["action_labelled"] += 1
            counts["action_correct"] += int(gold_action == action)
            counts["over_abstain"] += int(gold_action == "answer_directly" and action == "abstain")
            counts["over_retrieve"] += int(gold_action == "answer_directly" and action == "retrieve_more")
            counts["premature_answer"] += int(gold_action != "answer_directly" and action == "answer_directly")
            counts["schema_and_action_correct"] += int(not problems and gold_action == action)
        fact_similarity = None
        evidence_jaccard = None
        exact_evidence_set = None
        claim_citation_score = None
        if gold_action == "answer_directly":
            counts["gold_answer_rows"] += 1
            predicted_ids = cited_evidence_ids(payload)
            gold_ids = cited_evidence_ids(gold_payload)
            union = predicted_ids | gold_ids
            evidence_jaccard = len(predicted_ids & gold_ids) / len(union) if union else 0.0
            exact_evidence_set = predicted_ids == gold_ids
            fact_similarity = reference_fact_similarity(
                supported_fact_texts(payload), supported_fact_texts(gold_payload)
            )
            claim_citation_score = claim_citation_alignment(payload, gold_payload)
            scalar_sums["evidence_jaccard"] += evidence_jaccard
            scalar_sums["reference_fact_similarity"] += fact_similarity
            scalar_sums["claim_citation_alignment"] += claim_citation_score
            counts["exact_evidence_set"] += int(exact_evidence_set)
        records.append(
            {
                "id": key,
                "question": question,
                "question_key": normalized_question(question),
                "action": action,
                "gold_action": gold_action,
                "problems": problems,
                "evidence_jaccard": evidence_jaccard,
                "exact_evidence_set": exact_evidence_set,
                "reference_fact_similarity": fact_similarity,
                "claim_citation_alignment": claim_citation_score,
            }
        )

    total = counts["total"] or 1
    labelled = counts["action_labelled"] or 1
    gold_answer_rows = counts["gold_answer_rows"] or 1
    records_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_question[record["question_key"] or record["id"]].append(record)
    conflicting_action_families = sum(
        len({record["gold_action"] for record in family if record["gold_action"]}) > 1
        for family in records_by_question.values()
    )
    family_action_accuracy = sum(
        sum(record["action"] == record["gold_action"] for record in family) / len(family)
        for family in records_by_question.values()
    ) / (len(records_by_question) or 1)
    family_schema_action_accuracy = sum(
        sum(not record["problems"] and record["action"] == record["gold_action"] for record in family)
        / len(family)
        for family in records_by_question.values()
    ) / (len(records_by_question) or 1)
    summary = {
        "counts": dict(sorted(counts.items())),
        "rates": {
            "strict_json_rate": round(counts["strict_json"] / total, 6),
            "schema_valid_rate": round(counts["schema_valid"] / total, 6),
            "legacy_field_rate": round(counts["legacy_field_rows"] / total, 6),
            "invalid_e_id_rate": round(counts["invalid_e_id_rows"] / total, 6),
            "duplicate_fact_rate": round(counts["duplicate_fact_rows"] / total, 6),
            "action_accuracy": round(counts["action_correct"] / labelled, 6),
            "schema_and_action_accuracy": round(counts["schema_and_action_correct"] / labelled, 6),
            "over_abstain_rate": round(counts["over_abstain"] / labelled, 6),
            "over_retrieve_rate": round(counts["over_retrieve"] / labelled, 6),
            "premature_answer_rate": round(counts["premature_answer"] / labelled, 6),
            "exact_evidence_set_rate_on_gold_answers": round(
                counts["exact_evidence_set"] / gold_answer_rows, 6
            ),
            "mean_evidence_jaccard_on_gold_answers": round(
                scalar_sums["evidence_jaccard"] / gold_answer_rows, 6
            ),
            "mean_reference_fact_similarity_on_gold_answers": round(
                scalar_sums["reference_fact_similarity"] / gold_answer_rows, 6
            ),
            "mean_claim_citation_alignment_on_gold_answers": round(
                scalar_sums["claim_citation_alignment"] / gold_answer_rows, 6
            ),
            "question_family_macro_action_accuracy": round(family_action_accuracy, 6),
            "question_family_macro_schema_and_action_accuracy": round(
                family_schema_action_accuracy, 6
            ),
        },
        "question_families": {
            "unique": len(records_by_question),
            "duplicate": sum(len(family) > 1 for family in records_by_question.values()),
            "conflicting_gold_action": conflicting_action_families,
        },
        "action_matrix": dict(sorted(action_matrix.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
