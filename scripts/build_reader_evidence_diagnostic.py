#!/usr/bin/env python3
"""Build the reviewed reader-by-evidence diagnostic matrix.

This script converts frozen retrieval snapshots plus explicit review
annotations into three canonical Exx prompt conditions:

1. ``minimal_sufficient``: reviewer-approved minimum sufficient documents.
2. ``sufficient_with_distractors``: the same support plus reviewed real
   distractors.
3. ``actual_prompt``: the documents selected by the frozen runtime snapshot.

The builder deliberately does not infer "minimum sufficient" from citation
counts and does not treat arbitrary non-gold documents as distractors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.inference.evidence.rendering import (  # noqa: E402
    render_short_evidence_brief,
)
from asa_arknight_story_agent.inference.evidence.texts import (  # noqa: E402
    prefer_direct_prompt_text,
)
from asa_arknight_story_agent.inference.generation.exx_prompt import (  # noqa: E402
    EXX_SYSTEM_PROMPT,
    render_exx_user_prompt,
)


CONDITIONS = (
    "minimal_sufficient",
    "sufficient_with_distractors",
    "actual_prompt",
)
EVIDENCE_HEADER_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)


class DiagnosticBuildError(ValueError):
    """Raised when reviewed inputs cannot support an auditable matrix."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            for key in ("rows", "records", "items", "results"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
            else:
                values = [value]
        else:
            raise DiagnosticBuildError(f"expected JSON object/array: {path}")
    if not all(isinstance(row, dict) for row in values):
        raise DiagnosticBuildError(f"all rows must be JSON objects: {path}")
    return [row for row in values if isinstance(row, dict)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def unique_ids(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise DiagnosticBuildError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        item_id = str(item).strip()
        if not item_id:
            raise DiagnosticBuildError(f"{field} contains an empty document ID")
        if item_id in result:
            raise DiagnosticBuildError(f"{field} contains duplicate ID: {item_id}")
        result.append(item_id)
    return result


def annotation_reviewer(annotation: dict[str, Any]) -> str:
    reviewer = annotation.get("reviewer")
    if isinstance(reviewer, str) and reviewer.strip():
        return reviewer.strip()
    reviewers = annotation.get("reviewers")
    if isinstance(reviewers, list):
        names = [str(item).strip() for item in reviewers if str(item).strip()]
        if names:
            return ",".join(names)
    raise DiagnosticBuildError("approved annotation must name reviewer/reviewers")


def validate_answer_facts(
    annotation: dict[str, Any],
    *,
    minimal_ids: list[str],
) -> list[dict[str, Any]]:
    facts = annotation.get("answer_facts")
    if not isinstance(facts, list) or not facts:
        raise DiagnosticBuildError("answer_facts must be a non-empty list")
    minimal = set(minimal_ids)
    normalized: list[dict[str, Any]] = []
    for fact_index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise DiagnosticBuildError(f"answer_facts[{fact_index}] must be an object")
        text = str(fact.get("fact") or "").strip()
        if not text:
            raise DiagnosticBuildError(f"answer_facts[{fact_index}] has no fact text")
        groups_value = fact.get("support_groups")
        if not isinstance(groups_value, list) or not groups_value:
            raise DiagnosticBuildError(
                f"answer_facts[{fact_index}].support_groups must be non-empty"
            )
        groups: list[list[str]] = []
        for group_index, group in enumerate(groups_value):
            ids = unique_ids(
                group,
                field=(
                    f"answer_facts[{fact_index}].support_groups[{group_index}]"
                ),
            )
            if not 1 <= len(ids) <= 2:
                raise DiagnosticBuildError(
                    "each support group must contain 1-2 documents; split the "
                    f"fact before Exx evaluation: answer_facts[{fact_index}]"
                )
            groups.append(ids)
        if not any(set(group).issubset(minimal) for group in groups):
            raise DiagnosticBuildError(
                f"minimal evidence does not support answer_facts[{fact_index}]"
            )
        normalized.append({"fact": text, "support_groups": groups})
    return normalized


def validate_annotation(
    annotation: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    available_ids: set[str],
) -> dict[str, Any]:
    row_id = str(annotation.get("id") or "").strip()
    if row_id != str(snapshot.get("id") or "").strip():
        raise DiagnosticBuildError(f"annotation/snapshot ID mismatch: {row_id}")
    if annotation.get("review_status") != "approved":
        raise DiagnosticBuildError(f"{row_id}: review_status must be approved")
    reviewer = annotation_reviewer(annotation)
    annotated_family = str(annotation.get("question_family") or "").strip()
    snapshot_family = str(
        snapshot.get("question_family") or snapshot.get("question") or row_id
    ).strip()
    if annotated_family and annotated_family != snapshot_family:
        raise DiagnosticBuildError(f"{row_id}: question_family mismatch")
    minimal_ids = unique_ids(
        annotation.get("minimal_sufficient_evidence_ids"),
        field="minimal_sufficient_evidence_ids",
    )
    distractor_ids = unique_ids(
        annotation.get("distractor_evidence_ids"),
        field="distractor_evidence_ids",
    )
    if not minimal_ids:
        raise DiagnosticBuildError(f"{row_id}: minimal evidence must not be empty")
    if not distractor_ids:
        raise DiagnosticBuildError(
            f"{row_id}: at least one reviewed distractor is required"
        )
    overlap = sorted(set(minimal_ids) & set(distractor_ids))
    if overlap:
        raise DiagnosticBuildError(
            f"{row_id}: distractors overlap minimal evidence: {','.join(overlap)}"
        )
    facts = validate_answer_facts(annotation, minimal_ids=minimal_ids)
    support_ids = {
        doc_id
        for fact in facts
        for group in fact["support_groups"]
        for doc_id in group
    }
    disguised_support = sorted(set(distractor_ids) & support_ids)
    if disguised_support:
        raise DiagnosticBuildError(
            f"{row_id}: distractors overlap an accepted support group: "
            + ",".join(disguised_support)
        )
    requested = set(minimal_ids) | set(distractor_ids) | support_ids
    unknown = sorted(requested - available_ids)
    if unknown:
        raise DiagnosticBuildError(
            f"{row_id}: reviewed document IDs absent from snapshot: "
            + ",".join(unknown)
        )
    return {
        "id": row_id,
        "question_family": snapshot_family,
        "reviewer": reviewer,
        "minimal_ids": minimal_ids,
        "distractor_ids": distractor_ids,
        "answer_facts": facts,
        "notes": str(annotation.get("notes") or "").strip(),
    }


def snapshot_hit_to_runtime(hit: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(hit.get("id") or "").strip()
    clean_text = str(hit.get("clean_text") or hit.get("search_text") or "").strip()
    if not doc_id or not clean_text:
        raise DiagnosticBuildError("snapshot hit must contain id and clean_text")
    document = {
        "id": doc_id,
        "clean_text": clean_text,
        "search_text": str(hit.get("search_text") or clean_text),
        "activity_name": hit.get("activity_name"),
        "story_name": hit.get("story_name"),
        "stage_code": hit.get("stage_code"),
        "source_path": hit.get("source_path"),
    }
    runtime = {
        "doc_index": hit.get("doc_index"),
        "document": document,
        "evidence_chain_text": hit.get("evidence_chain_text"),
        "evidence_chain_roles": hit.get("evidence_chain_roles"),
        "evidence_chain_member_ids": hit.get("evidence_chain_member_ids"),
    }
    return prefer_direct_prompt_text(runtime)


def documents_from_snapshot(
    snapshot: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    stages = snapshot.get("stages")
    if not isinstance(stages, dict):
        raise DiagnosticBuildError("snapshot has no stages object")
    documents: dict[str, dict[str, Any]] = {}
    for stage_name in ("rerank", "prompt", "fusion", "dense", "sparse", "minirag"):
        hits = stages.get(stage_name) or []
        if not isinstance(hits, list):
            raise DiagnosticBuildError(f"snapshot stage {stage_name} is not a list")
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            doc_id = str(hit.get("id") or "").strip()
            if doc_id and doc_id not in documents:
                documents[doc_id] = snapshot_hit_to_runtime(hit)
    prompt_hits = stages.get("prompt")
    if not isinstance(prompt_hits, list) or not prompt_hits:
        raise DiagnosticBuildError("snapshot prompt stage must be non-empty")
    prompt_ids = [
        str(hit.get("id") or "").strip()
        for hit in prompt_hits
        if isinstance(hit, dict) and str(hit.get("id") or "").strip()
    ]
    if not prompt_ids:
        raise DiagnosticBuildError("snapshot prompt stage has no document IDs")
    return documents, prompt_ids


def shuffled_ids(
    ids: list[str],
    *,
    seed: int,
    source_id: str,
    condition: str,
) -> list[str]:
    result = list(ids)
    stable = hashlib.sha256(
        f"{seed}:{source_id}:{condition}".encode("utf-8")
    ).digest()
    random.Random(stable).shuffle(result)
    return result


def render_condition_evidence(
    document_ids: list[str],
    *,
    documents: dict[str, dict[str, Any]],
    max_chars_per_doc: int,
    max_total_chars: int,
) -> tuple[str, list[str], dict[str, str]]:
    if len(document_ids) != len(set(document_ids)):
        raise DiagnosticBuildError("condition contains duplicate document IDs")
    items = [documents[doc_id] for doc_id in document_ids]
    evidence_text = render_short_evidence_brief(
        items,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
        preserve_complete_evidence=True,
        label_on_own_line=True,
    )
    labels = EVIDENCE_HEADER_RE.findall(evidence_text)
    if not labels:
        raise DiagnosticBuildError("evidence budget removed every document")
    visible_ids = document_ids[: len(labels)]
    mapping = {
        doc_id: f"E{index}"
        for index, doc_id in enumerate(visible_ids, start=1)
    }
    return evidence_text, visible_ids, mapping


def support_group_for_visible(
    fact: dict[str, Any],
    visible_ids: set[str],
) -> list[str] | None:
    for group in fact["support_groups"]:
        if set(group).issubset(visible_ids):
            return group
    return None


def follow_up_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    hypothesis = snapshot.get("hypothesis")
    hypothesis = hypothesis if isinstance(hypothesis, dict) else {}
    question = str(snapshot.get("question") or "").strip()
    follow_up = {
        "question": str(hypothesis.get("question") or question),
        "query_type": str(hypothesis.get("query_type") or "reasoning"),
        "entities": [str(item) for item in hypothesis.get("entities") or []],
        "keywords": [str(item) for item in hypothesis.get("keywords") or []],
        "expected_answer_type": str(
            hypothesis.get("expected_answer_type") or ""
        ),
    }
    dialogue_context = str(hypothesis.get("dialogue_context") or "").strip()
    if dialogue_context:
        follow_up["dialogue_context"] = dialogue_context
    return {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": follow_up,
    }


def expected_payload(
    *,
    snapshot: dict[str, Any],
    answer_facts: list[dict[str, Any]],
    visible_ids: list[str],
    evidence_mapping: dict[str, str],
) -> tuple[dict[str, Any], str]:
    visible = set(visible_ids)
    selected_groups = [
        support_group_for_visible(fact, visible) for fact in answer_facts
    ]
    if any(group is None for group in selected_groups):
        return follow_up_payload(snapshot), "insufficient"
    supported_facts = []
    for fact, group in zip(answer_facts, selected_groups, strict=True):
        assert group is not None
        supported_facts.append(
            {
                "fact": fact["fact"],
                "evidence_ids": [evidence_mapping[doc_id] for doc_id in group],
            }
        )
    return {
        "next_action": "answer_directly",
        "supported_facts": supported_facts,
    }, "sufficient"


def build_condition_row(
    *,
    snapshot: dict[str, Any],
    review: dict[str, Any],
    condition: str,
    document_ids: list[str],
    documents: dict[str, dict[str, Any]],
    max_chars_per_doc: int,
    max_total_chars: int,
    round_value: str,
) -> dict[str, Any]:
    evidence_text, visible_ids, mapping = render_condition_evidence(
        document_ids,
        documents=documents,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
    )
    payload, sufficiency = expected_payload(
        snapshot=snapshot,
        answer_facts=review["answer_facts"],
        visible_ids=visible_ids,
        evidence_mapping=mapping,
    )
    if condition != "actual_prompt" and sufficiency != "sufficient":
        raise DiagnosticBuildError(
            f"{review['id']}:{condition} lost required evidence under the "
            "configured prompt budget"
        )
    question = str(snapshot.get("question") or "").strip()
    hypothesis = snapshot.get("hypothesis")
    hypothesis = hypothesis if isinstance(hypothesis, dict) else {}
    prompt = render_exx_user_prompt(
        question=question,
        hypothesis=hypothesis,
        round_value=round_value,
        evidence_text=evidence_text,
    )
    return {
        "id": f"{review['id']}::{condition}",
        "task_type": "grounded_action_generation",
        "system": EXX_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": compact_json(payload)},
        ],
        "diagnostic": {
            "source_id": review["id"],
            "question_family": review["question_family"],
            "condition": condition,
            "review_status": "approved",
            "reviewer": review["reviewer"],
            "reference_sufficiency": sufficiency,
            "expected_action": payload["next_action"],
            "selected_corpus_ids": document_ids,
            "visible_corpus_ids": visible_ids,
            "evidence_id_to_corpus_id": {
                evidence_id: doc_id for doc_id, evidence_id in mapping.items()
            },
            "evidence_chars": len(evidence_text),
            "notes": review["notes"],
        },
    }


def build_matrix(
    *,
    snapshots: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    seed: int,
    max_chars_per_doc: int,
    max_total_chars: int,
    round_value: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshots_by_id: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        row_id = str(snapshot.get("id") or "").strip()
        if not row_id or row_id in snapshots_by_id:
            raise DiagnosticBuildError("snapshot IDs must be non-empty and unique")
        snapshots_by_id[row_id] = snapshot
    annotations_by_id: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        row_id = str(annotation.get("id") or "").strip()
        if not row_id or row_id in annotations_by_id:
            raise DiagnosticBuildError("annotation IDs must be non-empty and unique")
        annotations_by_id[row_id] = annotation
    unknown_annotations = sorted(set(annotations_by_id) - set(snapshots_by_id))
    if unknown_annotations:
        raise DiagnosticBuildError(
            "annotations reference unknown snapshots: " + ",".join(unknown_annotations)
        )
    matrix: list[dict[str, Any]] = []
    families: set[str] = set()
    condition_counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()
    for row_id in sorted(annotations_by_id):
        snapshot = snapshots_by_id[row_id]
        documents, actual_prompt_ids = documents_from_snapshot(snapshot)
        review = validate_annotation(
            annotations_by_id[row_id],
            snapshot=snapshot,
            available_ids=set(documents),
        )
        if review["question_family"] in families:
            raise DiagnosticBuildError(
                f"duplicate reviewed question family: {review['question_family']}"
            )
        families.add(review["question_family"])
        minimal_ids = shuffled_ids(
            review["minimal_ids"],
            seed=seed,
            source_id=row_id,
            condition="minimal_sufficient",
        )
        sufficient_with_distractors = shuffled_ids(
            review["minimal_ids"] + review["distractor_ids"],
            seed=seed,
            source_id=row_id,
            condition="sufficient_with_distractors",
        )
        condition_ids = {
            "minimal_sufficient": minimal_ids,
            "sufficient_with_distractors": sufficient_with_distractors,
            "actual_prompt": actual_prompt_ids,
        }
        for condition in CONDITIONS:
            row = build_condition_row(
                snapshot=snapshot,
                review=review,
                condition=condition,
                document_ids=condition_ids[condition],
                documents=documents,
                max_chars_per_doc=max_chars_per_doc,
                max_total_chars=max_total_chars,
                round_value=round_value,
            )
            matrix.append(row)
            condition_counts[condition] += 1
            sufficiency_counts[
                f"{condition}:{row['diagnostic']['reference_sufficiency']}"
            ] += 1
    audit = {
        "protocol": "asa_reader_evidence_diagnostic_v1",
        "review_policy": {
            "minimum_sufficient_requires_explicit_approved_annotation": True,
            "distractors_require_explicit_review": True,
            "support_groups_allow_alternative_proofs": True,
            "one_question_per_family": True,
            "automatic_gold_id_minimality_inference": False,
        },
        "seed": seed,
        "round_value": round_value,
        "prompt_budget": {
            "max_chars_per_doc": max_chars_per_doc,
            "max_total_chars": max_total_chars,
            "preserve_complete_evidence": True,
        },
        "snapshot_rows": len(snapshots),
        "reviewed_questions": len(annotations),
        "question_families": len(families),
        "matrix_rows": len(matrix),
        "condition_counts": dict(sorted(condition_counts.items())),
        "reference_sufficiency": dict(sorted(sufficiency_counts.items())),
    }
    return matrix, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--round-value", default="1/2")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {args.output_dir}")
    snapshots = read_rows(args.snapshots)
    annotations = read_rows(args.annotations)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    inference = runtime.get("inference")
    if not isinstance(inference, dict):
        parser.error("runtime config has no inference object")
    max_chars_per_doc = int(
        inference.get("prompt_evidence_max_chars_per_doc", 1000)
    )
    max_total_chars = int(
        inference.get("prompt_conclusion_evidence_max_total_chars", 8000)
    )
    matrix, audit = build_matrix(
        snapshots=snapshots,
        annotations=annotations,
        seed=args.seed,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
        round_value=args.round_value,
    )
    audit["inputs"] = {
        "snapshots": {
            "path": str(args.snapshots.resolve()),
            "sha256": sha256_file(args.snapshots),
        },
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": sha256_file(args.annotations),
        },
        "runtime_config": {
            "path": str(args.runtime_config.resolve()),
            "sha256": sha256_file(args.runtime_config),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "reader_evidence_matrix.json"
    matrix_path.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit["output"] = {
        "path": str(matrix_path.resolve()),
        "sha256": sha256_file(matrix_path),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
