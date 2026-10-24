#!/usr/bin/env python3
"""Build a content-bound alternative-proof intervention pilot.

The source annotations already contain reviewed support groups. This script
does not infer support from retrieval scores and never labels a state missing
merely because one cited ID was removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asa_arknight_story_agent.inference.generation.exx_prompt import (  # noqa: E402
    EXX_SYSTEM_PROMPT,
    render_exx_user_prompt,
)


PROTOCOL = "asa_alternative_proof_pilot_v1"
CASE_METADATA = {
    "42dcaacc6867aefabc75eb5b": {
        "short_name": "sword_identity",
        "proof_topology": "independent_single_fact_alternatives",
    },
    "9af639a51c9a8d7bf0cfb860": {
        "short_name": "rosmontis_comparison",
        "proof_topology": "highly_overlapping_redundant_alternatives",
    },
    "eb0d4f1233fa9a6d571b8f2f": {
        "short_name": "banshee_target",
        "proof_topology": "independent_single_fact_alternatives",
    },
    "ffc93a494808c61623543ea9": {
        "short_name": "freda_attribution",
        "proof_topology": "partial_subproof_alternative",
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(OrderedDict.fromkeys(values))


def collect_snapshot_documents(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for stage_name, stage_rows in (row.get("stages") or {}).items():
        for document in stage_rows or []:
            corpus_id = str(document.get("id") or "")
            text = str(document.get("clean_text") or "")
            if not corpus_id or not text:
                continue
            prior = documents.get(corpus_id)
            if prior is not None and prior["text"] != text:
                raise ValueError(f"conflicting text for {corpus_id}")
            if prior is None:
                documents[corpus_id] = {
                    "corpus_id": corpus_id,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "source_path": document.get("source_path"),
                    "seen_in_stages": [stage_name],
                }
            elif stage_name not in prior["seen_in_stages"]:
                prior["seen_in_stages"].append(stage_name)
    return documents


def fact_groups(annotation: dict[str, Any]) -> list[list[list[str]]]:
    groups: list[list[list[str]]] = []
    for fact in annotation.get("answer_facts") or []:
        fact_support = [
            [str(item) for item in group]
            for group in fact.get("support_groups") or []
            if group
        ]
        if not fact_support:
            raise ValueError(f"fact has no support group: {annotation.get('id')}")
        groups.append(fact_support)
    if not groups:
        raise ValueError(f"annotation has no answer facts: {annotation.get('id')}")
    return groups


def proof_variant(groups: list[list[list[str]]], index: int) -> list[str]:
    return unique_in_order(
        corpus_id
        for fact_support in groups
        for corpus_id in fact_support[min(index, len(fact_support) - 1)]
    )


def complete_fact_indices(
    visible_ids: Iterable[str], groups: list[list[list[str]]]
) -> list[int]:
    visible = set(visible_ids)
    return [
        index
        for index, fact_support in enumerate(groups)
        if any(set(group) <= visible for group in fact_support)
    ]


def choose_missing_fact(groups: list[list[list[str]]]) -> int:
    # Prefer a fact with only one legal proof. This makes the broken state
    # genuinely incomplete even when another sub-fact has alternatives.
    for index, fact_support in enumerate(groups):
        if len(fact_support) == 1:
            return index
    return 0


def expected_answer(
    annotation: dict[str, Any],
    visible_ids: list[str],
    eid_map: dict[str, str],
) -> dict[str, Any]:
    visible = set(visible_ids)
    facts = []
    for fact in annotation["answer_facts"]:
        support = next(
            (group for group in fact["support_groups"] if set(group) <= visible),
            None,
        )
        if support is None:
            raise ValueError("expected answer requested for incomplete proof")
        facts.append(
            {
                "fact": fact["fact"],
                "evidence_ids": [eid_map[corpus_id] for corpus_id in support],
            }
        )
    return {"next_action": "answer_directly", "supported_facts": facts}


def question_only_hypothesis(question: str, query_type: str) -> dict[str, Any]:
    return {
        "question": question,
        "query_type": query_type or "reasoning",
        "entities": [],
        "keywords": [],
        "expected_answer_type": "short_text",
    }


def render_row(
    *,
    snapshot: dict[str, Any],
    annotation: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    condition: str,
    visible_ids: list[str],
    round_value: str,
    expected_action: str,
    missing_fact_indices: list[int],
    pair_group: str,
) -> dict[str, Any]:
    eid_map = {
        corpus_id: f"E{index}" for index, corpus_id in enumerate(visible_ids, start=1)
    }
    evidence_text = "\n".join(
        f"[{eid_map[corpus_id]}]\n{documents[corpus_id]['text']}"
        for corpus_id in visible_ids
    )
    prompt = render_exx_user_prompt(
        question=snapshot["question"],
        hypothesis=snapshot["hypothesis"],
        round_value=round_value,
        evidence_text=evidence_text,
    )
    if expected_action == "answer_directly":
        target = expected_answer(annotation, visible_ids, eid_map)
        reference_sufficiency = "sufficient"
    elif expected_action == "retrieve_more":
        target = {
            "next_action": "retrieve_more",
            "follow_up_hypothesis": question_only_hypothesis(
                snapshot["question"], str(snapshot.get("query_type") or "")
            ),
        }
        reference_sufficiency = "insufficient"
    elif expected_action == "abstain":
        target = {
            "next_action": "abstain",
            "reason": "当前可见证据不足以完整回答问题。",
        }
        reference_sufficiency = "insufficient"
    else:
        raise ValueError(f"unsupported action: {expected_action}")

    identity = canonical(
        {
            "protocol": PROTOCOL,
            "source_id": annotation["id"],
            "condition": condition,
            "round": round_value,
            "visible_ids": visible_ids,
        }
    )
    row_id = (
        f"{annotation['id']}::{condition}::"
        f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:10]}"
    )
    diagnostic = {
        "protocol": PROTOCOL,
        "source_id": annotation["id"],
        "question_family": annotation["question_family"],
        "short_name": CASE_METADATA[annotation["id"]]["short_name"],
        "proof_topology": CASE_METADATA[annotation["id"]]["proof_topology"],
        "condition": condition,
        "pair_group": pair_group,
        "round": round_value,
        "reference_sufficiency": reference_sufficiency,
        "expected_action": expected_action,
        "visible_corpus_ids": visible_ids,
        "evidence_id_to_corpus_id": {
            eid_map[corpus_id]: corpus_id for corpus_id in visible_ids
        },
        "complete_fact_indices": complete_fact_indices(
            visible_ids, fact_groups(annotation)
        ),
        "required_fact_count": len(annotation["answer_facts"]),
        "missing_fact_indices": missing_fact_indices,
        "answer_facts": annotation["answer_facts"],
        "evidence_chars": len(evidence_text),
        "evidence_truncated": False,
        "review_status": annotation.get("review_status"),
        "reviewer": annotation.get("reviewer"),
        "source_notes": annotation.get("notes"),
    }
    return {
        "id": row_id,
        "task_type": "grounded_action_generation",
        "system": EXX_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": canonical(target)},
        ],
        "diagnostic": diagnostic,
    }


def build_case(
    snapshot: dict[str, Any], annotation: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = collect_snapshot_documents(snapshot)
    groups = fact_groups(annotation)
    proof_a = proof_variant(groups, 0)
    proof_b = proof_variant(groups, 1)
    dual = unique_in_order([*proof_a, *proof_b])
    distractors = [str(item) for item in annotation.get("distractor_evidence_ids") or []]
    missing_fact = choose_missing_fact(groups)

    preserved_other_facts = unique_in_order(
        corpus_id
        for index, fact_support in enumerate(groups)
        if index != missing_fact
        for corpus_id in fact_support[0]
    )
    missing_fact_all_support = {
        corpus_id for group in groups[missing_fact] for corpus_id in group
    }
    broken = unique_in_order(
        [
            corpus_id
            for corpus_id in [*preserved_other_facts, *distractors]
            if corpus_id not in missing_fact_all_support
        ]
    )
    restoration_group = groups[missing_fact][
        min(1, len(groups[missing_fact]) - 1)
    ]
    restored = unique_in_order([*broken, *restoration_group])
    renumbered_b = unique_in_order([*distractors, *proof_b])

    required_ids = set(
        [
            *proof_a,
            *proof_b,
            *dual,
            *distractors,
            *broken,
            *restored,
            *renumbered_b,
        ]
    )
    missing_documents = required_ids - documents.keys()
    if missing_documents:
        raise ValueError(
            f"snapshot lacks required documents for {annotation['id']}: "
            f"{sorted(missing_documents)}"
        )

    states = [
        ("proof_a", proof_a, "1/2", "answer_directly"),
        ("proof_b", proof_b, "1/2", "answer_directly"),
        ("dual_proof", dual, "1/2", "answer_directly"),
        ("renumbered_b_with_distractor", renumbered_b, "1/2", "answer_directly"),
        ("broken_round1", broken, "1/2", "retrieve_more"),
        ("broken_round2", broken, "2/2", "abstain"),
        ("restored_round2", restored, "2/2", "answer_directly"),
    ]
    rows = []
    for condition, visible_ids, round_value, expected_action in states:
        complete = complete_fact_indices(visible_ids, groups)
        missing = sorted(set(range(len(groups))) - set(complete))
        if (expected_action == "answer_directly") != (not missing):
            raise ValueError(
                f"support graph/action mismatch {annotation['id']} {condition}: {missing}"
            )
        rows.append(
            render_row(
                snapshot=snapshot,
                annotation=annotation,
                documents=documents,
                condition=condition,
                visible_ids=visible_ids,
                round_value=round_value,
                expected_action=expected_action,
                missing_fact_indices=missing,
                pair_group=f"{annotation['id']}::alternative_proof_cycle",
            )
        )

    compact = {
        "source_id": annotation["id"],
        "question": snapshot["question"],
        "question_family": annotation["question_family"],
        "hypothesis": snapshot["hypothesis"],
        "answer_facts": annotation["answer_facts"],
        "minimal_sufficient_evidence_ids": annotation[
            "minimal_sufficient_evidence_ids"
        ],
        "distractor_evidence_ids": distractors,
        "proof_topology": CASE_METADATA[annotation["id"]]["proof_topology"],
        "documents": [
            documents[corpus_id] for corpus_id in sorted(required_ids)
        ],
        "source_notes": annotation.get("notes"),
    }
    return rows, compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--source-docx", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    annotations = {
        str(row["id"]): row for row in read_jsonl(args.annotations)
    }
    snapshots = {
        str(row["id"]): row for row in read_jsonl(args.snapshots)
    }
    selected_ids = [
        source_id
        for source_id in CASE_METADATA
        if source_id in annotations
        and any(
            len(fact.get("support_groups") or []) > 1
            for fact in annotations[source_id].get("answer_facts") or []
        )
    ]
    if selected_ids != list(CASE_METADATA):
        raise ValueError(
            f"expected the four reviewed alternative-proof cases, got {selected_ids}"
        )

    matrix: list[dict[str, Any]] = []
    compact_cases = []
    for source_id in selected_ids:
        if source_id not in snapshots:
            raise ValueError(f"snapshot missing source ID: {source_id}")
        case_rows, compact = build_case(
            snapshots[source_id], annotations[source_id]
        )
        matrix.extend(case_rows)
        compact_cases.append(compact)

    if len(matrix) != 28 or len({row["id"] for row in matrix}) != len(matrix):
        raise ValueError("pilot must contain 28 unique rows")
    if any(row["diagnostic"]["evidence_truncated"] for row in matrix):
        raise ValueError("pilot forbids evidence truncation")

    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_input.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.matrix.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    frozen = {
        "schema_version": PROTOCOL,
        "cases": compact_cases,
    }
    args.frozen_input.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": PROTOCOL,
        "source_files": {
            "annotations": {
                "path": str(args.annotations),
                "sha256": sha256_file(args.annotations),
            },
            "snapshots": {
                "path": str(args.snapshots),
                "sha256": sha256_file(args.snapshots),
            },
            "source_docx": {
                "path": str(args.source_docx),
                "sha256": sha256_file(args.source_docx),
            },
            "builder": {
                "path": str(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "outputs": {
            "matrix": {
                "path": str(args.matrix),
                "sha256": sha256_file(args.matrix),
                "rows": len(matrix),
            },
            "frozen_input": {
                "path": str(args.frozen_input),
                "sha256": sha256_file(args.frozen_input),
                "cases": len(compact_cases),
            },
        },
        "selection": {
            "cases": selected_ids,
            "case_count": len(selected_ids),
            "conditions_per_case": 7,
            "rows": len(matrix),
            "uses_only_reviewed_explicit_support_groups": True,
            "evidence_truncated": False,
            "training_performed": False,
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

