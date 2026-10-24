#!/usr/bin/env python3
"""Build split-aware Exx candidates from independently checked set judgements.

Deleting bad facts does not prove that the remaining answer is complete or
coherent. Changed answers are staged for a second set audit, never promoted
straight into SFT/KTO. Validation/test/unknown sources are diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import difflib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exx_set_grounding import (
    PROTOCOL, audited_input_sha256, input_key, validate_judgement,
)
from build_exx_eval_split import normalize_question, record_question, sha256_file
from glm_exx_semantic_reward import extract_judge_context, payload_is_judge_eligible
from evaluate_exx_outputs import generation_was_truncated
from audit_training_lineage import content_sha256


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"non-object row in {path}")
    return value


def parse_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(row.get("raw_output") or row.get("output") or "")
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prompt(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    return str(conversations[0].get("value") or "") if conversations else ""


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_record(
    source: dict[str, Any], output: dict[str, Any], *, tag: bool, suffix: str, source_kind: str,
    source_split: str = "unknown", training_eligible: bool = False,
) -> dict[str, Any]:
    digest = hashlib.sha256((str(source.get("id")) + suffix + canonical(output)).encode()).hexdigest()[:16]
    original_meta = source.get("meta") or {}
    original_meta = json.loads(original_meta) if isinstance(original_meta, str) else original_meta
    inherited = {
        key: info[key]
        for info in (original_meta, source.get("diagnostic") or {}, source)
        if isinstance(info, dict)
        for key in (
            "source_refs", "task_id", "question_key", "source_question_key",
            "event_family", "question_family", "story_families",
            "evidence_doc_ids", "gold_evidence_ids", "evidence_id_to_corpus_id",
        )
        if info.get(key)
    }
    return {
        "id": f"relevance_clean_{digest}_{suffix}",
        "task_type": source.get("task_type", "grounded_action_generation"),
        "system": source.get("system", ""),
        "tools": source.get("tools", ""),
        "kto_tag": tag,
        "conversations": [
            {"from": "human", "value": prompt(source)},
            {"from": "gpt", "value": canonical(output)},
        ],
        "meta": json.dumps(
            {
                **inherited,
                "source": source_kind,
                "source_id": source.get("id"),
                "source_question": record_question(source),
                "audited_source_sha256": content_sha256(source),
                "protocol": "audit_exx_set_grounding_v3",
                "source_split": source_split,
                "training_eligible": training_eligible,
                "requires_reaudit": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def cleaned_payload(payload: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any] | None:
    """Return only an unchanged, complete and jointly supported answer.

    A fact-level filter alone cannot certify coverage or cross-fact reference
    resolution after deletion. See ``propose_repair`` for non-training output.
    """
    if payload.get("next_action") != "answer_directly":
        return None
    try:
        validate_judgement(judgement, payload)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        judgement.get("set_support") != "complete"
        or judgement.get("context_sufficiency") != "sufficient"
        or judgement.get("action_appropriateness") != "appropriate"
    ):
        return None
    return payload


def propose_repair(payload: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any] | None:
    """Produce a candidate requiring a *new* set-level audit before training."""
    if payload.get("next_action") != "answer_directly":
        return None
    facts = payload.get("supported_facts")
    judged = judgement.get("facts")
    if not isinstance(facts, list) or not isinstance(judged, list) or len(facts) != len(judged):
        return None
    kept = [
        fact
        for fact, audit in zip(facts, judged)
        if audit.get("support") == "entailed"
        and audit.get("citation_complete") is True
        and audit.get("question_relevance") in {"direct", "supporting"}
    ]
    if not kept:
        return None
    return {"next_action": "answer_directly", "supported_facts": kept}


def source_ids(row: dict[str, Any]) -> set[str]:
    ids = {str(row["id"])} if row.get("id") else set()
    meta = row.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if isinstance(meta, dict) and meta.get("source_id"):
        ids.add(str(meta["source_id"]))
    return ids


def heldout_overlap(
    row: dict[str, Any], heldout: list[dict[str, Any]], threshold: float = 0.88
) -> bool:
    key = normalize_question(record_question(row))
    ids = source_ids(row)
    questions = {normalize_question(record_question(other)) for other in heldout}
    questions.discard("")
    return (
        any(ids & source_ids(other) for other in heldout)
        or key in questions
        or bool(key and difflib.get_close_matches(key, sorted(questions), n=1, cutoff=threshold))
    )


def audit_binding_issue(
    source: dict[str, Any], judged: dict[str, Any], summary: dict[str, Any],
) -> str | None:
    """Fail closed on stale/legacy verdicts, even when IDs and E-IDs match."""
    if generation_was_truncated(source):
        return "generation_truncated"
    if judged.get("finish_reason") != "stop" or judged.get("full_evidence") is not True:
        return "incomplete_audit_provenance"
    if judged.get("audited_input_sha256") != audited_input_sha256(source):
        return "audited_content_mismatch"
    if (
        summary.get("protocol") != PROTOCOL
        or not isinstance(summary.get("endpoint"), str) or not summary["endpoint"]
        or not isinstance(summary.get("judge_model"), str) or not summary["judge_model"]
        or type(summary.get("max_tokens")) is not int or summary["max_tokens"] < 1
    ):
        return "missing_judge_contract"
    expected = input_key(
        source, summary["endpoint"], summary["judge_model"], summary["max_tokens"]
    )
    if judged.get("input_key") != expected:
        return "judge_input_key_mismatch"
    return None


def source_training_issue(source: dict[str, Any]) -> str | None:
    """A CLI split flag cannot override explicit per-row evaluation provenance."""
    meta = source.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return "invalid_source_meta"
    if not isinstance(meta, dict):
        return "invalid_source_meta"
    for info in (source, meta, source.get("diagnostic") or {}, source.get("repair_provenance") or {}):
        if not isinstance(info, dict):
            return "invalid_source_provenance"
        if info.get("evaluation_only") is True or info.get("training_eligible") is False:
            return "evaluation_only_source"
        if info.get("requires_reaudit") is True:
            return "repair_requires_explicit_reaudit_promotion"
        if any(str(info.get(field) or "").lower() in {
            "val", "validation", "eval", "test", "heldout", "unknown",
        } for field in ("source_split", "split")):
            return "nontrain_source_split"
    return None


def lineage_issue(
    source: dict[str, Any], partitions: dict[str, dict[str, Any]],
    staged: dict[str, dict[str, Any]],
) -> str | None:
    partition = partitions.get(str(source.get("id")))
    if (
        not partition or partition.get("split") != "train"
        or partition.get("partition_candidate") is not True
        or partition.get("quarantine_reasons")
    ):
        return "lineage_partition_not_approved"
    source_sha = content_sha256(source)
    if source_sha == partition.get("row_sha256"):
        return None
    staged_row = staged.get(str(source.get("id")))
    provenance = source.get("candidate_provenance") or {}
    if (
        not staged_row
        or provenance.get("source_row_sha256") != partition.get("row_sha256")
        or provenance.get("component_id") != partition.get("component_id")
        or hashlib.sha256(str(source.get("raw_output") or "").encode()).hexdigest()
        != staged_row.get("candidate_output_sha256")
    ):
        return "lineage_source_content_mismatch"
    return None


def lineage_heldout_issue(lineage: dict[str, Any], heldout_paths: list[Path]) -> str | None:
    """The story-exposure audit must cover the *same* frozen heldout inputs."""
    inputs = lineage.get("inputs") or {}
    recorded = [
        item.get("sha256") for name, item in inputs.items()
        if name in {"val", "test", "development"} and isinstance(item, dict)
    ]
    if not recorded or any(not value for value in recorded):
        return "missing_lineage_heldout_provenance"
    if set(recorded) != {sha256_file(path) for path in heldout_paths}:
        return "lineage_heldout_content_mismatch"
    return None


def review_item_issue(source: dict[str, Any], reviews: dict[str, dict[str, Any]]) -> str | None:
    """Optional second review can veto teacher labels, never promote failures."""
    item = reviews.get(str(source.get("id")))
    if not item:
        return "missing_secondary_review"
    if item.get("audited_input_sha256") != audited_input_sha256(source):
        return "secondary_review_content_mismatch"
    if item.get("decision") != "accept":
        return "secondary_review_quarantine"
    return None


def load_bound_review(
    review_path: Path, predictions_path: Path, audit_path: Path,
) -> dict[str, dict[str, Any]]:
    """Validate a review against the frozen inputs it actually examined."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("protocol") != "asa_set_candidate_review_v1"
        or review.get("predictions_sha256") != sha256_file(predictions_path)
        or review.get("audit_sha256") != sha256_file(audit_path)
        or not review.get("reviewer")
        or not isinstance(review.get("results"), list)
    ):
        raise ValueError("secondary review does not match frozen predictions/audit")
    sources = {}
    for row in load_json(predictions_path):
        if not row.get("id") or str(row["id"]) in sources:
            raise ValueError("duplicate/missing review source ID")
        sources[str(row["id"])] = row
    reviews = {}
    for item in review["results"]:
        if (
            not isinstance(item, dict) or not item.get("id")
            or str(item["id"]) in reviews
            or item.get("decision") not in {"accept", "quarantine"}
            or not isinstance(item.get("reason"), str) or not item["reason"].strip()
        ):
            raise ValueError("invalid secondary review item")
        key = str(item["id"])
        if key not in sources or item.get("audited_input_sha256") != audited_input_sha256(sources[key]):
            raise ValueError("secondary_review_content_mismatch")
        reviews[key] = item
    return reviews


def prior_review_veto(
    source: dict[str, Any], prior_reviews: list[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Carry only prior rejections, including when a candidate has been edited.

    A new teacher vote or new text does not resolve a documented disagreement.
    Prior accepts confer no approval and never replace the new set audit.
    """
    vetoes = []
    for review in prior_reviews:
        for key in sorted(source_ids(source) & review.keys()):
            item = review[key]
            if item["decision"] == "quarantine":
                vetoes.append({
                    "reviewed_id": key,
                    "reason": item["reason"],
                    "prior_audited_input_sha256": item["audited_input_sha256"],
                    "current_audited_input_sha256": audited_input_sha256(source),
                    "same_audited_content": item["audited_input_sha256"] == audited_input_sha256(source),
                })
    return vetoes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-split", choices=("train", "validation", "test", "unknown"), default="unknown",
        help="Only train sources can produce training filenames; other splits are diagnostic-only.",
    )
    parser.add_argument(
        "--heldout-json", type=Path, action="append", default=[],
        help="Required for train sources. Excludes matching IDs, questions and near-duplicates.",
    )
    parser.add_argument("--lineage-audit", type=Path, help="Required train partition/lineage report.")
    parser.add_argument("--candidate-report", type=Path, help="Content bridge for staged, repaired candidate rows.")
    parser.add_argument("--secondary-review", type=Path, help="Content-bound optional review; veto only, never promotes a failed audit.")
    parser.add_argument(
        "--prior-secondary-review", type=Path, nargs=3, action="append", default=[],
        metavar=("REVIEW", "PREDICTIONS", "AUDIT"),
        help="Carry prior vetoes from their frozen source artifacts; prior accepts are ignored.",
    )
    args = parser.parse_args()
    if args.source_split == "train" and not args.heldout_json:
        parser.error("train sources require --heldout-json leakage checks")
    if args.source_split == "train" and not args.lineage_audit:
        parser.error("train sources require --lineage-audit partition checks")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    rows = load_json(args.predictions)
    heldout = [row for path in args.heldout_json for row in load_json(path)]
    if args.source_split == "train" and (
        not heldout or any(not record_question(row) for row in heldout)
    ):
        parser.error("heldout data must be non-empty and every row must have a question")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    reviews = {}
    try:
        if args.secondary_review:
            reviews = load_bound_review(args.secondary_review, args.predictions, args.audit)
        prior_reviews = [load_bound_review(*paths) for paths in args.prior_secondary_review]
    except ValueError as exc:
        parser.error(str(exc))
    partitions, staged = {}, {}
    if args.lineage_audit:
        lineage = json.loads(args.lineage_audit.read_text(encoding="utf-8"))
        if lineage.get("protocol") != "asa_training_lineage_audit_v1":
            parser.error("unrecognized lineage audit")
        if args.source_split == "train" and (issue := lineage_heldout_issue(lineage, args.heldout_json)):
            parser.error(issue)
        for row in lineage.get("records") or []:
            if row.get("split") != "train":
                continue
            if not row.get("id") or str(row["id"]) in partitions:
                parser.error("duplicate/missing train lineage ID")
            partitions[str(row["id"])] = row
    if args.candidate_report:
        staged_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
        if (
            not args.lineage_audit
            or staged_report.get("output_sha256") != sha256_file(args.predictions)
            or staged_report.get("inputs", {}).get("lineage", {}).get("sha256")
            != sha256_file(args.lineage_audit)
        ):
            parser.error("candidate report does not match predictions and lineage")
        for row in staged_report.get("records") or []:
            if not row.get("id") or str(row["id"]) in staged:
                parser.error("duplicate/missing staged record ID")
            staged[str(row["id"])] = row
    audit_rows = audit.get("results", [])
    results = {}
    for row in audit_rows:
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("audit rows must have non-empty IDs")
        key = str(row["id"])
        if key in results:
            raise ValueError(f"duplicate audit ID: {key}")
        results[key] = row
    if any(not row.get("id") for row in rows) or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("prediction IDs must be non-empty and unique")
    counts = Counter()
    sft: list[dict[str, Any]] = []
    kto: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    carried_vetoes: list[dict[str, Any]] = []
    for source in rows:
        if args.source_split == "train" and (source_issue := source_training_issue(source)):
            counts[f"excluded_{source_issue}"] += 1
            continue
        if args.source_split == "train" and (partition_issue := lineage_issue(source, partitions, staged)):
            counts[f"excluded_{partition_issue}"] += 1
            continue
        if not record_question(source):
            counts["excluded_missing_question"] += 1
            continue
        if heldout_overlap(source, heldout):
            counts["excluded_heldout_overlap"] += 1
            continue
        if vetoes := prior_review_veto(source, prior_reviews):
            counts["excluded_prior_secondary_review_quarantine"] += 1
            carried_vetoes.append({"id": source["id"], "vetoes": vetoes})
            continue
        original = parse_payload(source)
        judged = results.get(str(source.get("id")))
        judgement = judged.get("judgement") if judged else None
        if (
            not original or not judged or judged.get("status") != "ok"
            or not isinstance(judgement, dict)
        ):
            counts["missing_or_invalid_audit"] += 1
            continue
        try:
            if issue := audit_binding_issue(source, judged, audit.get("summary") or {}):
                counts[f"excluded_{issue}"] += 1
                continue
            validate_judgement(judgement, original)
            context = extract_judge_context(prompt(source))
            if not payload_is_judge_eligible(original, context["evidence"]):
                raise ValueError("invalid_prediction_protocol")
        except (KeyError, TypeError, ValueError):
            counts["excluded_protocol_or_binding_mismatch"] += 1
            continue
        if args.secondary_review and (issue := review_item_issue(source, reviews)):
            counts[f"excluded_{issue}"] += 1
            continue
        accepted = cleaned_payload(original, judgement)
        # Keep independently appropriate non-answer actions too. An answer-only
        # recipe teaches premature answers and hides the sufficiency boundary.
        if (
            original.get("next_action") in {"retrieve_more", "abstain"}
            and judgement.get("context_sufficiency") == "insufficient"
            and judgement.get("action_appropriateness") == "appropriate"
        ):
            accepted = original
        if accepted is not None:
            sft.append(
                make_record(
                    source, accepted, tag=True, suffix="chosen", source_kind="relevance_verified",
                    source_split=args.source_split, training_eligible=args.source_split == "train",
                )
            )
            counts["sft_chosen"] += 1
            counts[f"accepted_action:{original['next_action']}"] += 1
        else:
            candidate = propose_repair(original, judgement)
            if candidate is None:
                counts["excluded_no_safe_subset"] += 1
            else:
                repairs.append({
                    **source,
                    "raw_output": canonical(candidate),
                    "conversations": [
                        {"from": "human", "value": prompt(source)},
                        {"from": "gpt", "value": canonical(candidate)},
                    ],
                    "repair_provenance": {
                        "original_output": canonical(original),
                        "source_split": args.source_split,
                        "training_eligible": False,
                        "requires_reaudit": True,
                        "original_set_support": judgement.get("set_support"),
                        "audit_sha256": sha256_file(args.audit),
                    },
                })
                counts["staged_for_set_reaudit"] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.source_split == "train" else "diagnostic_"
    for name, rows_out in (
        (f"{prefix}sft.json", sft), (f"{prefix}kto.json", kto), ("repair_candidates.json", repairs)
    ):
        (args.out_dir / name).write_text(
            json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "predictions": str(args.predictions),
        "audit": str(args.audit),
        "counts": dict(sorted(counts.items())),
        "sft_rows": len(sft),
        "kto_rows": len(kto),
        "repair_candidates": len(repairs),
        "source_split": args.source_split,
        "training_eligible": args.source_split == "train",
        "training_recipe_ready": False,
        "independent_human_gold": False,
        "recipe_note": "Accepted samples do not certify a balanced Full/Distractor/Missing action recipe or independent evaluation.",
        "policy": "No subset promotion; edited answers require a fresh set audit. No inferred KTO pairs.",
        "audit_content_and_contract_verified": True,
        "explicit_evaluation_provenance_cannot_be_overridden": True,
        "lineage_audit": (
            {"path": str(args.lineage_audit), "sha256": sha256_file(args.lineage_audit)}
            if args.lineage_audit else None
        ),
        "candidate_report": (
            {"path": str(args.candidate_report), "sha256": sha256_file(args.candidate_report)}
            if args.candidate_report else None
        ),
        "secondary_review": (
            {"path": str(args.secondary_review), "sha256": sha256_file(args.secondary_review),
             "policy": "veto_only_no_promotion"}
            if args.secondary_review else None
        ),
        "prior_secondary_reviews": [
            {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in zip(("review", "predictions", "audit"), paths)
            }
            for paths in args.prior_secondary_review
        ],
        "carried_vetoes": carried_vetoes,
        "source_sha256": sha256_file(args.predictions),
        "audit_sha256": sha256_file(args.audit),
        "outputs": {
            name: {"path": str(args.out_dir / name), "sha256": sha256_file(args.out_dir / name)}
            for name in (f"{prefix}sft.json", f"{prefix}kto.json", "repair_candidates.json")
        },
        "heldout_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.heldout_json
        ],
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
