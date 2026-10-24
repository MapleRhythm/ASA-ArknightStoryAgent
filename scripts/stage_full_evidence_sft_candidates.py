#!/usr/bin/env python3
"""Rebuild candidate answers from safe fact bindings, requiring fresh set review.

Only content-matching source rows that pass the conservative lineage audit are
staged. Fully supported facts may be restored from full-evidence sidecars; true
partial/ambiguous/unsupported facts are excluded, never silently atomized.
Dropping facts or fixing bindings does not certify completeness or an action.
Outputs intentionally contain no training registry and are not training-ready.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_binding_hardneg import parse_evidence
from audit_training_lineage import content_sha256
from build_binding_verifier_pairs import (
    classify_with_adjudication, full_adjudication_matches, judgement_coverage_issue,
    judgement_status_issues, contradiction_kind,
)
from build_exx_eval_split import record_question
from build_recalibrated_retrieval_manifest import read_json_rows
from rejudge_gold_claim_scope import (
    PROTOCOL as SCOPE_PROTOCOL, reusable as reusable_scope,
)
from evaluate_exx_outputs import validate_payload


def unique(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row.get(field)
        if not key or key in result:
            raise ValueError(f"duplicate_or_missing_{field}")
        result[key] = row
    return result


def resolved_status(fact, judgement, previous, scope):
    if any(row and row.get("training_eligible") is False for row in (previous, scope)):
        return "review_quarantine", [], "secondary_review_veto"
    if not scope and previous and previous.get("protocol") == SCOPE_PROTOCOL:
        scope = previous
    if scope:
        if not reusable_scope(fact, scope):
            raise ValueError("invalid_scope_sidecar")
        return scope["label"], scope["keep_eids"], "full_evidence_scope_v2"
    if previous:
        if (
            not full_adjudication_matches(fact, previous)
            or any(previous.get(k) != fact.get(k) for k in ("fact_id", "row_id", "fact_index"))
        ):
            raise ValueError("invalid_full_evidence_sidecar")
        return previous["label"], previous["keep_eids"], "full_evidence_legacy"
    if not judgement:
        return "missing_judgement", [], "unverified"
    if any(judgement.get(k) != fact.get(k) for k in ("fact_id", "row_id", "fact_index", "claim", "cited_eids")):
        raise ValueError("fact_judgement_mismatch")
    if (
        judgement_status_issues(judgement) or judgement_coverage_issue(fact, judgement)
        or contradiction_kind(judgement)
    ):
        return "unsafe_legacy_judgement", [], "unverified"
    status, kept = classify_with_adjudication(judgement, None)
    return status, kept, "legacy_untruncated"


def stage(sources, facts, judgements, prior, scope, lineage):
    if lineage.get("protocol") != "asa_training_lineage_audit_v1":
        raise ValueError("unrecognized_lineage_protocol")
    unique(sources, "id")
    facts_by_id = unique(facts, "fact_id")
    judges = unique(judgements, "fact_id")
    previous = unique(prior, "fact_id")
    scopes = unique(scope, "fact_id")
    for name, mapping in (("prior", previous), ("scope", scopes)):
        if set(mapping) - set(facts_by_id):
            raise ValueError(f"unresolved_{name}_fact")
    partitions = unique(
        [row for row in lineage["records"] if row["split"] == "train"], "id"
    )
    by_record = defaultdict(list)
    for fact in facts:
        by_record[fact["row_id"]].append(fact)
    counts, rows, audit = Counter(), [], []
    for source in sources:
        if source.get("task_type") != "grounded_action_generation":
            counts["excluded_other_task"] += 1
            continue
        partition = partitions.get(source["id"])
        if (
            not partition or partition.get("partition_candidate") is not True
            or partition.get("quarantine_reasons")
        ):
            counts["excluded_partition"] += 1
            continue
        if partition.get("row_sha256") != content_sha256(source):
            raise ValueError("lineage_source_content_mismatch")
        original = by_record.get(source["id"], [])
        if not original:
            counts["no_fact_manifest"] += 1
            continue
        indexes = [fact["fact_index"] for fact in original]
        if len(set(indexes)) != len(indexes):
            raise ValueError("duplicate_fact_index")
        conversations = source.get("conversations") or []
        if len(conversations) != 2 or conversations[0].get("from") != "human":
            raise ValueError("requires_single_turn_source")
        prompt = conversations[0]["value"]
        visible = parse_evidence(prompt)
        decisions, kept = [], []
        for fact in sorted(original, key=lambda f: f["fact_index"]):
            if fact.get("query") != record_question(source):
                raise ValueError("fact_question_mismatch")
            if any(fact["cited_texts"].get(eid) != visible.get(eid) for eid in fact["cited_eids"]):
                raise ValueError("fact_visible_evidence_mismatch")
            status, eids, provenance = resolved_status(
                fact, judges.get(fact["fact_id"]),
                previous.get(fact["fact_id"]), scopes.get(fact["fact_id"]),
            )
            accepted = status in {"supported_by_union", "clean_support", "partial"} and 1 <= len(eids) <= 2
            candidate = {"fact": fact["claim"], "evidence_ids": eids}
            if accepted and candidate not in kept:
                kept.append(candidate)
            counts[f"fact_status:{status}"] += 1
            decisions.append({
                "fact_id": fact["fact_id"], "status": status, "kept": accepted,
                "provenance": provenance,
            })
        if not kept:
            counts["no_fully_supported_candidate_facts"] += 1
            continue
        payload = {"next_action": "answer_directly", "supported_facts": kept}
        problems = validate_payload(payload, set(visible))
        if problems:
            counts["candidate_protocol_rejected"] += 1
            continue
        out = copy.deepcopy(source)
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        out["raw_output"] = text
        out["conversations"][1] = {"from": "gpt", "value": text}
        out["candidate_provenance"] = {
            "source_row_sha256": content_sha256(source), "component_id": partition["component_id"],
            "source_split": "train", "partition_checked": True, "set_audit_pending": True,
            "fully_supported_facts_are_not_complete_answers": True,
        }
        rows.append(out)
        audit.append({"id": source["id"], "decisions": decisions,
                      "candidate_output_sha256": hashlib.sha256(text.encode()).hexdigest()})
        counts["staged_candidates"] += 1
    return rows, {"counts": dict(counts), "records": audit,
                  "training_ready": False, "requires_fresh_set_audit": True,
                  "no_nonanswer_action_inferred_from_fact_deletion": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "facts", "judgements", "prior", "lineage", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scope", type=Path)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        parser.error("refusing to overwrite staged candidates")
    rows, report = stage(
        read_json_rows(args.sources), read_json_rows(args.facts), read_json_rows(args.judgements),
        read_json_rows(args.prior), read_json_rows(args.scope) if args.scope else [],
        json.loads(args.lineage.read_text(encoding="utf-8")),
    )
    report["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name in ("sources", "facts", "judgements", "prior", "lineage", "scope")
        if (path := getattr(args, name)) is not None
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), **report["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
