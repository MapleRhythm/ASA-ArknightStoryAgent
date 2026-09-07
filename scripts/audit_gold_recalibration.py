#!/usr/bin/env python3
"""Audit GLM gold-binding recalibration inputs without changing source data.

The original recalibration records the number of characters shown to the
judge in ``chunk_chars``.  This auditor joins the full fact manifest with the
judgements and quarantines a fact when:

* any cited evidence text is longer than the text shown to the judge;
* the merged verdict conflicts with the per-E-ID verdicts; or
* judgement coverage cannot be established from the recorded provenance.

The report deliberately keeps the historical strict labels separate from the
conservative training-usable subset.  A quarantined label must be rejudged
against full evidence before it can remove a fact, flip an action, or become a
hard training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("clean_support", "partial", "unsupported")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_no}")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_fingerprint(fact: dict[str, Any]) -> str:
    eids = [str(item) for item in fact.get("cited_eids") or []]
    texts = fact.get("cited_texts") or {}
    payload = {
        "claim": str(fact.get("claim") or ""),
        "cited_eids": eids,
        "cited_texts": {eid: str(texts.get(eid) or "") for eid in eids},
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def classify_judgement(judgement: dict[str, Any]) -> tuple[str, list[str]]:
    merged = judgement.get("merged") or {}
    individual = judgement.get("individual") or {}
    eids = [str(item) for item in judgement.get("cited_eids") or []]
    supported = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") == "supported"
    ]
    unsupported = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") == "unsupported"
    ]
    uncertain = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") in {"skip", "error", None}
    ]
    if merged.get("verdict") != "supported":
        return "unsupported", []
    if uncertain:
        return ("partial" if supported else "unsupported"), supported
    if unsupported:
        return ("partial" if supported else "unsupported"), supported
    return "clean_support", eids


def contradiction_kind(judgement: dict[str, Any]) -> str | None:
    merged = (judgement.get("merged") or {}).get("verdict")
    individual = judgement.get("individual") or {}
    eids = [str(item) for item in judgement.get("cited_eids") or []]
    supported = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") == "supported"
    ]
    if merged == "unsupported" and supported:
        return "merged_unsupported_ind_supported"
    if merged == "supported" and not supported:
        return "merged_supported_no_ind_supported"
    return None


def judgement_status_issues(judgement: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if (judgement.get("merged") or {}).get("status") != "ok":
        issues.append("merged_status_not_ok")
    individual = judgement.get("individual") or {}
    eids = [str(item) for item in judgement.get("cited_eids") or []]
    if any(
        not isinstance(individual.get(eid), dict)
        or (individual.get(eid) or {}).get("status") != "ok"
        for eid in eids
    ):
        issues.append("individual_status_not_ok")
    return issues


def coverage(
    fact: dict[str, Any], judgement: dict[str, Any]
) -> tuple[bool, int | None, dict[str, int], list[str]]:
    raw_limit = judgement.get("chunk_chars")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = None
    if limit is not None and limit <= 0:
        limit = None
    eids = [str(item) for item in fact.get("cited_eids") or []]
    texts = fact.get("cited_texts") or {}
    lengths = {eid: len(str(texts.get(eid) or "")) for eid in eids}
    truncated = [eid for eid, length in lengths.items() if limit is not None and length > limit]
    known_full = limit is not None and not truncated
    return known_full, limit, lengths, truncated


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if not value or count != 1)


def build_report(
    facts: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
    changes: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    changes = changes or []
    errors: list[str] = []
    fact_ids = [str(row.get("fact_id") or "") for row in facts]
    judgement_ids = [str(row.get("fact_id") or "") for row in judgements]
    bad_fact_ids = duplicate_values(fact_ids)
    bad_judgement_ids = duplicate_values(judgement_ids)
    if bad_fact_ids:
        errors.append(f"fact manifest has {len(bad_fact_ids)} empty or duplicate IDs")
    if bad_judgement_ids:
        errors.append(f"judgements have {len(bad_judgement_ids)} empty or duplicate IDs")

    fact_by_id = {
        str(row.get("fact_id")): row for row in facts if str(row.get("fact_id") or "")
    }
    judgement_by_id = {
        str(row.get("fact_id")): row
        for row in judgements
        if str(row.get("fact_id") or "")
    }
    missing = sorted(set(fact_by_id) - set(judgement_by_id))
    extra = sorted(set(judgement_by_id) - set(fact_by_id))
    if missing:
        errors.append(f"missing judgements: {len(missing)}")
    if extra:
        errors.append(f"extra judgements: {len(extra)}")

    labels = Counter()
    labels_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    truncated_by_label = Counter()
    conflict_by_kind = Counter()
    conservative_by_label = Counter()
    judge_contract = Counter()
    merged_status = Counter()
    individual_status = Counter()
    query_exposure: dict[str, Counter[str]] = defaultdict(Counter)
    quarantine: list[dict[str, Any]] = []
    unsafe_ids: set[str] = set()
    facts_by_row: dict[tuple[str, str], list[str]] = defaultdict(list)

    for fact_id in sorted(set(fact_by_id) & set(judgement_by_id)):
        fact = fact_by_id[fact_id]
        judgement = judgement_by_id[fact_id]
        for field in ("split", "row_id", "fact_index", "claim", "cited_eids"):
            if fact.get(field) != judgement.get(field):
                errors.append(f"{fact_id}: mismatched {field}")
        label, kept_eids = classify_judgement(judgement)
        labels[label] += 1
        split = str(fact.get("split") or "unknown")
        query_type = str(fact.get("query_type") or "unknown")
        labels_by_split[split][label] += 1
        facts_by_row[(split, str(fact.get("row_id") or ""))].append(fact_id)
        query_exposure[query_type]["facts"] += 1

        known_full, limit, lengths, truncated_eids = coverage(fact, judgement)
        conflict = contradiction_kind(judgement)
        reasons = judgement_status_issues(judgement)
        if limit is None:
            reasons.append("coverage_unknown")
        elif truncated_eids:
            reasons.append("judge_input_truncated")
            truncated_by_label[label] += 1
            query_exposure[query_type]["truncated"] += 1
        if conflict:
            reasons.append(conflict)
            conflict_by_kind[conflict] += 1
        if reasons:
            unsafe_ids.add(fact_id)
            query_exposure[query_type]["quarantined"] += 1
            quarantine.append(
                {
                    "fact_id": fact_id,
                    "split": split,
                    "row_id": fact.get("row_id"),
                    "question_key": fact.get("question_key"),
                    "fact_index": fact.get("fact_index"),
                    "query_type": query_type,
                    "claim": fact.get("claim"),
                    "strict_label": label,
                    "strict_kept_eids": kept_eids,
                    "reasons": reasons,
                    "chunk_chars": limit,
                    "evidence_chars": lengths,
                    "truncated_eids": truncated_eids,
                    "evidence_sha256": evidence_fingerprint(fact),
                }
            )
        elif known_full:
            conservative_by_label[label] += 1

        judge_contract[
            (
                str(judgement.get("judge_model") or "missing"),
                str(judgement.get("reasoning_effort") or "missing"),
                str(judgement.get("chunk_chars") or "missing"),
            )
        ] += 1
        merged = judgement.get("merged") or {}
        merged_status[
            (str(merged.get("status") or "missing"), str(merged.get("verdict") or "missing"))
        ] += 1
        for item in (judgement.get("individual") or {}).values():
            if isinstance(item, dict):
                individual_status[
                    (
                        str(item.get("status") or "missing"),
                        str(item.get("verdict") or "missing"),
                    )
                ] += 1

    fact_changes = [row for row in changes if row.get("entity_type") == "fact"]
    row_changes = [row for row in changes if row.get("entity_type") == "row"]
    expected_changed = {
        fact_id
        for fact_id, judgement in judgement_by_id.items()
        if classify_judgement(judgement)[0] != "clean_support"
    }
    actual_changed = {
        str(row.get("fact_id") or "") for row in fact_changes if row.get("fact_id")
    }
    action_flip_safety = Counter()
    for row in row_changes:
        if row.get("change_type") != "action_flip":
            continue
        action_flip_safety["total"] += 1
        key = (str(row.get("split") or ""), str(row.get("row_id") or ""))
        row_fact_ids = facts_by_row.get(key, [])
        if any(fact_id in unsafe_ids for fact_id in row_fact_ids):
            action_flip_safety["depends_on_quarantined_fact"] += 1
        else:
            action_flip_safety["conservative_usable"] += 1

    report = {
        "schema_version": "gold_recalibration_safety_v1",
        "facts": len(facts),
        "judgements": len(judgements),
        "fact_id_unique": len(set(fact_ids)),
        "judgement_id_unique": len(set(judgement_ids)),
        "missing_judgements": len(missing),
        "extra_judgements": len(extra),
        "strict_labels": {label: labels.get(label, 0) for label in LABELS},
        "strict_labels_by_split": {
            split: {label: counts.get(label, 0) for label in LABELS}
            for split, counts in sorted(labels_by_split.items())
        },
        "judge_contracts": [
            {
                "judge_model": model,
                "reasoning_effort": effort,
                "chunk_chars": chunk_chars,
                "count": count,
            }
            for (model, effort, chunk_chars), count in sorted(judge_contract.items())
        ],
        "merged_status": [
            {"status": status, "verdict": verdict, "count": count}
            for (status, verdict), count in sorted(merged_status.items())
        ],
        "individual_status": [
            {"status": status, "verdict": verdict, "count": count}
            for (status, verdict), count in sorted(individual_status.items())
        ],
        "truncated_by_strict_label": {
            label: truncated_by_label.get(label, 0) for label in LABELS
        },
        "contradictions": dict(sorted(conflict_by_kind.items())),
        "quarantined_facts": len(quarantine),
        "conservative_usable_facts": sum(conservative_by_label.values()),
        "conservative_usable_by_label": {
            label: conservative_by_label.get(label, 0) for label in LABELS
        },
        "query_type_exposure": {
            query_type: dict(sorted(counts.items()))
            for query_type, counts in sorted(query_exposure.items())
        },
        "changes": {
            "rows": len(changes),
            "fact_changes": len(fact_changes),
            "action_flips": sum(
                row.get("change_type") == "action_flip" for row in row_changes
            ),
            "expected_changed_fact_ids": len(expected_changed),
            "actual_changed_fact_ids": len(actual_changed),
            "missing_changed_fact_ids": len(expected_changed - actual_changed),
            "extra_changed_fact_ids": len(actual_changed - expected_changed),
            "action_flip_safety": dict(action_flip_safety),
        },
        "training_gate": {
            "status": (
                "blocked_pending_full_evidence_rejudge"
                if quarantine
                else "full_evidence_provenance_pass"
            ),
            "policy": (
                "Quarantined facts cannot remove gold, flip actions, or become hard "
                "training targets until a full-evidence adjudication matches "
                "evidence_sha256."
            ),
        },
        "validation_errors": errors,
    }
    return report, quarantine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    args = parser.parse_args()

    facts = read_jsonl(args.facts)
    judgements = read_jsonl(args.judgements)
    changes = read_jsonl(args.changes) if args.changes else []
    report, quarantine = build_report(facts, judgements, changes)
    report["source"] = {
        "facts": str(args.facts),
        "facts_sha256": sha256(args.facts),
        "judgements": str(args.judgements),
        "judgements_sha256": sha256(args.judgements),
        "changes": str(args.changes) if args.changes else None,
        "changes_sha256": sha256(args.changes) if args.changes else None,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    with args.quarantine.open("w", encoding="utf-8") as handle:
        for row in quarantine:
            handle.write(compact(row) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
