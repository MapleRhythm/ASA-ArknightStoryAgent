#!/usr/bin/env python3
"""Build clean pointwise/pairwise binding-verifier data.

The source hard-negative file contains both audited and provisional candidates.
This script joins it with the GLM gold recalibration manifest and emits only:

* clean/partial gold bindings as positives (partial keeps only confirmed E-IDs);
* candidates explicitly judged ``unsupported`` by GLM as negatives.

By default, un-audited candidates and suspected missed positives are omitted,
preserving the original dataset.  ``--include-suspected-missed-positives`` adds
GLM-confirmed missed evidence as additional positive variants.  An optional
ambiguity sidecar can override the strict merged/individual decision for the
86 contradiction rows: union-supported labels are retained, while
partially-supported claims, ambiguous rows and explicitly unsupported rows
are excluded.  A ``supported_by_some`` claim must be atomized or rewritten
before it can become a positive; keeping an E-ID does not make the original
compound claim fully supported.

Gold judgements whose status failed, recorded input was truncated, or merged
and individual verdicts conflict are rejected by default.  An unsafe judgement
can only be restored by a successful full-evidence adjudication whose content
fingerprint matches the manifest.  ``--allow-unverified-gold-judgements`` is
available solely to reproduce historical datasets.

The output is grouped by source record before the deterministic train/eval
split so a question cannot leak across the two sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POSITIVE_GOLD_STATUSES = {
    "clean_support",
    "partial",
    "supported_by_union",
}
ADJUDICATION_LABELS = {
    "supported_by_union",
    "supported_by_some",
    "unsupported",
    "ambiguous",
}
LEGACY_HARDNEG_TEXT_CAP = 2000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_fingerprint(task: dict[str, Any]) -> str:
    eids = [str(item) for item in task.get("cited_eids") or []]
    texts = task.get("cited_texts") or {}
    payload = {
        "claim": str(task.get("claim") or ""),
        "cited_eids": eids,
        "cited_texts": {eid: str(texts.get(eid) or "") for eid in eids},
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def judgement_coverage_issue(
    task: dict[str, Any], judgement: dict[str, Any]
) -> str | None:
    try:
        limit = int(judgement.get("chunk_chars"))
    except (TypeError, ValueError):
        return "coverage_unknown"
    texts = task.get("cited_texts") or {}
    if any(
        len(str(texts.get(str(eid)) or "")) > limit
        for eid in judgement.get("cited_eids") or []
    ):
        return "judge_input_truncated"
    return None


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


def full_adjudication_matches(
    task: dict[str, Any], adjudication: dict[str, Any] | None
) -> bool:
    if not (
        adjudication
        and adjudication.get("status") == "ok"
        and adjudication.get("full_evidence") is True
        and adjudication.get("evidence_sha256") == evidence_fingerprint(task)
    ):
        return False
    label = str(adjudication.get("label") or "")
    if label not in ADJUDICATION_LABELS:
        return False
    cited = {str(item) for item in task.get("cited_eids") or []}
    kept = [str(item) for item in adjudication.get("keep_eids") or [] if str(item)]
    if any(eid not in cited for eid in kept):
        return False
    if label in {"supported_by_union", "supported_by_some"}:
        return bool(kept)
    return not kept


def negative_evidence_fingerprint(
    record_id: str,
    fact_index: int,
    claim: str,
    candidate: dict[str, Any],
) -> str:
    payload = {
        "record_id": record_id,
        "fact_index": fact_index,
        "claim": claim,
        "eid": str(candidate.get("eid") or ""),
        "text": str(candidate.get("text") or ""),
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def negative_candidate_key(
    record_id: str, fact_index: int, eid: str, candidate_role: str
) -> tuple[str, int, str, str]:
    return (record_id, fact_index, eid, candidate_role)


def full_negative_rejudgement_matches(
    record_id: str,
    fact_index: int,
    claim: str,
    original_candidate: dict[str, Any],
    candidate_role: str,
    rejudgement: dict[str, Any] | None,
) -> bool:
    """Validate a full-evidence hard-negative sidecar row.

    The sidecar is trusted only when it is an explicitly successful judgement,
    carries the expected role/key, contains complete text, fingerprints that
    exact text, and extends the historical candidate as a prefix.  This keeps
    stale, cross-role, or hand-edited rows quarantined.
    """
    try:
        sidecar_fact_index = int(rejudgement.get("fact_index", -1)) if rejudgement else -1
    except (TypeError, ValueError):
        sidecar_fact_index = -1
    if not (
        rejudgement
        and rejudgement.get("status") == "ok"
        and rejudgement.get("full_evidence") is True
        and str(rejudgement.get("record_id") or "") == record_id
        and sidecar_fact_index == fact_index
        and str(rejudgement.get("claim") or "") == claim
        and str(rejudgement.get("eid") or "") == str(original_candidate.get("eid") or "")
        and str(rejudgement.get("candidate_role") or "") == candidate_role
        and str(rejudgement.get("glm_verdict") or "")
        in {"supported", "unsupported"}
    ):
        return False
    text = str(rejudgement.get("text") or "")
    if not text:
        return False
    original_text = str(original_candidate.get("text") or "")
    if not text.startswith(original_text) or len(text) <= len(original_text):
        return False
    if (
        str(rejudgement.get("evidence_sha256") or "")
        != negative_evidence_fingerprint(
            record_id, fact_index, claim, original_candidate | {"text": text}
        )
    ):
        return False
    try:
        evidence_chars = int(rejudgement.get("evidence_chars"))
    except (TypeError, ValueError):
        return False
    return evidence_chars == len(text)


def negative_candidate_coverage_issue(
    record_id: str,
    fact_index: int,
    claim: str,
    candidate: dict[str, Any],
) -> str | None:
    text = str(candidate.get("text") or "")
    if (
        candidate.get("full_evidence") is True
        and candidate.get("evidence_sha256")
        == negative_evidence_fingerprint(record_id, fact_index, claim, candidate)
    ):
        return None
    if len(text) >= LEGACY_HARDNEG_TEXT_CAP:
        return "judge_input_truncated"
    return None


def classify(judgement: dict[str, Any]) -> tuple[str, list[str]]:
    merged = judgement.get("merged") or {}
    individual = judgement.get("individual") or {}
    eids = [str(item) for item in judgement.get("cited_eids") or []]
    merged_verdict = merged.get("verdict")
    supported = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "supported"]
    unsupported = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "unsupported"]
    uncertain = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") in {"skip", "error", None}
    ]
    if merged_verdict != "supported":
        return "unsupported", []
    if uncertain:
        kept = supported
        return ("partial" if kept else "unsupported"), kept
    if unsupported:
        return ("partial" if supported else "unsupported"), supported
    return "clean_support", eids


def load_adjudications(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row.get("fact_id")): row
        for row in read_jsonl(path)
        if str(row.get("fact_id") or "")
    }


def classify_with_adjudication(
    judgement: dict[str, Any],
    adjudication: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Apply the third-pass ambiguity adjudication when one exists.

    The strict first pass is deliberately kept as the default.  The sidecar is
    only consulted when explicitly supplied, and its ``keep_eids`` are trusted
    only for supported_by_union/supported_by_some labels.
    """
    if adjudication:
        label = str(adjudication.get("label") or "")
        keep_eids = [
            str(item)
            for item in adjudication.get("keep_eids") or []
            if str(item)
        ]
        if label in {"ambiguous", "unsupported"}:
            return label, []
        if label in {"supported_by_union", "supported_by_some"} and keep_eids:
            return label, keep_eids
    return classify(judgement)


def stable_eval(record_id: str, eval_ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < eval_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--full-evidence-rejudgements",
        type=Path,
        default=None,
        help="Optional hard-negative full-evidence rejudgement JSONL sidecar.",
    )
    parser.add_argument(
        "--ambiguity",
        type=Path,
        default=None,
        help="Optional third-pass adjudication JSONL keyed by fact_id.",
    )
    parser.add_argument(
        "--include-suspected-missed-positives",
        action="store_true",
        help="Add GLM-confirmed suspected-missed evidence as positive variants.",
    )
    parser.add_argument(
        "--allow-unverified-gold-judgements",
        action="store_true",
        help="Historical reproduction only: allow missing/truncated judge provenance.",
    )
    parser.add_argument(
        "--allow-truncated-negative-candidates",
        action="store_true",
        help="Historical reproduction only: allow legacy 2,000-character candidates.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(args.manifest)
    adjudications = load_adjudications(args.ambiguity)
    full_negative_rows = (
        read_jsonl(args.full_evidence_rejudgements)
        if args.full_evidence_rejudgements
        else []
    )
    full_negative_lookup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    duplicate_full_negative_keys: set[tuple[str, int, str, str]] = set()
    for row in full_negative_rows:
        key = negative_candidate_key(
            str(row.get("record_id") or ""),
            int(row.get("fact_index", -1)),
            str(row.get("eid") or ""),
            str(row.get("candidate_role") or ""),
        )
        if key in full_negative_lookup:
            duplicate_full_negative_keys.add(key)
        full_negative_lookup[key] = row
    task_lookup = {
        (str(row.get("row_id") or ""), int(row.get("fact_index", -1))): row
        for row in manifest
    }
    judgement_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(args.judgements):
        key = (str(row.get("row_id") or ""), int(row.get("fact_index", -1)))
        judgement_lookup[key] = row

    pairs: list[dict[str, Any]] = []
    stats = Counter()
    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in read_jsonl(args.hardneg):
        record_id = str(source.get("record_id") or "")
        key = (record_id, int(source.get("fact_index", -1)))
        task = task_lookup.get(key)
        judgement = judgement_lookup.get(key)
        if task is None or judgement is None:
            stats["missing_join"] += 1
            continue
        adjudication = adjudications.get(str(judgement.get("fact_id") or ""))
        coverage_issue = judgement_coverage_issue(task, judgement)
        conflict = contradiction_kind(judgement)
        status_issues = judgement_status_issues(judgement)
        verified_adjudication = full_adjudication_matches(task, adjudication)
        trusted_adjudication = (
            adjudication
            if verified_adjudication
            or (
                args.allow_unverified_gold_judgements
                and adjudication
                and adjudication.get("status") == "ok"
            )
            else None
        )
        if (
            (coverage_issue or conflict or status_issues)
            and not args.allow_unverified_gold_judgements
            and not verified_adjudication
        ):
            if coverage_issue:
                stats[f"dropped_gold:{coverage_issue}"] += 1
            if conflict:
                stats[f"dropped_gold:{conflict}"] += 1
            for issue in status_issues:
                stats[f"dropped_gold:{issue}"] += 1
            continue
        status, kept_eids = classify_with_adjudication(
            judgement, trusted_adjudication
        )
        stats[f"gold_status:{status}"] += 1
        if status not in POSITIVE_GOLD_STATUSES or not kept_eids:
            stats["dropped_nonpositive_gold"] += 1
            continue
        positive = "\n\n".join(
            f"[{eid}]\n{str((task.get('cited_texts') or {}).get(eid) or '').strip()}"
            for eid in kept_eids
            if str((task.get("cited_texts") or {}).get(eid) or "").strip()
        ).strip()
        if not positive:
            stats["dropped_empty_positive"] += 1
            continue
        claim = str(source.get("claim") or task.get("claim") or "").strip()
        confirmed: list[dict[str, Any]] = []
        candidate_items: dict[
            tuple[str, str], dict[str, Any] | None
        ] = {}
        for candidate_role, candidates in (
            ("hard_negative", source.get("hard_negatives") or []),
            (
                "suspected_missed_positive",
                source.get("suspected_missed_positives") or [],
            ),
        ):
            for original_item in candidates:
                eid = str(original_item.get("eid") or "")
                candidate_key = (candidate_role, eid)
                item: dict[str, Any] | None = original_item
                sidecar_key = negative_candidate_key(
                    record_id,
                    int(source.get("fact_index", -1)),
                    eid,
                    candidate_role,
                )
                sidecar = full_negative_lookup.get(sidecar_key)
                if sidecar is not None:
                    valid = (
                        sidecar_key not in duplicate_full_negative_keys
                        and full_negative_rejudgement_matches(
                            record_id,
                            int(source.get("fact_index", -1)),
                            claim,
                            original_item,
                            candidate_role,
                            sidecar,
                        )
                    )
                    if not valid:
                        stats["invalid_full_evidence_rejudgements"] += 1
                        stats[
                            (
                                "dropped_negative:"
                                if candidate_role == "hard_negative"
                                else "dropped_missed_positive:"
                            )
                            + "invalid_full_evidence_rejudgement"
                        ] += 1
                        item = None
                    else:
                        item = {
                            **original_item,
                            "text": str(sidecar.get("text") or ""),
                            "glm_verdict": str(sidecar.get("glm_verdict") or ""),
                            "glm_reason": str(sidecar.get("glm_reason") or ""),
                            "full_evidence": True,
                            "evidence_sha256": str(
                                sidecar.get("evidence_sha256") or ""
                            ),
                        }
                        stats["accepted_full_evidence_rejudgements"] += 1
                        stats[
                            f"full_evidence_rejudgement_verdict:{item['glm_verdict']}"
                        ] += 1
                elif candidate_role == "hard_negative":
                    stats["unrejudged_hard_negative_candidates"] += 1
                else:
                    stats["unrejudged_suspected_missed_positive_candidates"] += 1
                candidate_items[candidate_key] = item

                if candidate_role != "hard_negative" or item is None:
                    continue
                if (
                    str(item.get("glm_verdict") or "") != "unsupported"
                    or not str(item.get("text") or "").strip()
                ):
                    continue
                issue = negative_candidate_coverage_issue(
                    record_id, int(source.get("fact_index", -1)), claim, item
                )
                if issue and not args.allow_truncated_negative_candidates:
                    stats[f"dropped_negative:{issue}"] += 1
                    continue
                confirmed.append(item)
        stats["confirmed_candidates"] += len(confirmed)
        stats["provisional_candidates"] += sum(
            str(item.get("glm_verdict") or "") == "not_run"
            for item in source.get("hard_negatives") or []
        )
        stats["suspected_missed_positives"] += len(source.get("suspected_missed_positives") or [])
        positive_variants: list[dict[str, Any]] = [
            {
                "text": positive,
                "eids": kept_eids,
                "variant": "gold",
                "glm_reason": "",
            }
        ]
        if args.include_suspected_missed_positives:
            for original_item in source.get("suspected_missed_positives") or []:
                key = (
                    "suspected_missed_positive",
                    str(original_item.get("eid") or ""),
                )
                if key in candidate_items and candidate_items[key] is None:
                    continue
                item = candidate_items.get(key, original_item)
                if str(item.get("glm_verdict") or "") != "supported":
                    continue
                missed_text = str(item.get("text") or "").strip()
                missed_eid = str(item.get("eid") or "").strip()
                issue = negative_candidate_coverage_issue(
                    record_id, int(source.get("fact_index", -1)), claim, item
                )
                if issue and not args.allow_truncated_negative_candidates:
                    stats[f"dropped_missed_positive:{issue}"] += 1
                    continue
                if not missed_text or not missed_eid or missed_text == positive:
                    continue
                positive_variants.append(
                    {
                        "text": missed_text,
                        "eids": [missed_eid],
                        "variant": "suspected_missed_positive",
                        "glm_reason": str(item.get("glm_reason") or ""),
                    }
                )
                stats["included_suspected_missed_positives"] += 1
        for item in confirmed:
            negative = str(item.get("text") or "").strip()
            if not negative:
                stats["dropped_empty_or_equal_negative"] += 1
                continue
            for variant in positive_variants:
                variant_text = str(variant["text"])
                if negative == variant_text:
                    stats["dropped_empty_or_equal_negative"] += 1
                    continue
                pair = {
                    "query": claim,
                    "query_type": str(source.get("query_type") or task.get("query_type") or "unknown"),
                    "source_name": str(source.get("source_name") or record_id),
                    "positive": variant_text,
                    "negative": negative,
                    "positive_score": 1.0,
                    "negative_score": 0.0,
                    "negative_type": "binding_glm_confirmed_unsupported",
                    "positive_type": variant["variant"],
                    "positive_glm_reason": variant["glm_reason"],
                    "answer": "",
                    "answer_evidence": kept_eids,
                    "answer_focus": "",
                    "positive_chain": variant["eids"],
                    "negative_chain": [str(item.get("eid") or "")],
                    "record_id": record_id,
                    "fact_index": int(source.get("fact_index", -1)),
                    "gold_status": status,
                    "negative_glm_reason": str(item.get("glm_reason") or ""),
                    "negative_cos_sim": item.get("cos_sim"),
                }
                pairs.append(pair)
                group_records[record_id].append(pair)

    eval_ids = {
        record_id
        for record_id in group_records
        if stable_eval(record_id, args.eval_ratio, args.seed)
    }
    # Keep both partitions non-empty when data permits.
    if not eval_ids and len(group_records) > 1:
        eval_ids.add(sorted(group_records)[0])
    train = [pair for pair in pairs if pair["record_id"] not in eval_ids]
    eval_rows = [pair for pair in pairs if pair["record_id"] in eval_ids]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(eval_rows)

    for name, rows in (("pairwise.jsonl", pairs), ("train.jsonl", train), ("eval.jsonl", eval_rows)):
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(compact(row) + "\n")
    report = {
        "source": {
            "hardneg": str(args.hardneg),
            "manifest": str(args.manifest),
            "judgements": str(args.judgements),
            "ambiguity": str(args.ambiguity) if args.ambiguity else None,
            "full_evidence_rejudgements": (
                str(args.full_evidence_rejudgements)
                if args.full_evidence_rejudgements
                else None
            ),
        },
        "include_suspected_missed_positives": args.include_suspected_missed_positives,
        "allow_unverified_gold_judgements": args.allow_unverified_gold_judgements,
        "allow_truncated_negative_candidates": args.allow_truncated_negative_candidates,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "full_evidence_rejudgement_rows": len(full_negative_rows),
        "duplicate_full_evidence_rejudgement_keys": len(
            duplicate_full_negative_keys
        ),
        "pairs": len(pairs),
        "train_pairs": len(train),
        "eval_pairs": len(eval_rows),
        "groups": len(group_records),
        "train_groups": len({row["record_id"] for row in train}),
        "eval_groups": len({row["record_id"] for row in eval_rows}),
        "query_types": dict(Counter(str(row.get("query_type") or "unknown") for row in pairs)),
        "gold_status": dict(Counter(str(row.get("gold_status") or "unknown") for row in pairs)),
        "negative_type": dict(Counter(str(row.get("negative_type") or "unknown") for row in pairs)),
        "stats": dict(stats),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "binding_verifier_train": {"file_name": "train.jsonl"},
                "binding_verifier_eval": {"file_name": "eval.jsonl"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
