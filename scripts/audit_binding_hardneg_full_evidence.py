#!/usr/bin/env python3
"""Audit a full-evidence hard-negative rejudgement sidecar.

The report joins the quarantined historical candidates with their complete
evidence rejudgements, checks the four-part key, prefix relation, SHA256
fingerprint and character count, and summarizes old-to-new verdict changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(
    record_id: str, fact_index: int, claim: str, eid: str, text: str
) -> str:
    payload = {
        "record_id": record_id,
        "fact_index": fact_index,
        "claim": claim,
        "eid": eid,
        "text": text,
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def key(
    record_id: str, fact_index: int, eid: str, candidate_role: str
) -> tuple[str, int, str, str]:
    return (record_id, fact_index, eid, candidate_role)


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--rejudgements", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    hardneg_rows = read_jsonl(args.hardneg)
    quarantine_rows = read_jsonl(args.quarantine)
    rejudgement_rows = read_jsonl(args.rejudgements)

    candidates: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for source in hardneg_rows:
        record_id = str(source.get("record_id") or "")
        fact_index = as_int(source.get("fact_index", -1))
        claim = str(source.get("claim") or "")
        for role, items in (
            ("hard_negative", source.get("hard_negatives") or []),
            ("suspected_missed_positive", source.get("suspected_missed_positives") or []),
        ):
            for item in items:
                eid = str(item.get("eid") or "")
                candidates[key(record_id, fact_index, eid, role)] = {
                    "record_id": record_id,
                    "fact_index": fact_index,
                    "claim": claim,
                    "eid": eid,
                    "candidate_role": role,
                    "old_verdict": str(item.get("glm_verdict") or "missing"),
                    "old_text": str(item.get("text") or ""),
                }

    sidecar: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    duplicate_keys: set[tuple[str, int, str, str]] = set()
    for row in rejudgement_rows:
        row_key = key(
            str(row.get("record_id") or ""),
            as_int(row.get("fact_index", -1)),
            str(row.get("eid") or ""),
            str(row.get("candidate_role") or ""),
        )
        if row_key in sidecar:
            duplicate_keys.add(row_key)
        sidecar[row_key] = row

    validation_errors: list[str] = []
    old_verdicts = Counter()
    new_verdicts = Counter()
    roles = Counter()
    transitions = Counter()
    valid = 0
    for quarantine in quarantine_rows:
        row_key = key(
            str(quarantine.get("record_id") or ""),
            as_int(quarantine.get("fact_index", -1)),
            str(quarantine.get("eid") or ""),
            str(quarantine.get("candidate_role") or ""),
        )
        source = candidates.get(row_key)
        result = sidecar.get(row_key)
        if source is None:
            validation_errors.append(f"missing historical candidate: {row_key}")
            continue
        old_verdicts[source["old_verdict"]] += 1
        roles[source["candidate_role"]] += 1
        if result is None:
            validation_errors.append(f"missing rejudgement: {row_key}")
            continue
        if row_key in duplicate_keys:
            validation_errors.append(f"duplicate rejudgement key: {row_key}")
            continue
        text = str(result.get("text") or "")
        expected_sha = fingerprint(
            source["record_id"],
            source["fact_index"],
            source["claim"],
            source["eid"],
            text,
        )
        checks = [
            (result.get("status") == "ok", "status_not_ok"),
            (result.get("full_evidence") is True, "full_evidence_false"),
            (
                str(result.get("glm_verdict") or "") in {"supported", "unsupported"},
                "invalid_verdict",
            ),
            (text.startswith(source["old_text"]), "not_historical_prefix"),
            (str(result.get("evidence_sha256") or "") == expected_sha, "sha_mismatch"),
            (
                as_int(result.get("evidence_chars", -1)) == len(text),
                "char_count_mismatch",
            ),
        ]
        for passed, reason in checks:
            if not passed:
                validation_errors.append(f"{row_key}: {reason}")
        if all(passed for passed, _ in checks):
            valid += 1
        new_verdict = str(result.get("glm_verdict") or "missing")
        new_verdicts[new_verdict] += 1
        transitions[f"{source['old_verdict']} -> {new_verdict}"] += 1

    report = {
        "schema_version": "binding_hardneg_full_evidence_rejudge_v1",
        "source": {
            "hardneg": str(args.hardneg),
            "hardneg_sha256": sha256(args.hardneg),
            "quarantine": str(args.quarantine),
            "quarantine_sha256": sha256(args.quarantine),
            "rejudgements": str(args.rejudgements),
            "rejudgements_sha256": sha256(args.rejudgements),
        },
        "candidates": len(quarantine_rows),
        "completed": valid,
        "pending_after_run": len(quarantine_rows) - valid,
        "candidate_roles": dict(sorted(roles.items())),
        "old_verdicts": dict(sorted(old_verdicts.items())),
        "new_verdicts": dict(sorted(new_verdicts.items())),
        "transition_matrix": dict(sorted(transitions.items())),
        "duplicate_keys": len(duplicate_keys),
        "validation_errors": validation_errors,
        "training_gate": {
            "status": (
                "full_evidence_provenance_pass_manual_set_review_required"
                if valid == len(quarantine_rows) and not validation_errors
                else "blocked_pending_full_evidence_provenance"
            ),
            "policy": (
                "Only status=ok, full_evidence=true, four-key matched, historical-prefix "
                "matched, and SHA256-validated rows may replace old candidate text and "
                "verdicts. A supported hard-negative is excluded from the negative pool; "
                "a supported suspected-missed-positive still requires explicit opt-in."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
