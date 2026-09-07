#!/usr/bin/env python3
"""Audit binding hard-negative labels against full prompt evidence.

The historical v2 builder selected full chunks but then sliced every chosen
candidate to 2,000 characters before GLM judging and serialization.  This
auditor reconstructs each E-ID from the immutable source prompt, verifies
that the stored text is an exact prefix, and quarantines labels whose source
chunk continued beyond the judged prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EVIDENCE_HEADER_RE = re.compile(r"(?m)^\[(E\d+)\]\s*(?:\n|$)")
STOP_LINE_RE = re.compile(r"(?m)^(?:output_schema|action_set|rules):")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{path}: expected a JSON array of objects")
        return value
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
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


def parse_evidence(prompt: str) -> dict[str, str]:
    marker = re.search(r"(?m)^evidence:\s*(?:\n|$)", prompt)
    if not marker:
        return {}
    body = prompt[marker.end() :]
    stop = STOP_LINE_RE.search(body)
    if stop:
        body = body[: stop.start()]
    matches = list(EVIDENCE_HEADER_RE.finditer(body))
    evidence: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            evidence[match.group(1)] = text
    return evidence


def source_evidence(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        conversations = row.get("conversations") or []
        if not row_id or not conversations:
            continue
        prompt = str((conversations[0] or {}).get("value") or "")
        result[row_id] = parse_evidence(prompt)
    return result


def evidence_fingerprint(
    record_id: str,
    fact_index: int,
    claim: str,
    eid: str,
    text: str,
) -> str:
    payload = {
        "record_id": record_id,
        "fact_index": fact_index,
        "claim": claim,
        "eid": eid,
        "text": text,
    }
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def candidate_rows(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        *[("hard_negative", item) for item in row.get("hard_negatives") or []],
        *[
            ("suspected_missed_positive", item)
            for item in row.get("suspected_missed_positives") or []
        ],
    ]


def build_report(
    hardneg_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence_by_record = source_evidence(source_rows)
    errors: list[str] = []
    counts = Counter()
    verdicts = Counter()
    truncated_verdicts = Counter()
    conservative_verdicts = Counter()
    role_counts = Counter()
    quarantine: list[dict[str, Any]] = []
    fact_keys: set[tuple[str, int]] = set()

    for row in hardneg_rows:
        record_id = str(row.get("record_id") or "")
        fact_index = int(row.get("fact_index", -1))
        fact_key = (record_id, fact_index)
        if fact_key in fact_keys:
            errors.append(f"duplicate fact key: {record_id}#{fact_index}")
        fact_keys.add(fact_key)
        claim = str(row.get("claim") or "")
        visible = evidence_by_record.get(record_id)
        if visible is None:
            errors.append(f"missing source row: {record_id}")
            continue
        for role, candidate in candidate_rows(row):
            counts["candidates"] += 1
            role_counts[role] += 1
            eid = str(candidate.get("eid") or "")
            verdict = str(candidate.get("glm_verdict") or "missing")
            verdicts[verdict] += 1
            full_text = visible.get(eid)
            if full_text is None:
                errors.append(f"{record_id}#{fact_index}: missing source E-ID {eid}")
                continue
            shown_text = str(candidate.get("text") or "")
            if shown_text != full_text[: len(shown_text)]:
                errors.append(
                    f"{record_id}#{fact_index}/{eid}: stored text is not source prefix"
                )
                continue
            truncated = len(full_text) > len(shown_text)
            if truncated:
                counts["truncated_candidates"] += 1
                truncated_verdicts[verdict] += 1
                if verdict in {"supported", "unsupported"}:
                    counts["judged_truncated_candidates"] += 1
                quarantine.append(
                    {
                        "record_id": record_id,
                        "question_key": row.get("question_key"),
                        "fact_index": fact_index,
                        "query_type": row.get("query_type"),
                        "claim": claim,
                        "eid": eid,
                        "candidate_role": role,
                        "glm_verdict": verdict,
                        "shown_chars": len(shown_text),
                        "full_chars": len(full_text),
                        "evidence_sha256": evidence_fingerprint(
                            record_id, fact_index, claim, eid, full_text
                        ),
                    }
                )
            else:
                conservative_verdicts[verdict] += 1

    report = {
        "schema_version": "binding_hardneg_safety_v1",
        "rows": len(hardneg_rows),
        "unique_fact_keys": len(fact_keys),
        "source_rows": len(source_rows),
        "source_records_with_evidence": len(evidence_by_record),
        "candidates": counts["candidates"],
        "candidate_roles": dict(sorted(role_counts.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "truncated_candidates": counts["truncated_candidates"],
        "judged_truncated_candidates": counts["judged_truncated_candidates"],
        "truncated_by_verdict": dict(sorted(truncated_verdicts.items())),
        "conservative_full_input_by_verdict": dict(
            sorted(conservative_verdicts.items())
        ),
        "quarantined_candidates": len(quarantine),
        "training_gate": {
            "status": (
                "blocked_pending_full_evidence_rejudge"
                if quarantine
                else "full_evidence_provenance_pass"
            ),
            "policy": (
                "A truncated candidate cannot be a confirmed negative or alternate "
                "positive until it is rejudged against full evidence and the "
                "evidence_sha256 matches."
            ),
        },
        "validation_errors": errors,
    }
    return report, quarantine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    args = parser.parse_args()

    hardneg_rows = read_records(args.hardneg)
    source_rows = read_records(args.source_rows)
    report, quarantine = build_report(hardneg_rows, source_rows)
    report["source"] = {
        "hardneg": str(args.hardneg),
        "hardneg_sha256": sha256(args.hardneg),
        "source_rows": str(args.source_rows),
        "source_rows_sha256": sha256(args.source_rows),
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
