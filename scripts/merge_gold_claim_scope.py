#!/usr/bin/env python3
"""Create a verified, immutable legacy-compatible view of claim-scope verdicts.

The original full-evidence sidecar remains unchanged. Every legacy `some`
record must have a completed v2 scope verdict bound to its actual judge input.
This view is usable as --ambiguity for the pair builder, but does not certify
semantic gold, answer completeness, or an externally held-out training split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rejudge_gold_claim_scope import candidates, compact, read_jsonl, reusable, unique_rows


def merge(facts, prior, scope, quarantine=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = {row["fact_id"]: row for row in candidates(facts, prior)}
    replacements = unique_rows(scope)
    if set(replacements) != set(selected):
        raise ValueError("scope_must_cover_exactly_all_legacy_some_records")
    for fact_id, row in replacements.items():
        if not reusable(selected[fact_id], row):
            raise ValueError(f"invalid_scope_provenance:{fact_id}")
    vetoes = {}
    if quarantine is not None:
        if (
            quarantine.get("protocol") != "asa_claim_scope_review_quarantine_v1"
            or not quarantine.get("reviewer")
        ):
            raise ValueError("invalid_scope_quarantine_protocol")
        vetoes = unique_rows(quarantine.get("results") or [])
        for fact_id, item in vetoes.items():
            row = replacements.get(fact_id)
            if (
                not row or not isinstance(item.get("reason"), str) or not item["reason"].strip()
                or any(item.get(k) != row.get(k) for k in ("input_key", "evidence_sha256"))
            ):
                raise ValueError("scope_quarantine_content_mismatch")
    result = []
    for old in prior:
        row = replacements.get(old["fact_id"], old)
        if row["fact_id"] in vetoes:
            row = row | {
                "training_eligible": False,
                "scope_review_quarantine": {
                    "reviewer": quarantine["reviewer"],
                    "reason": vetoes[row["fact_id"]]["reason"],
                    "original_teacher_verdict_preserved": True,
                },
            }
        result.append(row)
    report = {
        "protocol": "asa_verified_claim_scope_projection_v1",
        "records": len(result),
        "scope_replacements": len(replacements),
        "before_labels": dict(Counter(row["label"] for row in prior)),
        "after_labels": dict(Counter(row["label"] for row in result)),
        "claim_support": dict(Counter(row["claim_support"] for row in scope)),
        "scope_review_quarantined": len(vetoes),
        "single_evidence_full_is_not_partial": True,
        "independent_human_gold": False,
        "training_ready": False,
        "original_sidecar_modified": False,
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("facts", "prior", "scope", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, help="Optional content-bound secondary-review vetoes; not affirmative gold.")
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        parser.error("refusing to overwrite adjudication view")
    for path in (args.facts, args.prior, args.scope):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    quarantine = json.loads(args.quarantine.read_text(encoding="utf-8")) if args.quarantine else None
    if quarantine is not None and quarantine.get("scope_sha256") != hashlib.sha256(args.scope.read_bytes()).hexdigest():
        parser.error("quarantine does not match frozen scope sidecar")
    rows, report = merge(read_jsonl(args.facts), read_jsonl(args.prior), read_jsonl(args.scope), quarantine)
    report["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name in ("facts", "prior", "scope", "quarantine") if (path := getattr(args, name))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
