#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


INTERNAL_EVIDENCE_META_RE = re.compile(
    r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]+\]\s*|\[E\d+\]\s*"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_text(item: dict[str, Any]) -> str:
    source = str(item.get("evidence_chain_text") or item.get("clean_text") or "")
    source = INTERNAL_EVIDENCE_META_RE.sub("", source)
    return re.sub(r"\s+", " ", source).strip()


def reconstruct_final_evidence(
    row: dict[str, Any],
    trace_row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    by_source = {str(item.get("id") or ""): item for item in row.get("evidence") or []}
    lines: list[str] = []
    checks: list[dict[str, Any]] = []
    for packing in trace_row.get("evidence_packing") or []:
        if packing.get("status") != "included":
            continue
        evidence_id = str(packing.get("evidence_id") or "")
        source_id = str(packing.get("source_id") or "")
        item = by_source.get(source_id)
        if item is None:
            checks.append(
                {
                    "evidence_id": evidence_id,
                    "source_id": source_id,
                    "status": "source_missing_from_final_evidence",
                }
            )
            continue
        rendered = f"[{evidence_id}] {prompt_text(item)}"
        expected_chars = int(packing.get("block_chars") or 0)
        checks.append(
            {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "status": "exact_length_match" if len(rendered) == expected_chars else "length_mismatch",
                "expected_block_chars": expected_chars,
                "reconstructed_block_chars": len(rendered),
            }
        )
        lines.append(rendered)
    return "\n".join(lines), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    source_rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_rows: list[dict[str, Any]] = []
    check_statuses: dict[str, int] = {}
    for index, row in enumerate(source_rows, start=1):
        trace = row.get("retrieval_trace") or []
        if not trace:
            output_rows.append(
                {
                    "id": f"fixed58-{index:03d}",
                    "status": "error",
                    "error": "missing_retrieval_trace",
                    "question": row.get("question"),
                }
            )
            continue
        final_trace = trace[-1]
        conclusion = final_trace.get("conclusion") or {}
        evidence_brief, checks = reconstruct_final_evidence(row, final_trace)
        for check in checks:
            status = str(check["status"])
            check_statuses[status] = check_statuses.get(status, 0) + 1
        row_status = (
            "ok"
            if checks
            and all(check["status"] == "exact_length_match" for check in checks)
            and evidence_brief
            else "error"
        )
        output_rows.append(
            {
                "id": f"fixed58-{index:03d}",
                "status": row_status,
                "error": "" if row_status == "ok" else "visible_evidence_reconstruction_failed",
                "question": row.get("question"),
                "current_round": final_trace.get("round"),
                "max_retrieval_rounds": args.max_rounds,
                "evidence_brief": evidence_brief,
                "evidence_checks": checks,
                "grounded": {
                    "next_action": conclusion.get("next_action"),
                    "answer": row.get("answer"),
                },
            }
        )

    report = {
        "version": "fixed58_visible_answer_analysis_v1",
        "evaluation_only": True,
        "training_ready": False,
        "source_results": str(args.results),
        "source_results_sha256": sha256(args.results),
        "rows": len(output_rows),
        "statuses": {
            "ok": sum(row.get("status") == "ok" for row in output_rows),
            "error": sum(row.get("status") != "ok" for row in output_rows),
        },
        "evidence_check_statuses": check_statuses,
        "limitations": [
            "Reconstructs only the exact final-round evidence visible to the displayed answer.",
            "No historical reference answer or hidden corpus text is included.",
            "This artifact is for provisional semantic review and is not training-ready.",
        ],
        "results": output_rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "rows": report["rows"],
            "statuses": report["statuses"],
            "evidence_check_statuses": check_statuses,
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
