#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-analysis", type=Path, required=True)
    parser.add_argument("--rebinding-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    visible = json.loads(args.visible_analysis.read_text(encoding="utf-8"))
    rebinding = json.loads(args.rebinding_audit.read_text(encoding="utf-8"))
    visible_rows = visible["results"]
    rejected = rebinding["rejected"]
    if len(visible_rows) != rebinding["rows"]:
        raise ValueError("source row count mismatch")

    results = []
    for item in rejected:
        source_index = int(item["index"])
        source = visible_rows[source_index - 1]
        if source["question"] != item["question"]:
            raise ValueError(f"question mismatch at source row {source_index}")
        answer = str(item.get("candidate_answer") or "").strip()
        if not answer:
            continue
        results.append(
            {
                "id": f"{source['id']}::rejected-candidate",
                "source_id": source["id"],
                "source_index": source_index,
                "question": source["question"],
                "current_round": source.get("current_round", 2),
                "max_retrieval_rounds": source.get("max_retrieval_rounds", 2),
                "evidence_brief": source["evidence_brief"],
                "grounded": {
                    "next_action": "answer_directly",
                    "answer": answer,
                },
                "original_displayed": source["grounded"],
                "validator_issues": item.get("validator_issues") or [],
                "facts": item.get("facts") or [],
                "status": "ok",
            }
        )

    report = {
        "version": "asa_rejected_candidate_visible_analysis_v1",
        "evaluation_only": True,
        "training_ready": False,
        "source_visible_analysis_sha256": sha256(args.visible_analysis),
        "source_rebinding_audit_sha256": sha256(args.rebinding_audit),
        "source_rows": len(visible_rows),
        "validator_rejected_rows": len(rejected),
        "auditable_nonempty_candidates": len(results),
        "boundaries": [
            "Only the model's original rejected final-answer text is audited.",
            "Evidence is exactly the final-round evidence visible to that model run.",
            "No rejected answer is promoted or used as a training label by this artifact.",
            "A content-supported candidate still requires a separately proven safe grounding repair.",
        ],
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_rows": len(visible_rows),
                "validator_rejected_rows": len(rejected),
                "auditable_nonempty_candidates": len(results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
