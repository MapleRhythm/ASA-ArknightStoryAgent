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
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    visible_rows = json.loads(
        args.visible_analysis.read_text(encoding="utf-8")
    )["results"]
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    replay_rows = replay["results"]
    if len(visible_rows) != len(replay_rows):
        raise ValueError("row count mismatch")

    results = []
    for visible, candidate in zip(visible_rows, replay_rows, strict=True):
        if visible["id"] != candidate["id"]:
            raise ValueError(f"ID mismatch: {visible['id']} != {candidate['id']}")
        if (
            candidate.get("status") != "ok"
            or candidate.get("candidate_action") == candidate.get("original_action")
        ):
            continue
        results.append(
            {
                "id": f"{visible['id']}::safe-rebinding-candidate",
                "source_id": visible["id"],
                "question": visible["question"],
                "current_round": visible.get("current_round", 2),
                "max_retrieval_rounds": visible.get("max_retrieval_rounds", 2),
                "evidence_brief": visible["evidence_brief"],
                "grounded": {
                    "next_action": candidate["candidate_action"],
                    "answer": candidate["candidate_answer"],
                },
                "grounding_repairs": candidate.get("grounding_repairs") or [],
                "status": "ok",
            }
        )

    report = {
        "version": "asa_rebinding_changed_visible_analysis_v1",
        "evaluation_only": True,
        "training_ready": False,
        "source_visible_analysis_sha256": sha256(args.visible_analysis),
        "source_replay_sha256": sha256(args.replay),
        "rows": len(results),
        "boundaries": [
            "Contains only rows whose final action changes under the isolated safe-rebinding candidate.",
            "Uses the candidate's actually rendered final answer and the original exact visible evidence.",
            "No semantic judgement is converted into a runtime or training label.",
        ],
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
