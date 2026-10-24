"""Aggregate human-readable assistant decisions without repairing any answers."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_native46 import audit
from native46_eval import file_sha, rows

ROOT = Path(__file__).resolve().parent


def aggregate(run, review_path, old_path):
    integrity = audit(run, old_path, require_complete=True)
    review = json.loads(review_path.read_text())
    tasks = rows(run / "tasks.jsonl")
    records = review["cases"]
    assert [r["position"] for r in records] == list(range(1, 47))
    old_ids = set(json.loads(old_path.read_text())["ids"])
    cases = dict(zip((t["id"] for t in tasks), records))
    originals = {}
    for stem in ("fourb", "bonsai_shard0", "bonsai_shard1"):
        for output in rows(run / f"{stem}.results.jsonl"):
            originals[output["arm"], output["id"]] = output
    assert len(originals) == 92
    for task, record in zip(tasks, records):
        assert set(record["source_eids"]) <= {d["evidence_id"] for d in task["documents"]}
        for arm in ("fourb", "bonsai"):
            decision = record[arm]
            assert decision["category"] in {"complete", "partial", "wrong", "no_answer"}
            assert type(decision["core_complete"]) is bool
            assert decision["category"] != "complete" or decision["core_complete"]
            assert decision["note"].strip()
            original = originals[arm, task["id"]]
            action = (original["parsed_output_for_offline_inspection_only"] or {}).get("next_action")
            if decision["category"] == "no_answer":
                assert action == "retrieve_more"
                assert decision["action"] in {
                    "reasonable_retrieval", "unnecessary_retrieval", "retrieval_after_round_limit"
                }
    summary = {}
    subsets = {
        "all46": set(cases),
        "previous8": old_ids,
        "new38": set(cases) - old_ids,
        "evidence_gap6_diagnostic_only": {
            t["id"] for t, r in zip(tasks, records) if r.get("evidence_gap")
        },
    }
    for arm in ("fourb", "bonsai"):
        summary[arm] = {}
        for name, ids in subsets.items():
            selected = [cases[i][arm] for i in ids]
            counts = Counter(d["category"] for d in selected)
            n = len(selected)
            summary[arm][name] = {
                "n": n,
                "categories": {k: counts[k] for k in ("complete", "partial", "wrong", "no_answer")},
                "complete_rate": counts["complete"] / n if n else None,
                "core_complete_diagnostic_not_accuracy": sum(d["core_complete"] for d in selected),
            }
        all_decisions = [r[arm] for r in records]
        summary[arm]["retrieval_actions"] = dict(Counter(
            d["action"] for d in all_decisions if d["category"] == "no_answer"
        ))
        summary[arm]["content_correct_despite_invalid_json"] = [
            r["position"] for t, r in zip(tasks, records)
            if r[arm]["category"] == "complete"
            and originals[arm, t["id"]]["parsed_output_for_offline_inspection_only"] is None
        ]
        summary[arm]["borderline_positions"] = [
            r["position"] for r in records if r[arm].get("borderline")
        ]
        n_correct = summary[arm]["all46"]["categories"]["complete"]
        n_borderline = len(summary[arm]["borderline_positions"])
        summary[arm]["borderline_liberal_complete_count"] = n_correct + n_borderline
        summary[arm]["borderline_liberal_rate"] = (n_correct + n_borderline) / 46
        groups = {}
        for task, record in zip(tasks, records):
            groups.setdefault(task["question_group"], []).append(record[arm]["category"] == "complete")
        summary[arm]["question_group_equal_weight"] = {
            "n": len(groups),
            "complete_rate": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
        }
    paired = Counter(
        (
            "both" if r["fourb"]["category"] == r["bonsai"]["category"] == "complete"
            else "fourb_only" if r["fourb"]["category"] == "complete"
            else "bonsai_only" if r["bonsai"]["category"] == "complete"
            else "neither"
        )
        for r in records
    )
    return {
        "integrity_passed": integrity["integrity_passed"],
        "reviewed_decisions": 92,
        "review_sha256": file_sha(review_path),
        "tasks_sha256": file_sha(run / "tasks.jsonl"),
        "results_sha256": {
            stem: file_sha(run / f"{stem}.results.jsonl")
            for stem in ("fourb", "bonsai_shard0", "bonsai_shard1")
        },
        "format_and_citation_binding_excluded_from_content_score": True,
        "summary": summary,
        "paired_complete": dict(paired),
        "caution": "Assistant adjudication, historical development questions, fixed evidence; not independent human gold, full-chain RAG, deployment assurance, or causal isolation of P-tuning.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "native46_4b_sftkto_vs_bonsai_pt_20261014")
    parser.add_argument("--review", type=Path, default=ROOT / "NATIVE46_CONTENT_REVIEW_20261014.json")
    parser.add_argument("--old-selection", type=Path, default=ROOT / "heldout8_selection_20261014.json")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.out, args.review, args.old_selection), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
