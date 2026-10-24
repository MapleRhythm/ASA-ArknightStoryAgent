"""Verify two immutable review runs; --verify-only makes no external calls or writes."""
import argparse
import json
from pathlib import Path
from collections import Counter

import annotation_protocol as p
from expand_reader_errors_v3 import review_messages
import glm_review_original_20261018 as original


def build_summary():
    root = Path(__file__).resolve().parent
    source = root / "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"
    rows = {r["id"]: r for r in original.load(source)}
    effective = {}
    calls = tokens = 0
    for name in ("glm_original_review_301_486_20261018",
                 "glm_original_review_472_recovery_20261019"):
        run = root / name
        settings = p.Settings(**json.loads((run / "run_settings.json").read_text()))
        assert settings.endpoint == p.CODING_PLAN_ENDPOINT
        assert settings.model == "glm-5.3"
        status = json.loads((run / "status.json").read_text())
        calls += status["api_calls"]
        tokens += status["accounted_tokens"]
        receipt_rows = list(p.json_rows(run / "api_receipts.jsonl"))
        receipts = {r["call_id"]: r for r in receipt_rows}
        assert len(receipts) == len(receipt_rows) == status["api_calls"]
        assert sum(r["usage"]["total_tokens"] for r in receipt_rows) == status["accounted_tokens"]
        for receipt in receipt_rows:
            row = rows[receipt["task_id"]]
            assert receipt["endpoint"] == settings.endpoint
            assert receipt["requested_model"] == receipt["returned_model"] == settings.model
            assert receipt["phase"] == "review_original"
            assert receipt["teacher_prompt_sha256"] == row["teacher_prompt_sha256"]
            assert receipt["cache_key"] == p.sha(p.compact({
                "phase": "review_original",
                "messages": review_messages(row, json.loads(row["target"])),
                "settings": settings.cache_settings(),
            }))
        for case in json.loads((run / "reviews.json").read_text()):
            row = rows[case["id"]]
            assert case["position"] == row["review_position"]
            if case["state"] == "pending":
                effective.setdefault(case["id"], dict(case, run=name))
                continue
            result = case["result"]
            receipt = result["review"]
            assert receipt == receipts[receipt["call_id"]]
            assert receipt["task_id"] == case["id"]
            assert receipt["accepted"] and receipt["finish_reason"] == "stop"
            assert result["original_target_sha256"] == p.sha(row["target"])
            expected = p.sha(p.compact({
                "phase": "review_original",
                "messages": review_messages(row, json.loads(row["target"])),
                "settings": settings.cache_settings(),
            }))
            assert expected == receipt["cache_key"]
            review = p.check_review(receipt["parsed"])
            passed = (review["label_supported"] and review["decision_appropriate"]
                      and not review["issues"])
            assert case["state"] == ("accepted" if passed else "quarantined")
            assert case["id"] not in effective or effective[case["id"]]["state"] == "pending"
            effective[case["id"]] = dict(case, run=name)
    assert len(effective) == 186
    counts = Counter(c["state"] for c in effective.values())
    assert counts == {"accepted": 185, "quarantined": 1}
    summary = dict(
        source_sha256=original.SOURCE_SHA, start=301, end=486, total=186,
        states=dict(counts), api_calls=calls, accounted_tokens=tokens,
        all186_api_reviewed=True, all486_semantically_reviewed=False,
        full20_prompt_and_original_target_integrity_verified=True,
        ready_for_training=False, independent_human_gold=False,
        not_model_QA_accuracy=True,
        quarantined=[dict(position=c["position"], id=c["id"],
                          issues=c["result"]["review"]["parsed"]["issues"])
                     for c in effective.values() if c["state"] == "quarantined"],
        cases=[dict(id=c["id"], position=c["position"], state=c["state"], run=c["run"])
               for c in sorted(effective.values(), key=lambda c: c["position"])],
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="Recompute and compare the existing summary without writing.")
    args = parser.parse_args()
    summary = build_summary()
    path = Path(__file__).resolve().parent / "GLM_ORIGINAL_REVIEW_SUMMARY_20261019.json"
    if args.verify_only:
        assert json.loads(path.read_text()) == summary, "saved summary mismatch"
    else:
        p.atomic_json(path, summary)
    print(p.compact({k: v for k, v in summary.items() if k != "cases"}))


if __name__ == "__main__":
    main()
