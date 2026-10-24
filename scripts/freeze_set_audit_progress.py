#!/usr/bin/env python3
"""Freeze a content-checked partial set audit without invoking any model.

Missing/error rows stay unapproved. A snapshot can be used for conservative
data export or a content-bound secondary veto while the original audit runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_exx_set_grounding import (
    PROTOCOL, audited_input_sha256, completion_value, input_key,
    parse_json_object, read_rows, validate_judgement,
)

ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"


def freeze(rows, records, *, endpoint, model, max_tokens):
    if not rows or len({row.get("id") for row in rows}) != len(rows) or any(not row.get("id") for row in rows):
        raise ValueError("source_ids_must_be_nonempty_and_unique")
    keys = {row["id"]: input_key(row, endpoint, model, max_tokens) for row in rows}
    latest, ignored = {}, 0
    for record in records:
        row_id = record.get("id")
        if row_id not in keys or record.get("input_key") != keys[row_id]:
            ignored += 1
            continue
        latest[row_id] = record
    results = []
    for index, row in enumerate(rows):
        item = latest.get(row["id"])
        if item is None:
            results.append({"id": row["id"], "index": index, "status": "pending"})
            continue
        if item.get("audited_input_sha256") != audited_input_sha256(row):
            raise ValueError("progress_content_mismatch")
        if item.get("status") not in {"ok", "error", "ineligible"}:
            raise ValueError("unknown_progress_status")
        if item["status"] == "ok":
            if item.get("finish_reason") != "stop" or item.get("full_evidence") is not True:
                raise ValueError("progress_incomplete_audit")
            validate_judgement(item["judgement"], parse_json_object(completion_value(row)))
        results.append({**item, "index": index})
    counts = Counter(item["status"] for item in results)
    summary = {
        "protocol": PROTOCOL, "endpoint": endpoint, "judge_model": model,
        "max_tokens": max_tokens, "sample_families": False,
        "selected_ids": [row["id"] for row in rows], "rows": len(rows),
        "ok": counts["ok"], "errors": counts["error"],
        "ineligible": counts["ineligible"], "pending": counts["pending"],
        "snapshot_only": True, "ignored_other_contract_records": ignored,
        "complete_audit": counts["ok"] + counts["ineligible"] == len(rows),
        "set_support": dict(Counter(
            item["judgement"]["set_support"] for item in results if item["status"] == "ok"
        )),
        "context_sufficiency": dict(Counter(
            item["judgement"]["context_sufficiency"] for item in results if item["status"] == "ok"
        )),
        "response_models": sorted({
            item["response_model"] for item in results if item.get("response_model")
        }),
        "missing_response_model": sum(
            item["status"] == "ok" and not item.get("response_model") for item in results
        ),
    }
    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("output must be new and max-tokens positive")
    raw = args.progress.read_bytes()
    lines = raw.splitlines()
    # The writer fsyncs whole JSONL records. A concurrent read may catch only
    # the last line partway through its write; ignore that unfinished tail.
    unfinished_tail = bool(raw and not raw.endswith(b"\n"))
    if unfinished_tail:
        lines = lines[:-1]
    records = [json.loads(line) for line in lines if line.strip()]
    result = freeze(
        read_rows(args.predictions), records, endpoint=args.endpoint,
        model=args.model, max_tokens=args.max_tokens,
    )
    result["summary"].update(
        predictions=str(args.predictions),
        predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        progress=str(args.progress),
        progress_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        unfinished_tail_ignored=unfinished_tail,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
