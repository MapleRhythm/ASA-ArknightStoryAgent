#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = SCRIPT_ROOT / "canonical_eval_transfer_20260908.Vfpf4E"
sys.path.insert(0, str(AUDIT_ROOT))
import answer_semantics_content_v2 as content
import judge_visible_answers as transport


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument(
        "--endpoint",
        default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
    )
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--attempt-label", default="retry1")
    args = parser.parse_args()
    journal_path = args.output.with_suffix(".progress.jsonl")
    if args.output.exists() or journal_path.exists():
        parser.error("output/journal already exists")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    initial = json.loads(args.initial.read_text(encoding="utf-8"))
    rows = analysis["results"]
    if initial["source_analysis_sha256"] != sha256(args.analysis):
        raise ValueError("initial audit is not bound to this analysis")
    if initial["rows"] != len(rows):
        raise ValueError("initial audit row count mismatch")
    if initial["requested_model"] != args.model or initial["endpoint"] != args.endpoint:
        raise ValueError("model or endpoint mismatch")

    source_by_id = {row["id"]: row for row in rows}
    initial_by_id = {row["id"]: row for row in initial["results"]}
    if (
        len(source_by_id) != len(rows)
        or len(initial_by_id) != len(rows)
        or set(source_by_id) != set(initial_by_id)
    ):
        raise ValueError("source/initial ID mismatch")

    transport.PROTOCOL = content.PROTOCOL
    transport.messages = content.messages
    transport.validate = content.validate

    retained = []
    pending = []
    for row in rows:
        old = dict(initial_by_id[row["id"]])
        if old.get("status") == "ok":
            old.setdefault("attempt", "initial")
            old.setdefault(
                "audit_max_tokens",
                initial.get("max_tokens")
                or initial.get("retry_max_tokens")
                or initial.get("initial_max_tokens"),
            )
            retained.append(old)
        else:
            pending.append(row)

    key = (
        os.environ.get("BIGMODEL_API_KEY")
        or os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("GLM_API_KEY")
    )
    if pending and not key:
        parser.error("provider key unavailable")

    retried = []
    with journal_path.open("x", encoding="utf-8") as journal:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(transport.judge, row, args, key): row for row in pending
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                result["attempt"] = args.attempt_label
                result["audit_max_tokens"] = args.max_tokens
                retried.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(
                    json.dumps(
                        {
                            "completed": len(retried),
                            "total": len(pending),
                            "id": result["id"],
                            "status": result["status"],
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    final_by_id = {row["id"]: row for row in [*retained, *retried]}
    if len(final_by_id) != len(rows) or set(final_by_id) != set(source_by_id):
        raise ValueError("merged audit ID mismatch")
    results = [final_by_id[row["id"]] for row in rows]
    report = {
        "version": f"visible_answer_content_audit_v3_{args.attempt_label}",
        "protocol": content.PROTOCOL,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_only": True,
        "training_ready": False,
        "judge_type": "model_not_human",
        "source_analysis_sha256": sha256(args.analysis),
        "previous_audit_sha256": sha256(args.initial),
        "retry_script_sha256": sha256(Path(__file__)),
        "rubric_script_sha256": sha256(AUDIT_ROOT / "answer_semantics_content_v2.py"),
        "transport_script_sha256": sha256(AUDIT_ROOT / "judge_visible_answers.py"),
        "requested_model": args.model,
        "endpoint": args.endpoint,
        "previous_default_max_tokens": (
            initial.get("max_tokens")
            or initial.get("retry_max_tokens")
            or initial.get("initial_max_tokens")
        ),
        "retry_max_tokens": args.max_tokens,
        "rows": len(rows),
        "retained_initial_successes": len(retained),
        "retried_initial_failures": len(pending),
        "statuses": dict(Counter(row["status"] for row in results)),
        "results": results,
        "limitations": [
            "Only failed initial judge calls were retried; successful initial calls are retained byte-for-byte apart from audit annotations.",
            "Per-row audit_max_tokens records whether the judgement used the initial or retry budget.",
            "Model judgements remain provisional and require source review.",
            "No result is an approved training label or independently certified production accuracy.",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"rows": len(rows), "statuses": report["statuses"]},
            ensure_ascii=False,
        )
    )
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
