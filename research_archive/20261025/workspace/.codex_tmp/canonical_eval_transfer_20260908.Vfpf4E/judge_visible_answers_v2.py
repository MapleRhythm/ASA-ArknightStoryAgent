"""Evidence-only audit with retained pipeline failures and hash-checked reuse.

Uses the frozen v1 rubric unchanged. Artifacts remain provisional evaluation,
not training labels. All selected input rows remain in the output denominator.
"""
import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from judge_visible_answers import PROTOCOL, digest, judge, messages, validate


def expected_key(row, args):
    payload = {"model": args.model, "messages": messages(row),
               "temperature": 0, "max_tokens": args.max_tokens}
    return digest({"payload": payload, "endpoint": args.endpoint,
                   "id": row["id"], "grounded": row["grounded"]})


def accepted_reuse(row, args, previous):
    if previous.get("status") != "ok" or previous.get("input_key") != expected_key(row, args):
        return None
    if (previous.get("requested_model") != args.model or previous.get("response_model") != args.model
            or previous.get("finish_reason") != "stop" or previous.get("id") != row["id"]):
        return None
    validate(previous["judgement"], row)
    result = copy.deepcopy(previous)
    result["reused_exact_request"] = True
    return result


def unavailable(row):
    return {
        "id": row["id"], "status": "not_auditable_pipeline_error", "protocol": PROTOCOL,
        "pipeline_status": row.get("status"), "pipeline_error": row.get("error"),
        "pipeline_error_stage": row.get("error_stage"), "denominator_retained": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    parser.add_argument("--endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    journal_path = args.output.with_suffix(".progress.jsonl")
    if args.output.exists() or journal_path.exists():
        parser.error("output/journal exists; choose a new artifact")
    rows = json.loads(args.analysis.read_text())["results"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        parser.error("nonempty unique rows required")
    previous = {}
    for path in args.reuse:
        raw = path.read_text()
        prior = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                 if path.suffix == ".jsonl" else json.loads(raw)["results"])
        for result in prior:
            if result.get("status") == "ok":
                previous[(result["id"], result["input_key"])] = result
    results, pending = [], []
    for row in rows:
        if row.get("status") != "ok" or "grounded" not in row:
            results.append(unavailable(row))
            continue
        prior = previous.get((row["id"], expected_key(row, args)), {})
        reused = accepted_reuse(row, args, prior)
        if reused:
            results.append(reused)
        else:
            pending.append(row)
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    if pending and not key:
        parser.error("no provider key in environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("x", encoding="utf-8") as journal:
        for result in results:
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge, row, args, key): row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["status"] == "ok" and result.get("response_model") != args.model:
                    result["status"] = "response_model_mismatch"
                results.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"completed": len(results), "total": len(rows),
                                  "id": result["id"], "status": result["status"],
                                  "error": result.get("error")}), flush=True)
    order = {row["id"]: i for i, row in enumerate(rows)}
    results.sort(key=lambda r: order[r["id"]])
    if len(results) != len(rows) or {r["id"] for r in results} != set(order):
        raise ValueError("audit denominator changed")
    report = {
        "version": "visible_answer_audit_v2", "protocol": PROTOCOL,
        "evaluation_only": True, "training_ready": False, "judge_type": "model_not_human",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_analysis_sha256": hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        "judge_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rubric_script_sha256": hashlib.sha256(Path(__file__).with_name("judge_visible_answers.py").read_bytes()).hexdigest(),
        "requested_model": args.model, "endpoint": args.endpoint, "max_tokens": args.max_tokens,
        "reuse_sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.reuse],
        "rows": len(rows), "api_requests": len(pending),
        "statuses": dict(Counter(r["status"] for r in results)),
        "results": results,
        "limitations": [
            "Evidence-only provisional model judgments, not independent human gold or production accuracy.",
            "Judge errors and unavailable pipeline outputs remain in the full denominator, not rated correct.",
            "Exact-request reuse requires matching request hash, response model, finish and valid schema.",
            "This displayed-answer rubric differs from historical Exx set audit v3.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "api_requests": len(pending), "statuses": report["statuses"]}))
    return int(any(r["status"] not in {"ok", "not_auditable_pipeline_error"} for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
