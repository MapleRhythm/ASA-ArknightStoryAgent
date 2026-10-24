"""Content-aware audit with accurately bound rubric, transport and source hashes."""
import argparse
from collections import Counter
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path

import answer_semantics_content_v2 as content
import judge_visible_answers as transport


def install_rubric():
    # Only this standalone process is affected. Frozen source files are intact.
    transport.PROTOCOL = content.PROTOCOL
    transport.messages = content.messages
    transport.validate = content.validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--endpoint", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions")
    parser.add_argument("--model", default="glm-5.3")
    args = parser.parse_args()
    journal_path = args.output.with_suffix(".progress.jsonl")
    if args.output.exists() or journal_path.exists():
        parser.error("Existing output/journal; choose a new artifact")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    rows = json.loads(args.analysis.read_text())["results"]
    if not rows or len(rows) != len({r["id"] for r in rows}):
        parser.error("nonempty unique rows required")
    install_rubric()
    pending, results = [], []
    for row in rows:
        if row.get("status") == "ok" and "grounded" in row:
            pending.append(row)
        else:
            results.append({
                "id": row["id"], "status": "not_auditable_pipeline_error",
                "pipeline_error": row.get("error"), "denominator_retained": True,
            })
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    if pending and not key:
        parser.error("provider key unavailable")
    with journal_path.open("x") as journal:
        for result in results:
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(transport.judge, row, args, key) for row in pending]
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
    by_id = {r["id"]: r for r in results}
    assert len(results) == len(by_id) == len(rows) and set(by_id) == {r["id"] for r in rows}
    run = Path(__file__).resolve().parent
    report = {
        "version": "visible_answer_content_audit_v3", "protocol": content.PROTOCOL,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "judge_type": "model_not_human",
        "source_analysis_sha256": sha(args.analysis), "judge_script_sha256": sha(Path(__file__)),
        "rubric_script_sha256": sha(run / "answer_semantics_content_v2.py"),
        "transport_script_sha256": sha(run / "judge_visible_answers.py"),
        "requested_model": args.model, "max_tokens": args.max_tokens, "endpoint": args.endpoint,
        "rows": len(rows), "api_requests": len(pending), "statuses": dict(Counter(r["status"] for r in results)),
        "results": [by_id[r["id"]] for r in rows],
        "limitations": [
            "New content-aware rubric: old verdicts are not reused or silently relabeled.",
            "Supported spans must be verbatim, but their substance and correctness still require source review.",
            "Model judgments remain provisional; no independent gold or production accuracy claim.",
            "All records including pipeline and judge failures remain in the denominator.",
        ],
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "statuses": report["statuses"]}))
    return int(any(r["status"] not in {"ok", "not_auditable_pipeline_error"} for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
