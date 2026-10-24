"""Frozen full-evidence reader comparison, not an end-to-end retrieval test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from bonsai_api import BonsaiClient

SELECTED = [1, 3, 21, 23, 26, 30, 31, 39, 40, 50]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18127")
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.baseline / "calls.jsonl").read_text().splitlines()]
    audits = [json.loads(line) for line in (args.baseline / "audit.jsonl").read_text().splitlines()]
    args.out.mkdir(exist_ok=False, parents=True)
    client = BonsaiClient(args.url)
    requests = {(r["q"], r["call"]): r for r in rows if r["event"] == "request"}
    engine_results = {(r["q"], r["call"]): r for r in rows if r["event"] == "engine_result"}
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "frozen_9b_evidence_reader_only", "selected_questions": SELECTED,
        "baseline_calls_sha256": sha(args.baseline / "calls.jsonl"),
        "baseline_results_sha256": sha(args.baseline / "results.jsonl"),
        "thinking": args.thinking, "answer_validation": False,
        "old_answers_or_gold_sent": False, "server_properties": client.request("/props"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    started = time.monotonic()
    errors = 0
    with (args.out / "results.jsonl").open("x") as out:
        for q in SELECTED:
            # In this baseline all questions had one conclusion call (call 2).
            request = requests[q, 2]
            prompt = request["original_prompt"]
            audit = next(r for r in audits if r["q"] == q and r["event"] == "prompt_evidence"
                         and r["builder"] == "build_conclusion_prompt")
            assert len(audit["documents"]) == 20
            assert len(audit["checks"]) == 20 and all(c["full"] for c in audit["checks"])
            normalized = re.sub(r"\s+", " ", prompt)
            for doc in audit["documents"]:
                text = re.sub(r"\s+", " ", doc["clean_text"]).strip()
                # Strip only pipeline-internal evidence metadata, as the old runner did.
                if text not in normalized:
                    check = next(c for c in audit["checks"] if c["id"] == doc["id"])
                    assert check["full"]
            settings = engine_results[q, 2]
            options = request["options"]
            begin = time.monotonic()
            result = {
                "q": q, "question": request["question"],
                "original_prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "evidence_count": 20, "all_full_evidence": True,
                "evidence_ids": [d["id"] for d in audit["documents"]],
            }
            try:
                generated = client.generate(
                    prompt, thinking=args.thinking,
                    max_tokens=16384 if args.thinking else settings["max_tokens"],
                    temperature=1.0 if args.thinking else options.get("temperature", 0.2),
                    top_p=0.95 if args.thinking else options.get("top_p", 0.9),
                    repeat_penalty=1.0 if args.thinking else options.get("repeat_penalty", 1.05),
                )
                result.update(generated)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
            result["elapsed_sec"] = time.monotonic() - begin
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({k: result.get(k) for k in
                              ("q", "question", "finish_reason", "error", "elapsed_sec")},
                             ensure_ascii=False), flush=True)
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "count": len(SELECTED),
        "errors": errors, "elapsed_sec": time.monotonic() - started,
        "results_sha256": sha(args.out / "results.jsonl"),
    }
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion), flush=True)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
