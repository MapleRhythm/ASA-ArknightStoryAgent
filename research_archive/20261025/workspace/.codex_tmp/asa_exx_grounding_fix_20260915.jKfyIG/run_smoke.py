"""Eight preselected full-chain regression questions, no gold/old answers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

QIDS = [6, 8, 10, 11, 13, 16, 18, 19]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    questions = args.questions.read_text().splitlines()
    assert sha(args.questions) == "8217bab42c62d6130582d970a46400434eefeab28bdb9eed3771fb89e2fcb4e7"
    selected = [questions[q - 1] for q in QIDS]
    question_file = args.out / "questions.txt"
    question_file.write_text("\n".join(selected) + "\n")
    sys.path.insert(0, str(args.repo / "src"))
    from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    state = {"index": 0, "q": None, "call": 0}
    original_run = CPUInferencePipeline.run
    original_generate = VllmRunner.generate
    started = time.monotonic()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "kind": "targeted_full_chain_smoke_not_20_question_score", "qids": QIDS,
        "release_lock": json.loads((args.repo / "RELEASE_LOCK.json").read_text()),
        "questions_sha256": sha(question_file), "old_answers_or_gold_sent": False,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (args.out / "calls.jsonl").open("x") as calls:
        def emit(row):
            calls.write(json.dumps({**state, **row}, ensure_ascii=False) + "\n")
            calls.flush()

        def run(self, question, dialogue_context="", progress_callback=None):
            assert question == selected[state["index"]]
            state.update(q=QIDS[state["index"]], call=0)
            state["index"] += 1
            return original_run(self, question, dialogue_context, progress_callback)

        def generate(self, prompt, **kw):
            state["call"] += 1
            emit({"event": "request", "prompt": prompt, "options": kw})
            try:
                text = original_generate(self, prompt, **kw)
                emit({"event": "response", "text": text, "diagnostics": self.last_generation_diagnostics})
                return text
            except Exception as exc:
                emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
                raise

        CPUInferencePipeline.run = run
        VllmRunner.generate = generate
        sys.argv = [str(args.repo / "scripts/run_exx_official.py"),
                    "--questions-file", str(question_file),
                    "--batch-output", str(args.out / "results.jsonl"), "--enforce-eager"]
        code = 0
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            code = exc.code or 0
    rows = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    complete = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
        "count": len(rows), "questions_match": [r["question"] for r in rows] == selected,
        "runtime_errors": sum(bool(r.get("error")) for r in rows),
        "elapsed_sec": time.monotonic() - started,
        "results_sha256": sha(args.out / "results.jsonl"),
    }
    (args.out / "completion.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2))
    print(json.dumps(complete), flush=True)
    if code or len(rows) != len(QIDS) or complete["runtime_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
