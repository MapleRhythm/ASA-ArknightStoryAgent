#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_BIN = Path("/home/zhb/miniconda3/envs/train/bin/python")

DEFAULT_QUESTIONS = [
    "拉普兰德干了什么坏事？",
    "PCS是什么？",
    "真龙为什么要启动不反？",
    "澄闪在卡拉顿城识破的阴谋是什么？",
    "博士为什么要关闭全舰防御系统？",
    "安多恩为什么多次前往拉特兰？",
    "凛冬是怎样成为学生自治团领袖的？",
    "真理的本名是什么？她和古米、凛冬是什么关系？",
    "炎景公主一事具体指什么？",
    "岁陵那场危机是什么？",
]


def read_questions(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_QUESTIONS)
    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            questions.append(text)
    return questions


def latest_run_dir(log_dir: Path, before: set[Path]) -> Path | None:
    candidates = [path for path in log_dir.iterdir() if path.is_dir() and path not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API no-verifier RAG ablation on a fixed question set.")
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "cuda"))
    parser.add_argument(
        "--pipeline-mode",
        default=None,
        choices=("standard", "answer_then_retrieve_refine", "question_retrieve_answer_retrieve_refine"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    runtime_config = args.runtime_config
    if not runtime_config.is_absolute():
        runtime_config = PROJECT_ROOT / runtime_config
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    log_root = output_dir / "runs"
    log_root.mkdir(parents=True, exist_ok=True)

    questions = read_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CONDA_NO_PLUGINS"] = "true"
    env["DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT / '.python_packages' / 'train'}:{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"

    records: list[dict] = []
    started_all = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        before = {path for path in log_root.iterdir() if path.is_dir()}
        cmd = [
            str(args.python_bin),
            "api-mode/run_api_inference.py",
            "--runtime-config",
            str(runtime_config),
            "--log-dir",
            str(log_root),
            "--answer-only",
            "--device",
            args.device,
            question,
        ]
        if args.pipeline_mode:
            cmd.extend(["--pipeline-mode", args.pipeline_mode])
        print(f"[run] {index}/{len(questions)} {question}", flush=True)
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        elapsed = time.perf_counter() - started
        run_dir = latest_run_dir(log_root, before)
        summary = load_json(run_dir / "summary.json") if run_dir else {}
        result = load_json(run_dir / "result.json") if run_dir else {}
        record = {
            "index": index,
            "question": question,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "run_dir": str(run_dir) if run_dir else "",
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
            "summary": summary,
            "hypothesis": result.get("hypothesis") if isinstance(result, dict) else None,
            "answer": summary.get("answer") or result.get("answer") or "",
            "evidence_head": (result.get("evidence") or [])[:5] if isinstance(result, dict) else [],
            "retrieval_trace": result.get("retrieval_trace") if isinstance(result, dict) else None,
        }
        records.append(record)
        (output_dir / "records.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            print(proc.stderr[-1200:], file=sys.stderr, flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = {
        "runtime_config": str(runtime_config),
        "question_count": len(questions),
        "success_count": sum(1 for item in records if item["returncode"] == 0),
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "records_path": str(output_dir / "records.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
