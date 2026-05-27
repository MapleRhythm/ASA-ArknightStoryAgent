#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = PROJECT_ROOT / "scripts/run_action_target_kto_full_pipeline.sh"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/eval_action_target_bufan_regression.json"
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "api-mode/runtime_api.json"

QUESTIONS = [
    "真龙为什么要启动不反？",
    "真龙为什么要启动“不反”？",
    "真龙启动“不反”的直接目的是什么？",
    "真龙动用不反是为了解决什么？",
    "真龙开启不反的原因是什么？",
    "不反为什么需要由真龙启动？",
    "真龙启动不反是为了证明自己能力吗？",
    "真龙启动不反和岁陵危机有什么关系？",
    "真龙为什么要以自己的性命为代价启用不反？",
    "启动不反的直接原因和真龙对大炎的不满有什么区别？",
]

CRISIS_TERMS = ("岁陵", "危机")
COST_TERMS = ("性命", "生命", "代价", "万金之躯", "捐躯")
DRIFT_DEMOTION_TERMS = ("不是", "并非", "不能", "不应", "不要", "背景", "补充", "只可作为", "不能替代")
DRIFT_TERMS = (
    "证明自己的能力",
    "证明能力",
    "推动大炎",
    "碌碌无为",
    "大炎本应该做到更多",
    "能量不足以驱动军队",
    "改变现状",
    "昏睡不醒",
    "无法容忍大炎",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def required_groups_for_question(question: str) -> list[tuple[str, ...]]:
    compact_question = normalize(question)
    required: list[tuple[str, ...]] = [CRISIS_TERMS]
    if any(term in compact_question for term in ("由真龙", "性命", "代价", "万金之躯", "为什么需要")):
        required.append(COST_TERMS)
    if any(term in compact_question for term in ("直接目的", "直接原因", "为什么要启动", "为什么要开启", "开启不反的原因")):
        required.append(COST_TERMS)
    return required


def score_answer(answer: str, question: str = "") -> dict[str, Any]:
    compact = normalize(answer)
    required_groups = required_groups_for_question(question)
    required_hits = [
        any(term in compact for term in group)
        for group in required_groups
    ]
    drift_hits = [term for term in DRIFT_TERMS if term in compact]
    demoted_drift_hits: list[str] = []
    for term in drift_hits:
        start = compact.find(term)
        window = compact[max(0, start - 18) : start + len(term) + 18]
        if any(marker in window for marker in DRIFT_DEMOTION_TERMS):
            demoted_drift_hits.append(term)
    # Drift terms are acceptable only when locally rejected or demoted.
    drift_ok = not drift_hits or len(demoted_drift_hits) == len(drift_hits)
    passed = all(required_hits) and drift_ok
    return {
        "passed": passed,
        "required_groups": [list(group) for group in required_groups],
        "required_hits": required_hits,
        "drift_hits": drift_hits,
        "demoted_drift_hits": demoted_drift_hits,
        "drift_ok": drift_ok,
    }


def extract_answer(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    for start in range(len(lines)):
        if not lines[start].startswith("{"):
            continue
        try:
            payload = json.loads("\n".join(lines[start:]))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return str(payload.get("answer") or "")
    for line in reversed(lines):
        if line.startswith("Interactive inference ready.") or line.startswith("Exiting."):
            continue
        return line
    return lines[-1]


def pipeline_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(args.python_bin)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    env["GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    env["ANSWER_ONLY"] = "1"
    return env


def run_batch(args: argparse.Namespace, questions: list[str]) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="bufan_regression_") as tmpdir:
        questions_path = Path(tmpdir) / "questions.txt"
        batch_output = Path(tmpdir) / "answers.jsonl"
        questions_path.write_text("\n".join(questions) + "\n", encoding="utf-8")
        cmd = [
            "bash",
            str(args.pipeline),
            "--disable-crag-refinement",
            "--disable-mmr",
            "--questions-file",
            str(questions_path),
            "--batch-output",
            str(batch_output),
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=pipeline_env(args),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            stderr = str(exc.stderr or "")
            stdout = str(exc.stdout or "")
            return [
                {
                    "question": question,
                    "returncode": 124,
                    "elapsed_sec": round(elapsed, 3),
                    "answer": "",
                    "score": score_answer("", question),
                    "error": f"TimeoutExpired: batch exceeded {args.timeout}s",
                    "stderr_tail": stderr[-4000:],
                    "stdout_tail": stdout[-4000:],
                }
                for question in questions
            ]
        total_elapsed = time.perf_counter() - started
        rows: list[dict[str, Any]] = []
        if batch_output.exists():
            for line in batch_output.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        results: list[dict[str, Any]] = []
        row_by_question = {str(row.get("question") or ""): row for row in rows}
        for question in questions:
            row = row_by_question.get(question, {})
            answer = str(row.get("answer") or "")
            score = score_answer(answer, question)
            error = str(row.get("error") or "")
            if not row:
                error = "missing_batch_result"
            results.append(
                {
                    "question": question,
                    "returncode": proc.returncode if proc.returncode != 0 else (1 if error else 0),
                    "elapsed_sec": row.get("elapsed_sec", round(total_elapsed, 3)),
                    "answer": answer,
                    "score": score,
                    "error": error,
                    "stderr_tail": proc.stderr[-4000:],
                    "stdout_tail": proc.stdout[-4000:],
                }
            )
        return results


def run_question(args: argparse.Namespace, question: str) -> dict[str, Any]:
    cmd = [
        "bash",
        str(args.pipeline),
        "--disable-crag-refinement",
        "--disable-mmr",
        question,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=pipeline_env(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "question": question,
            "returncode": 124,
            "elapsed_sec": round(elapsed, 3),
            "answer": "",
            "score": score_answer("", question),
            "error": f"TimeoutExpired: question exceeded {args.timeout}s",
            "stderr_tail": str(exc.stderr or "")[-4000:],
            "stdout_tail": str(exc.stdout or "")[-4000:],
        }
    elapsed = time.perf_counter() - started
    answer = extract_answer(proc.stdout)
    score = score_answer(answer, question)
    return {
        "question": question,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "answer": answer,
        "score": score,
        "stderr_tail": proc.stderr[-4000:],
        "stdout_tail": proc.stdout[-4000:],
    }


class FakeActionTargetGenerator:
    backend_name = "fake-action-target-drift"

    def __init__(self, question: str, *, max_tokens: int = 4096) -> None:
        self.question = question
        self.max_tokens = max_tokens
        self.calls: list[str] = []

    def describe_runtime(self) -> dict[str, Any]:
        return {"generator_backend": self.backend_name, "runtime_mode": "fake_runtime_regression"}

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        del max_tokens, temperature, top_p, repeat_penalty
        self.calls.append(prompt)
        if "task: user_question_hypothesis_generation" in prompt:
            return json.dumps(
                {
                    "question": self.question,
                    "intent": "plot_reasoning",
                    "query_type": "causality",
                    "entities": ["真龙", "不反"],
                    "keywords": ["真龙", "启动", "开启", "动用", "不反", "岁陵", "危机", "代价", "性命"],
                    "expected_answer_type": "原因/动机",
                    "dialogue_context": "",
                },
                ensure_ascii=False,
            )
        if "请根据以下信息生成当前阶段结论" in prompt:
            compact_question = normalize(self.question)
            if "证明自己能力" in compact_question or "证明自己的能力" in compact_question:
                answer = "不是。真龙启动不反不是为了证明自己的能力，而是为了证明自己的能力并推动大炎发展。"
            elif "解决什么" in compact_question or "直接目的" in compact_question:
                answer = "真龙动用不反是为了解决岁陵危机。"
            elif "为什么需要" in compact_question or "性命" in compact_question or "代价" in compact_question:
                answer = "不反需要由真龙启动，是因为真龙与源石权柄绑定。"
            else:
                answer = "真龙启动不反是为了证明自己的能力，并推动大炎的发展。"
            return json.dumps(
                {
                    "question": self.question,
                    "next_action": "answer_directly",
                    "answer": answer,
                    "missing_slots": [],
                    "clarification_question": "",
                    "follow_up_hypothesis": None,
                },
                ensure_ascii=False,
            )
        return "现有证据不足以确认。"


def build_fake_retriever(args: argparse.Namespace):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from goldenglow.config import EMBEDDING_MODEL_DIR, MINIRAG_GRAPH_PATH, RERANKER_MODEL_DIR, QueryConfig
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever

    runtime_config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}
    configured_minirag = Path(str(retrieval_cfg.get("minirag_index_path") or MINIRAG_GRAPH_PATH))
    minirag_index_path = configured_minirag if configured_minirag.is_absolute() else PROJECT_ROOT / configured_minirag
    reranker_model = RERANKER_MODEL_DIR if args.fake_with_reranker else None
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=reranker_model,
        minirag_index_path=minirag_index_path,
        device=args.fake_device,
    )
    return retriever, retrieval_cfg, inference_cfg, QueryConfig


def build_fake_runtime_pipeline(
    args: argparse.Namespace,
    question: str,
    retriever: Any,
    retrieval_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
    query_config_cls: Any,
):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from goldenglow.inference.cpu_pipeline import CPUInferencePipeline

    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}
    generator = FakeActionTargetGenerator(question)
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=query_config_cls(
            dense_top_k=int(retrieval_cfg.get("dense_top_k", 120)),
            sparse_top_k=int(retrieval_cfg.get("sparse_top_k", 120)),
            minirag_top_k=int(retrieval_cfg.get("minirag_top_k", 120)),
            fusion_top_k=int(retrieval_cfg.get("fusion_top_k", 80)),
            rerank_top_k=int(retrieval_cfg.get("rerank_top_k", 32)),
            minirag_weight=float(retrieval_cfg.get("minirag_weight", 0.35)),
            minirag_mode_weights={str(key): float(value) for key, value in minirag_mode_weights.items()},
            minirag_fusion_mode=str(retrieval_cfg.get("minirag_fusion_mode", "score")),
            reranker_candidate_top_k=int(retrieval_cfg.get("reranker_candidate_top_k", 120)),
            rerank_batch_size=int(retrieval_cfg.get("rerank_batch_size", 4)),
        ),
        max_retrieval_rounds=1,
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 12)),
        enable_mmr=bool(inference_cfg.get("enable_mmr", True)),
        mmr_lambda=float(inference_cfg.get("mmr_lambda", 0.72)),
        enable_pyramid_order=bool(inference_cfg.get("enable_pyramid_order", True)),
        enable_crag_refinement=False,
        answer_grounding_mode=str(inference_cfg.get("answer_grounding_mode", "weak")),
    )
    return pipeline, generator


def run_fake_runtime_batch(args: argparse.Namespace, questions: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    retriever, retrieval_cfg, inference_cfg, query_config_cls = build_fake_retriever(args)
    for index, question in enumerate(questions, start=1):
        print(f"[fake {index}/{len(questions)}] {question}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            pipeline, generator = build_fake_runtime_pipeline(
                args,
                question,
                retriever,
                retrieval_cfg,
                inference_cfg,
                query_config_cls,
            )
            result = pipeline.run(question)
            elapsed = time.perf_counter() - started
            answer = result.answer
            conclusion_prompt = next(
                (prompt for prompt in generator.calls if "请根据以下信息生成当前阶段结论" in prompt),
                "",
            )
            score = score_answer(answer, question)
            score["prompt_has_suiling_crisis"] = "岁陵" in conclusion_prompt and "危机" in conclusion_prompt
            score["prompt_has_life_cost"] = "性命" in conclusion_prompt and "代价" in conclusion_prompt
            if not score["prompt_has_suiling_crisis"] or not score["prompt_has_life_cost"]:
                score["passed"] = False
            results.append(
                {
                    "question": question,
                    "returncode": 0 if score["passed"] else 1,
                    "elapsed_sec": round(elapsed, 3),
                    "answer": answer,
                    "score": score,
                    "error": "",
                    "generator_calls": len(generator.calls),
                    "evidence_top5": [
                        {
                            "id": item.get("id"),
                            "stage_code": item.get("stage_code"),
                            "anchors": [
                                anchor
                                for anchor in ("岁陵", "危机", "性命", "代价", "不反", "太尉", "莫佚")
                                if anchor in str(item.get("evidence_chain_text") or item.get("clean_text") or "")
                            ],
                        }
                        for item in result.evidence[:5]
                    ],
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "question": question,
                    "returncode": 1,
                    "elapsed_sec": round(elapsed, 3),
                    "answer": "",
                    "score": score_answer("", question),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action-target regression for the 不反 hard case.")
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--fake-runtime", action="store_true", help="Run a CPU fake-generator full pipeline regression.")
    parser.add_argument("--fake-device", default="cpu")
    parser.add_argument("--fake-with-reranker", action="store_true")
    parser.add_argument("--python-bin", type=Path, default=Path("/home/zhb/miniconda3/envs/train/bin/python"))
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--per-question-timeout", type=int, default=900)
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = list(args.question or QUESTIONS)
    if args.limit > 0:
        questions = questions[: args.limit]
    if not args.fake_runtime and not args.pipeline.exists():
        raise SystemExit(f"Pipeline script not found: {args.pipeline}")

    if args.fake_runtime:
        results = run_fake_runtime_batch(args, questions)
        for index, result in enumerate(results, start=1):
            status = "PASS" if result["returncode"] == 0 and result["score"]["passed"] else "FAIL"
            print(
                f"[{index}/{len(results)}][{status}] {result['elapsed_sec']}s {result['question']} -> {result['answer'][:120]}",
                file=sys.stderr,
                flush=True,
            )
    elif args.no_batch:
        results = []
        args.timeout = args.per_question_timeout
        for index, question in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question}", file=sys.stderr, flush=True)
            result = run_question(args, question)
            results.append(result)
            status = "PASS" if result["returncode"] == 0 and result["score"]["passed"] else "FAIL"
            print(f"[{status}] {result['elapsed_sec']}s {result['answer'][:160]}", file=sys.stderr, flush=True)
    else:
        args.timeout = max(args.timeout, args.per_question_timeout * max(len(questions), 1))
        print(f"[batch] {len(questions)} questions timeout={args.timeout}s", file=sys.stderr, flush=True)
        results = run_batch(args, questions)
        for index, result in enumerate(results, start=1):
            status = "PASS" if result["returncode"] == 0 and result["score"]["passed"] else "FAIL"
            print(
                f"[{index}/{len(results)}][{status}] {result['elapsed_sec']}s {result['question']} -> {result['answer'][:120]}",
                file=sys.stderr,
                flush=True,
            )

    passed = sum(1 for item in results if item["returncode"] == 0 and item["score"]["passed"])
    summary = {
        "count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("count", "passed", "failed", "pass_rate")}, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
