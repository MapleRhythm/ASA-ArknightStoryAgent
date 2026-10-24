"""Exercise the real argument parser and main in a child process.

Only expensive model/retriever/index dependencies are faked; the batch loop,
JSONL writing and SystemExit handling are production code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(os.environ.get(
    "ASA_RUN_CPU_INFERENCE_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "scripts" / "run_cpu_inference.py"),
))
CHILD = r"""
from dataclasses import dataclass
import importlib.util
import sys
import types

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
spec = importlib.util.spec_from_file_location("asa_batch_cli_under_test", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ensure_index = lambda args: None
module.load_runtime_config = lambda path: {}

@dataclass
class Result:
    question: str
    answer: str
    retrieval_trace: list
    evidence: list

class Pipeline:
    def __init__(self, **kwargs):
        assert kwargs["query_config"].minirag_top_k == 0
        assert kwargs["query_config"].minirag_weight == 0.0
    def run(self, question, dialogue_context, progress_callback):
        progress_callback("synthetic")
        if question == "失败题":
            raise RuntimeError("synthetic failure")
        return Result(question, "成功答案", [], [])

class Runner:
    def __init__(self, **kwargs):
        pass

class Retriever:
    @classmethod
    def from_paths(cls, **kwargs):
        assert kwargs["minirag_index_path"] is None
        return cls()

fake_inference = types.ModuleType("asa_arknight_story_agent.inference")
fake_inference.CPUInferencePipeline = Pipeline
fake_inference.LlamaCppRunner = Runner
fake_inference.VllmRunner = Runner
sys.modules[fake_inference.__name__] = fake_inference
fake_hybrid = types.ModuleType("asa_arknight_story_agent.retrieval.hybrid")
fake_hybrid.ArknightsHybridRetriever = Retriever
sys.modules[fake_hybrid.__name__] = fake_hybrid
module.main()
"""


@pytest.mark.parametrize("answer_only", [False, True])
@pytest.mark.parametrize(
    ("questions", "expected_code"),
    [
        (["成功题", "失败题", "后续成功题"], 1),
        (["成功题", "后续成功题"], 0),
        (["失败题"], 1),
    ],
)
def test_cli_exit_and_complete_jsonl(tmp_path, answer_only, questions, expected_code):
    source = tmp_path / "questions.txt"
    output = tmp_path / "results.jsonl"
    source.write_text("\n".join(questions) + "\n", encoding="utf-8")
    command = [
        sys.executable, "-c", CHILD, str(SCRIPT),
        "--questions-file", str(source), "--batch-output", str(output),
        "--no-reranker", "--disable-lora", "--disable-minirag", "--device", "cpu",
    ]
    if answer_only:
        command.append("--answer-only")
    process = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert process.returncode == expected_code, process.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["question"] for row in rows] == questions
    for row in rows:
        if row["question"] == "失败题":
            assert row["error"] == "RuntimeError: synthetic failure"
            assert row["answer"] == ""
        else:
            assert row["error"] == ""
            assert row["answer"] == "成功答案"
