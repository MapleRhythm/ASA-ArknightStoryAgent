from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import io
import json
import os
from pathlib import Path


LOCAL_CANDIDATE = Path(__file__).resolve().parent / "remote_src_v23" / "run_cpu_inference.py"
RELEASE_CANDIDATE = Path(__file__).resolve().parents[1] / "scripts" / "run_cpu_inference.py"
SCRIPT = Path(
    os.environ.get(
        "ASA_RUN_CPU_INFERENCE_SCRIPT",
        str(LOCAL_CANDIDATE if LOCAL_CANDIDATE.exists() else RELEASE_CANDIDATE),
    )
)
SPEC = importlib.util.spec_from_file_location("release_run_cpu_inference_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclass
class Result:
    question: str
    answer: str
    retrieval_trace: list
    evidence: list


class FakePipeline:
    def run(self, question, dialogue_context, progress_callback):
        progress_callback("test")
        if question == "失败题":
            raise RuntimeError("synthetic failure")
        return Result(
            question=question,
            answer="成功答案",
            retrieval_trace=[],
            evidence=[],
        )


def test_batch_returns_failure_count_without_dropping_rows():
    output = io.StringIO()
    failures = MODULE.run_questions_batch(
        pipeline=FakePipeline(),
        questions=["成功题", "失败题"],
        dialogue_context="",
        output_handle=output,
        answer_only=False,
    )
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert failures == 1
    assert [row["question"] for row in rows] == ["成功题", "失败题"]
    assert rows[0]["answer"] == "成功答案"
    assert rows[0]["error"] == ""
    assert rows[1]["answer"] == ""
    assert rows[1]["error"] == "RuntimeError: synthetic failure"
