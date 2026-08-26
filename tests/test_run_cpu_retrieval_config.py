from __future__ import annotations

import ast
from pathlib import Path


CPU_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cpu_inference.py"


def _source() -> str:
    return CPU_SCRIPT.read_text(encoding="utf-8")


def test_cpu_cli_exposes_query_cap_and_full_prompt_evidence() -> None:
    source = _source()
    assert '"--max-retrieval-queries"' in source
    assert '"--require-full-prompt-evidence"' in source
    assert '"prompt_evidence_require_full_documents"' in source


def test_cpu_pipeline_receives_query_cap_and_full_prompt_evidence() -> None:
    tree = ast.parse(_source(), filename=str(CPU_SCRIPT))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CPUInferencePipeline"
    ]
    assert calls
    keyword_names = {keyword.arg for keyword in calls[-1].keywords}
    assert "max_retrieval_queries" in keyword_names
    assert "prompt_evidence_require_full_documents" in keyword_names
