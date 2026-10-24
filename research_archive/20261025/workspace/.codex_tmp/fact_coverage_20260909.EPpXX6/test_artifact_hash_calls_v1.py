"""Regression for Path-only artifact hash helpers at workflow call sites."""
import ast
from pathlib import Path
import pytest

from build_fact_coverage_pairs_v1 import sha


@pytest.mark.parametrize("filename", [
    "build_reasoning_context_ablation_v1.py",
    "run_reasoning_context_ablation_v1.py",
    "check_native8b_v1.py",
])
def test_self_hash_call_passes_a_path(filename):
    path = Path(__file__).with_name(filename)
    source = path.read_text()
    tree = ast.parse(source)
    self_hash_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sha"
        and any(isinstance(sub, ast.Name) and sub.id == "__file__" for sub in ast.walk(node))
    ]
    assert self_hash_calls
    for call in self_hash_calls:
        expression = ast.Expression(call.args[0])
        value = eval(compile(expression, str(path), "eval"), {"Path": Path, "__file__": str(path)})
        assert isinstance(value, Path) and len(sha(value)) == 64
