from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest


EXPECTED = {
    "enable_minirag": False,
    "minirag_top_k": 0,
    "minirag_weight": 0.0,
    "minirag_mode_weights": {},
    "minirag_chapter_isolation": False,
    "minirag_auto_second_retrieval": False,
    "minirag_scope_seed_top_k": 0,
    "minirag_expansion_query_top_k": 0,
}


def paths_from_env(name: str) -> list[Path]:
    return [Path(value) for value in os.environ[name].split(os.pathsep) if value]


@pytest.mark.parametrize("path", paths_from_env("ASA_NO_MINIRAG_RUNNERS"), ids=str)
def test_runner_has_fail_closed_contract(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "require_minirag_disabled"
    ]
    assert len(functions) == 1, path
    namespace: dict = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    require_disabled = namespace["require_minirag_disabled"]
    assert require_disabled(False) == EXPECTED
    with pytest.raises(SystemExit, match="disabled"):
        require_disabled(True)

    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main) or ""
    assert "require_minirag_disabled(requested_minirag)" in main_source, path
    assert "if enable_minirag:" not in main_source, path


@pytest.mark.parametrize("path", paths_from_env("ASA_NO_MINIRAG_CONFIGS"), ids=str)
def test_runtime_config_is_graph_free(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        pytest.skip("not a retrieval runtime config")
    for key, value in EXPECTED.items():
        assert retrieval.get(key) == value, (path, key, retrieval.get(key), value)
