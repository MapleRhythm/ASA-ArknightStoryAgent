from __future__ import annotations

import ast
from pathlib import Path


API_SCRIPT = Path(__file__).resolve().parents[1] / "api-mode" / "run_api_inference.py"


def _source() -> str:
    return API_SCRIPT.read_text(encoding="utf-8")


def test_api_cli_exposes_hybrid_fusion_controls() -> None:
    source = _source()
    for option in (
        "--dense-weight",
        "--sparse-weight",
        "--dense-min-quota",
        "--sparse-min-quota",
    ):
        assert option in source


def test_api_query_config_receives_hybrid_fusion_controls() -> None:
    tree = ast.parse(_source(), filename=str(API_SCRIPT))
    query_config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QueryConfig"
    ]
    assert query_config_calls
    keyword_names = {keyword.arg for keyword in query_config_calls[-1].keywords}
    assert {
        "dense_weight",
        "sparse_weight",
        "dense_min_quota",
        "sparse_min_quota",
    } <= keyword_names
