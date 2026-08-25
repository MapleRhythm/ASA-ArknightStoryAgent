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


def test_api_cli_exposes_sidecar_index_controls() -> None:
    source = _source()
    assert '"--index-dir"' in source
    assert 'retrieval_cfg.get("index_dir")' in source
    assert 'retrieval_cfg.get("embedding_model_path")' in source
    assert '"documents_path": index_dir / "documents.jsonl"' in source
    assert '"sparse_index_path": index_dir / "sparse_index.pkl"' in source


def test_api_cli_exposes_independent_batch_and_query_cap() -> None:
    source = _source()
    for option in (
        "--questions-file",
        "--batch-output",
        "--max-retrieval-queries",
    ):
        assert option in source
    assert 'dialogue_context = "" if batch_questions is not None' in source
    assert '"stage_timings"' in source
    assert 'generator_cfg.get("task_extra_body")' in source
    assert "prompt_task_name(prompt)" in source


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
    assert "max_retrieval_queries" in keyword_names


def test_api_answer_prompts_prefer_direct_chunks_over_chain_metadata() -> None:
    source = _source()
    assert 'doc.get("clean_text") or doc.get("text") or item.get("evidence_chain_text")' in source


def test_api_respects_configured_json_output_ceiling() -> None:
    source = _source()
    assert "self.max_tokens or 0, 4096" not in source
