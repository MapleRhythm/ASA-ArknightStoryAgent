import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_exx_predictions_vllm.py"
SPEC = importlib.util.spec_from_file_location("generate_exx_predictions_vllm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_render_prompt_matches_qwen3_nothink_prefix() -> None:
    row = {
        "id": "x",
        "system": "只输出 JSON。",
        "conversations": [{"from": "human", "value": "问题与证据"}],
    }

    assert MODULE.render_prompt(row) == (
        "<|im_start|>system\n只输出 JSON。<|im_end|>\n"
        "<|im_start|>user\n问题与证据<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_reader_specs_supports_base_and_adapters() -> None:
    adapters = [("clean_sft", Path("/models/clean"))]

    assert MODULE.reader_specs("qwen35_4b_base", adapters) == [
        ("qwen35_4b_base", None),
        ("clean_sft", Path("/models/clean")),
    ]


def test_reader_specs_rejects_duplicate_or_unsafe_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        MODULE.reader_specs(
            "reader",
            [("reader", Path("/models/adapter"))],
        )
    with pytest.raises(Exception, match="reader name"):
        MODULE.validate_reader_name("../escape")


def test_prediction_row_preserves_diagnostic_metadata() -> None:
    row = {
        "id": "q1::minimal_sufficient",
        "task_type": "grounded_action_generation",
        "system": "只输出 JSON。",
        "conversations": [{"from": "human", "value": "问题与证据"}],
        "diagnostic": {
            "source_id": "q1",
            "condition": "minimal_sufficient",
        },
    }

    result = MODULE.prediction_row(
        row=row,
        text='{"next_action":"abstain","reason":"不足"}',
        prompt_token_count=20,
        generated_token_count=10,
        finish_reason="stop",
    )

    assert result["diagnostic"] == row["diagnostic"]
    assert result["hit_max_new_tokens"] is False
