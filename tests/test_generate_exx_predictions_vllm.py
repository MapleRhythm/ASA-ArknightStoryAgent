import importlib.util
import sys
from pathlib import Path


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
