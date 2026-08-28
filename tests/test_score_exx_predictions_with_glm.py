import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_exx_predictions_with_glm.py"
SPEC = importlib.util.spec_from_file_location("score_exx_predictions_with_glm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extracts_prompt_and_completion_from_prediction() -> None:
    row = {
        "id": "x",
        "conversations": [
            {"from": "human", "value": "prompt"},
            {"from": "gpt", "value": "completion"},
        ],
    }

    assert MODULE.prompt_value(row) == "prompt"
    assert MODULE.completion_value(row) == "completion"
