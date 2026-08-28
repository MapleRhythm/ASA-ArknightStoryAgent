import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_exx_predictions_transformers.py"
SPEC = importlib.util.spec_from_file_location("generate_exx_predictions_transformers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_chat_messages_preserves_system_and_full_user_prompt() -> None:
    row = {
        "system": "只输出 JSON。",
        "conversations": [
            {"from": "human", "value": "问题\n[E1]\n完整证据"},
            {"from": "gpt", "value": "ignored gold"},
        ],
    }

    assert MODULE.chat_messages(row) == [
        {"role": "system", "content": "只输出 JSON。"},
        {"role": "user", "content": "问题\n[E1]\n完整证据"},
    ]
