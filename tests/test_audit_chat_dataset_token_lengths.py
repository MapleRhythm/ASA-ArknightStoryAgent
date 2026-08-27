import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_chat_dataset_token_lengths.py"
SPEC = importlib.util.spec_from_file_location("audit_chat_dataset_token_lengths", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_chat_messages_maps_sharegpt_roles_and_system() -> None:
    row = {
        "system": "系统",
        "conversations": [
            {"from": "human", "value": "问题"},
            {"from": "gpt", "value": "回答"},
        ],
    }
    assert MODULE.chat_messages(row) == [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]


def test_chat_messages_rejects_non_string_content() -> None:
    with pytest.raises(ValueError, match="invalid role or content"):
        MODULE.chat_messages(
            {"conversations": [{"from": "human", "value": {"not": "serialized"}}]}
        )


def test_percentile_uses_deterministic_lower_index() -> None:
    assert MODULE.percentile([1, 2, 3, 4], 0.5) == 2
    assert MODULE.percentile([], 0.9) == 0
