from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rejudge_binding_hardneg_full_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rejudge_binding_hardneg_full_evidence", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prompt_is_blind_and_contains_full_candidate() -> None:
    messages = MODULE.prompt_messages(
        "主张",
        [("E1", "hard_negative", "完整证据" * 800)],
    )
    content = messages[-1]["content"]
    assert "完整证据" in content
    assert "glm_verdict" not in content
    assert "原 verdict" not in content


def test_fingerprint_is_stable() -> None:
    first = MODULE.evidence_fingerprint("row", 0, "主张", "E1", "证据")
    second = MODULE.evidence_fingerprint("row", 0, "主张", "E1", "证据")
    assert first == second
