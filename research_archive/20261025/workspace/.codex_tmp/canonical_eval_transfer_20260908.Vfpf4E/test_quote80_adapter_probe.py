import copy

import pytest

from run_quote80_adapter_probe_v1 import assert_control_input, text_sha, token_sha


def row():
    return {
        "runtime_prompt": "full prompt", "runtime_prompt_sha256": text_sha("full prompt"),
        "evidence_brief": "full evidence", "evidence_sha256": text_sha("full evidence"),
        "control_prompt_token_ids_sha256": token_sha([1, 2, 3]), "control_prompt_token_count": 3,
    }


def test_control_tokens_and_full_text_are_preserved():
    value = row()
    before = copy.deepcopy(value)
    assert_control_input(value, [1, 2, 3])
    assert value == before


def test_changed_tokenization_fails_even_if_token_count_matches():
    with pytest.raises(ValueError, match="actual input tokens"):
        assert_control_input(row(), [1, 2, 4])


def test_changed_evidence_or_prompt_is_rejected():
    for key in ("runtime_prompt", "evidence_brief"):
        value = row()
        value[key] += " changed"
        with pytest.raises(ValueError, match="Frozen"):
            assert_control_input(value, [1, 2, 3])


def test_control_token_count_mismatch_is_rejected():
    value = row()
    value["control_prompt_token_count"] = 2
    with pytest.raises(ValueError, match="count"):
        assert_control_input(value, [1, 2, 3])
