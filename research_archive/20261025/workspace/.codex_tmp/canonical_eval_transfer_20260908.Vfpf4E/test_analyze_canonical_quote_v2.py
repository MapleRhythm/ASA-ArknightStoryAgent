import hashlib
import json

import pytest

from analyze_canonical_quote_v2 import matched_rows, round_state, summarize


def test_last_round_is_not_hardcoded():
    assert round_state({"current_round": 3, "max_retrieval_rounds": 3}) == (3, 3)
    assert round_state({}) == (1, 2)


@pytest.mark.parametrize("row", [
    {"current_round": 0}, {"current_round": 3, "max_retrieval_rounds": 2},
    {"current_round": True}, {"max_retrieval_rounds": "2"},
])
def test_rejects_invalid_round(row):
    with pytest.raises(ValueError):
        round_state(row)


def test_matching_requires_exact_prompt_and_input_hash():
    row = {"id": "a", "runtime_prompt": "frozen"}
    pred = {"id": "a", "actual_prompt": "frozen", "input_row_sha256": hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    assert list(matched_rows({"rows": [row]}, [pred])) == [(row, pred)]
    with pytest.raises(ValueError):
        list(matched_rows({"rows": [row]}, [pred, pred]))
    pred["actual_prompt"] = "modified"
    with pytest.raises(ValueError, match="actual input"):
        list(matched_rows({"rows": [row]}, [pred]))


def test_single_renderer_summary_retains_failed_row():
    result = summarize([{
        "renderer": "after", "normalized_strict_json": False,
        "status": "parse_or_validation_error", "finish_reason": "length",
        "actual_total_tokens": 12000, "current_round": 2, "max_retrieval_rounds": 2,
        "generation_seconds": 2.0,
    }])
    assert set(result) == {"after"}
    assert result["after"]["rows"] == 1
    assert result["after"]["grounded_actions"] == {"error": 1}
    assert result["after"]["length_outputs"] == 1
