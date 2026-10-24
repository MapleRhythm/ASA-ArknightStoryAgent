import pytest

from selector_protocol import blocks, reader_prompt, strict_object, validate_selection


EVIDENCE = "[E1] 甲迁徙，是因风沙。\n[E2] 乙迁徙，是因征兆。\n[E4] 乙不是甲。"


def selection(ids):
    return {"evidence_ids": ids, "sufficiency": "sufficient", "reason": "NOT_SENT_TO_READER"}


def test_full_block_preservation_and_no_reason_leakage():
    selected = validate_selection(selection(["E2", "E4"]), EVIDENCE)
    assert selected == "[E2] 乙迁徙，是因征兆。\n[E4] 乙不是甲。"
    prompt = reader_prompt("prefix\nevidence_brief:\n\noutput_schema:suffix", selected)
    assert "NOT_SENT_TO_READER" not in prompt
    assert "E1" not in prompt
    assert selected in prompt


@pytest.mark.parametrize("ids", [["E99"], ["E2", "E2"], ["E4", "E2"], []])
def test_bad_selections_fail_without_silent_repair(ids):
    with pytest.raises(ValueError):
        validate_selection(selection(ids), EVIDENCE)


def test_same_selection_preserves_every_byte():
    assert validate_selection(selection(["E1", "E2", "E4"]), EVIDENCE) == EVIDENCE


def test_split_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        blocks("[E1] 甲。\n[E1] 乙。")


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError):
        strict_object('{"evidence_ids":["E2"],"evidence_ids":["E1"]}')


def test_reader_template_drift_is_rejected():
    with pytest.raises(ValueError):
        reader_prompt("unexpected prompt", EVIDENCE)
