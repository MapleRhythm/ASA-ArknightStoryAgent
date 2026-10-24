import pytest

from audit_paired_retry_stream_v1 import check_prefix, completed_journal_rows


def test_partial_last_line_waits_instead_of_dropping_a_record():
    assert completed_journal_rows('{"id":1}\n{"id":') == [{"id": 1}]
    assert completed_journal_rows('{"id":1}\n{"id":2}\n') == [{"id": 1}, {"id": 2}]


def test_malformed_complete_line_is_not_silently_skipped():
    with pytest.raises(ValueError):
        completed_journal_rows('{"id":broken}\n')


def test_prefix_keeps_exact_input_identity():
    matrix = [{"id": "a", "runtime_prompt": "full"}, {"id": "b", "runtime_prompt": "more"}]
    check_prefix([{"id": "a", "actual_prompt": "full"}], matrix)
    with pytest.raises(ValueError):
        check_prefix([{"id": "b", "actual_prompt": "more"}], matrix)
    with pytest.raises(ValueError):
        check_prefix([{"id": "a", "actual_prompt": "ful"}], matrix)
