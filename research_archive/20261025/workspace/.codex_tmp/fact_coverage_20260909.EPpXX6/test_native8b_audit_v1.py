import pytest
from audit_native8b_v1 import split_requests


def messages(case):
    return [{"role": "user", "content": case["text"]}]


def test_exact_success_reused():
    case = {"id": "same", "text": "original input"}
    result = {"id": "same", "status": "ok"}
    assert split_requests([case], [case], [result], messages) == ([result], [])


def test_exact_failure_reused_without_retry():
    case = {"id": "same", "text": "original input"}
    result = {"id": "same", "status": "error"}
    assert split_requests([case], [case], [result], messages) == ([result], [])


def test_unknown_input_new():
    case = {"id": "new", "text": "new"}
    assert split_requests([case], [], [], messages) == ([], [case])


def test_different_content_cannot_reuse_id():
    with pytest.raises(AssertionError, match="different provider input"):
        split_requests([{"id": "same", "text": "a"}], [{"id": "same", "text": "b"}],
                       [{"id": "same", "status": "ok"}], messages)


def test_duplicate_cases_rejected():
    case = {"id": "same", "text": "a"}
    with pytest.raises(AssertionError):
        split_requests([case, case], [], [], messages)
