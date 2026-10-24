import pytest

from thinking_protocol import answer_after_thinking


def test_no_think_preserves_json_verbatim():
    value = ' \n{"next_action":"abstain"}\n'
    assert answer_after_thinking(value, False) == (value, 0)


def test_completed_thinking_extracts_only_answer():
    assert answer_after_thinking('内部推理\n</think>\n\n{"x":1}', True) == ('{"x":1}', 5)


@pytest.mark.parametrize("raw,thinking", [
    ("推理未完成", True),
    ("推理</think> ", True),
    ("甲</think>乙</think>{}", True),
    ("甲</think><think>{}", True),
    ("<think>不应出现</think>{}", False),
])
def test_incomplete_or_unexpected_reasoning_is_not_salvaged(raw, thinking):
    with pytest.raises(ValueError):
        answer_after_thinking(raw, thinking)
