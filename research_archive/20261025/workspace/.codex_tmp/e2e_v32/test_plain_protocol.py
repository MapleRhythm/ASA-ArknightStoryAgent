from copy import deepcopy
import json
import pytest

from plain_protocol import messages_for, verify_messages


def example():
    return {
        "question": "人物甲为何离开？",
        "hypothesis": {"entities": ["人物甲"], "keywords": ["别名甲"]},
        "evidence_brief": "[E1] [CHAIN_MEMBER_1] 人物乙：我走了。\n[E2] 人物甲：我留下。",
        "review": "This annotation must not reach the model.",
        "conditions": {"think": {"raw_output": "Do not leak previous generations."}},
    }


def test_exact_payload_without_outputs():
    row = example()
    messages = messages_for(row)
    verify_messages(messages, row)
    assert set(json.loads(messages[1]["content"])) == {"question", "hypothesis", "evidence_brief"}
    assert "Do not leak" not in str(messages)
    assert "This annotation" not in str(messages)


def test_no_mutation():
    row = example()
    before = deepcopy(row)
    messages_for(row)
    assert row == before


@pytest.mark.parametrize("key", ["question", "hypothesis", "evidence_brief"])
def test_source_change_detected(key):
    row = example()
    messages = messages_for(row)
    row[key] = "changed"
    with pytest.raises(AssertionError):
        verify_messages(messages, row)


def test_evidence_whitespace_and_owner_preserved():
    row = example()
    row["evidence_brief"] += "\n  [E3] 「名字：引文」\t结尾  "
    actual = json.loads(messages_for(row)[1]["content"])["evidence_brief"]
    assert actual == row["evidence_brief"]


def test_reject_added_identity_hint():
    row = example()
    messages = messages_for(row)
    messages[0]["content"] += "人物甲并非人物乙。"
    with pytest.raises(AssertionError):
        verify_messages(messages, row)
