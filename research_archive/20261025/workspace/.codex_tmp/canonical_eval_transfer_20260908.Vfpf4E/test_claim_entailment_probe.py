import copy

import pytest

from probe_claim_entailment_v1 import VERSION, messages, validate


CASE = {"question": "谁说了什么？", "claim": "甲转述了乙的话。", "cited_blocks": {"E1": "甲说乙昨天说了这句话。"},
        "gold": "must not be sent", "prior_answer": "must not be sent"}


def verdict():
    return {"version": VERSION, "claim": CASE["claim"], "relation": "entailed",
            "evidence_refs": [{"evidence_id": "E1", "quote": "甲说乙昨天说了这句话。"}], "reason": "角色与转述对象一致。"}


def test_only_literal_claim_question_and_cited_blocks_are_sent():
    import json
    payload = json.loads(messages(CASE)[1]["content"])
    assert set(payload) == {"question", "claim", "cited_blocks"}
    assert "must not be sent" not in messages(CASE)[1]["content"]


def test_preserves_literal_claim_and_source():
    original = copy.deepcopy(CASE)
    assert validate(verdict(), CASE)["relation"] == "entailed"
    assert CASE == original


def test_rewritten_claim_is_rejected_even_if_verdict_is_positive():
    value = verdict()
    value["claim"] = "乙转述了甲的话。"
    with pytest.raises(ValueError, match="rewritten"):
        validate(value, CASE)


@pytest.mark.parametrize("bad_ref", [
    {"evidence_id": "E99", "quote": "甲说乙昨天说了这句话。"},
    {"evidence_id": "E1", "quote": "甲自己说了这句话。"},
    {"evidence_id": "E1", "quote": ""},
])
def test_fake_reference_or_altered_quote_is_rejected(bad_ref):
    value = verdict()
    value["evidence_refs"] = [bad_ref]
    with pytest.raises(ValueError, match="verbatim"):
        validate(value, CASE)


def test_established_verdict_requires_a_reference():
    value = verdict()
    value["evidence_refs"] = []
    with pytest.raises(ValueError, match="requires"):
        validate(value, CASE)
