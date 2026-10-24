import copy

import pytest

from answer_semantics_content_v2 import PROTOCOL, validate
from judge_visible_answers import validate as legacy_validate


ROW = {"grounded": {"next_action": "answer_directly", "answer": "甲与乙有什么区别？"}}


def no_answer():
    return {
        "protocol": PROTOCOL, "answer_content": "none", "verdict": "no_answer",
        "supported_answer_spans": [], "unsupported_spans": [], "missing_requirements": ["双方的区别"],
        "context_sufficiency": "sufficient", "action_appropriateness": "appropriate",
        "reason": "只有问句，没有给出实质回答。",
    }


def test_legacy_bug_forbids_no_answer_when_action_is_answer_directly():
    value = no_answer()
    value["protocol"] = "asa_visible_answer_semantics_v1"
    del value["answer_content"], value["supported_answer_spans"]
    with pytest.raises(ValueError, match="answer/action"):
        legacy_validate(value, ROW)


def test_content_aware_rubric_allows_no_answer_despite_direct_action():
    assert validate(no_answer(), ROW)["verdict"] == "no_answer"


@pytest.mark.parametrize("label", ["supported_complete", "supported_partial"])
def test_no_substantive_content_cannot_be_counted_as_supported(label):
    value = no_answer()
    value["verdict"] = label
    with pytest.raises(ValueError, match="empty substantive"):
        validate(value, ROW)


def test_partial_requires_a_real_verbatim_answer_span():
    value = no_answer()
    value.update(verdict="supported_partial", answer_content="substantive")
    with pytest.raises(ValueError, match="requires"):
        validate(value, ROW)
    value["supported_answer_spans"] = ["甲是一名医生"]
    with pytest.raises(ValueError, match="verbatim"):
        validate(value, ROW)


def test_substantive_partial_control_is_retained():
    row = copy.deepcopy(ROW)
    row["grounded"]["answer"] = "甲负责读书。"
    value = no_answer()
    value.update(verdict="supported_partial", answer_content="substantive", supported_answer_spans=["甲负责读书"])
    assert validate(value, row)["verdict"] == "supported_partial"


def test_non_answer_action_cannot_be_counted_as_supported():
    row = copy.deepcopy(ROW)
    row["grounded"]["next_action"] = "retrieve_more"
    value = no_answer()
    assert validate(value, row)["verdict"] == "no_answer"
    value.update(verdict="supported_partial", answer_content="substantive", supported_answer_spans=["甲与乙"])
    with pytest.raises(ValueError, match="non-answer action"):
        validate(value, row)


def test_driver_installs_the_versioned_rubric_in_transport():
    from unittest.mock import patch
    import judge_visible_answers_content_v3 as wrapper
    import judge_visible_answers as transport
    import answer_semantics_content_v2 as content
    with patch.object(transport, "PROTOCOL", transport.PROTOCOL), \
            patch.object(transport, "messages", transport.messages), \
            patch.object(transport, "validate", transport.validate):
        wrapper.install_rubric()
        assert transport.PROTOCOL == PROTOCOL
        assert transport.messages is content.messages
        assert transport.validate(no_answer(), ROW)["verdict"] == "no_answer"
