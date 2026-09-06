from asa_arknight_story_agent.inference.pipeline.trace import conclusion_to_tool_call
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def hypothesis() -> HypothesisDocument:
    return HypothesisDocument(
        question="问题", intent="plot_fact", query_type="fact", entities=["甲"],
        keywords=["乙"], expected_answer_type="事实",
    )


def test_answer_is_mapped_to_internal_tool_call_without_extra_text() -> None:
    result = conclusion_to_tool_call(
        ConclusionResult(
            next_action="answer_directly", answer="", missing_slots=[],
            clarification_question="", follow_up_hypothesis=None,
            supported_facts=[{"fact": "甲完成任务", "evidence_ids": ["E1"]}],
        )
    )
    assert result == {
        "protocol": "asa_round_tool_v1",
        "name": "answer",
        "arguments": {
            "supported_facts": [{"fact": "甲完成任务", "evidence_ids": ["E1"]}],
            "inferred_facts": [],
        },
    }


def test_retrieve_more_preserves_structured_follow_up_hypothesis() -> None:
    result = conclusion_to_tool_call(
        ConclusionResult(
            next_action="retrieve_more", answer="", missing_slots=["缺少原因"],
            clarification_question="", follow_up_hypothesis=hypothesis(),
        )
    )
    assert result["name"] == "search_more"
    assert result["arguments"]["missing_requirements"] == ["缺少原因"]
    assert result["arguments"]["follow_up_hypothesis"]["entities"] == ["甲"]


def test_unknown_terminal_action_falls_back_to_abstain() -> None:
    result = conclusion_to_tool_call(
        ConclusionResult(
            next_action="abstain", answer="证据不足", missing_slots=[],
            clarification_question="", follow_up_hypothesis=None,
        )
    )
    assert result["name"] == "abstain"
    assert result["arguments"]["reason"] == "证据不足"
