import pytest

from asa_arknight_story_agent.inference.grounding.discourse_support import discourse_support
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import validate_evidence_id_grounding
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult


def test_reported_mental_predicate_requires_visible_predicate():
    assert discourse_support("她想知道", "", "", ["甲：你想知道原因。"]) == "reported_epistemic_predicate"
    assert discourse_support("她想知道", "", "", ["甲：明天见。"]) is None
    assert discourse_support("她杀死他", "", "", ["甲杀死乙。"]) is None


def test_prerequisite_wording_requires_actual_prerequisite():
    assert discourse_support("前提条件", "", "", ["前提是遵守约定。"]) == "prerequisite_wording"
    assert discourse_support("前提条件", "", "", ["没有其他消息。"]) is None


def test_reporter_requires_literal_source_speaker_and_does_not_cover_proposition():
    assert discourse_support("研究员表示只", "", "", ["研究员：只要遵守约定就行。"])
    assert discourse_support("研究员提到她", "", "", ["研究员：这是我的记录。"])
    assert discourse_support("主管表示只", "", "", ["研究员：只要遵守约定就行。"]) is None
    assert discourse_support("研究员表示杀人", "", "", ["研究员：只要遵守约定就行。"]) is None


def test_temporal_scope_does_not_whitelist_arbitrary_question_content():
    assert discourse_support("与甲城交战时", "", "与甲城交战时为什么不同？", [])
    assert discourse_support("与甲城交战时", "", "乙城发生了什么？", []) is None
    assert discourse_support("与甲城交战时", "", "与甲城没有交战时为什么不同？", []) is None
    assert discourse_support("母亲是凶手", "", "母亲是凶手吗？", []) is None


@pytest.mark.parametrize("body,expected", [
    ("工人每天工作时间不能超过十个小时", True),
    ("工人每天工作时间不能超过十二个小时", False),
    ("工人每天工作时间能超过十个小时", False),
    ("工人每天工作时间不能超过十个小时并且没有工资", False),
])
def test_rule_frame_requires_entire_literal_body(body, expected):
    text = "法案上写，工人每天工作时间不能超过十个小时。"
    claim = "改善工人待遇的新法案规定" + body
    assert bool(discourse_support(
        "新法案规定工", claim, "改善工人待遇的新法案具体规定了哪些内容？", [text],
    )) is expected


def test_rule_frame_does_not_borrow_support_from_other_citation_or_question():
    assert discourse_support(
        "新法案规定工", "新法案规定工人每天工作时间不能超过十个小时",
        "新法案规定什么？", ["工人每天工作时间不能超过十个小时。"],
    ) is None
    assert discourse_support(
        "新法案规定工", "新法案规定工人每天工作时间不能超过十个小时",
        "新法案规定什么？", ["法案还没公布。", "工人每天工作时间不能超过十个小时。"],
    ) is None


@pytest.mark.parametrize("claim", [
    "研究员表示她任命甲某为整合运动领袖。",
    "研究员提到她是甲某的母亲。",
])
def test_reporter_bridge_does_not_override_substantive_relation_guard(claim):
    conclusion = ConclusionResult(
        "answer_directly", claim, [], "", None,
        supported_facts=[{"fact": claim, "evidence_refs": [{"evidence_id": "E1"}]}],
    )
    issues, _ = validate_evidence_id_grounding(
        conclusion=conclusion, evidence=[], question="研究员说了什么？",
        evidence_prompt_text="[E1]\n研究员：甲某提交了报告。",
    )
    assert issues


@pytest.mark.parametrize("evidence,expected", [
    ("队长：只要我的队友在身边，就不会出全力。现在不同，他们都死了。", True),
    ("队长：只要我的队友在身边，就不会出全力。现在不同，他们都死去。", True),
    ("队长：只要我的队友在身边，就不会出全力。现在不同，他们都没有死。", False),
    ("队长：我的队友在这里。工人来了，他们都死了。", False),
    ("队长：我的队友在这里。记者：他们都死了。", False),
    ("队长：我的队友在这里。他们有人死了。", False),
    ("队长：我的队友在这里。他们都回来了。", False),
])
def test_plural_antecedent_requires_same_turn_unambiguous_group_and_predicate(evidence, expected):
    assert bool(discourse_support(
        "队友都", "队长的队友都已死去，身边没有人。", "为什么不同？", [evidence],
    )) is expected
