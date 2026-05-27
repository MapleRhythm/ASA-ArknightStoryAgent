#!/usr/bin/env python3
from __future__ import annotations

from goldenglow.inference.cpu_pipeline import (
    ConclusionResult,
    HypothesisDocument,
    _answer_misses_action_target,
    validate_conclusion_grounding,
)


QUESTION_WHY = "真龙为什么要启动不反？"
QUESTION_SOLVE = "真龙动用不反是为了解决什么？"
EVIDENCE = [
    {
        "evidence_chain_text": (
            "太尉：陛下要以万金之躯启动“不反”，解决当下岁陵里的那场危机？ "
            "莫佚：凡是想要全力启用源石的，都得以真龙本人的性命为代价......故名曰“不反”。"
        )
    }
]


def make_hypothesis(question: str) -> HypothesisDocument:
    return HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["真龙", "不反", "岁陵"],
        keywords=["启动", "不反", "岁陵", "危机", "代价"],
        expected_answer_type="原因/目的",
        dialogue_context="",
    )


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_false(value: bool, message: str) -> None:
    if value:
        raise AssertionError(message)


def main() -> int:
    why_hypothesis = make_hypothesis(QUESTION_WHY)
    assert_true(
        _answer_misses_action_target(
            "真龙启动不反是为了解决岁陵危机。",
            ["不反"],
            question=QUESTION_WHY,
            hypothesis=why_hypothesis,
        ),
        "why-question answer missing life-cost anchor should be corrected",
    )
    assert_false(
        _answer_misses_action_target(
            "真龙启动不反是为了解决岁陵危机，代价是真龙性命。",
            ["不反"],
            question=QUESTION_WHY,
            hypothesis=why_hypothesis,
        ),
        "why-question answer with crisis and cost anchors should pass",
    )
    assert_true(
        _answer_misses_action_target(
            "真龙启动不反是为了证明自己的能力，推动大炎发展。",
            ["不反"],
            question=QUESTION_WHY,
            hypothesis=why_hypothesis,
        ),
        "background-motivation drift should be corrected",
    )

    corrected = validate_conclusion_grounding(
        question=QUESTION_WHY,
        hypothesis=why_hypothesis,
        evidence=EVIDENCE,
        conclusion=ConclusionResult(
            next_action="answer_directly",
            answer="真龙启动不反是为了解决岁陵危机。",
            missing_slots=[],
            clarification_question="",
            follow_up_hypothesis=None,
        ),
        max_round_reached=True,
        mode="weak",
    )
    assert_true("岁陵里的那场危机" in corrected.answer, "corrected answer should preserve the specific crisis")
    assert_true("真龙本人的性命" in corrected.answer, "corrected answer should include life-cost anchor")
    assert_true("证明自己的能力" not in corrected.answer, "corrected answer should not keep drift explanation")

    solve_hypothesis = make_hypothesis(QUESTION_SOLVE)
    assert_false(
        _answer_misses_action_target(
            "真龙动用不反是为了解决岁陵危机。",
            ["不反"],
            question=QUESTION_SOLVE,
            hypothesis=solve_hypothesis,
        ),
        "solve-object question should not require the life-cost anchor",
    )
    print("action-target runtime correction tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
