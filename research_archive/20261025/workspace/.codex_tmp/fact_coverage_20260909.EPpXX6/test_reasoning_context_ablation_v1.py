from copy import deepcopy
import json
import pytest

from build_reasoning_context_ablation_v1 import remove_reasoning_context


def row():
    hyp = {"question": "q", "keywords": ["not evidence"]}
    evidence = "[E1] 完整证据\nhypothesis: 这是原文，不得删除。\nminirag_hints: 原文内同名字段。"
    return {
        "question": "q", "hypothesis": hyp, "evidence_brief": evidence,
        "current_round": 1, "max_retrieval_rounds": 2,
        "user": "\n".join([
            "task: conclusion_generation", "question: q",
            "hypothesis: " + json.dumps(hyp, ensure_ascii=False),
            "round: 1/2", "evidence_brief:", evidence,
            "minirag_hints: entities=x | relations=x-相关-y | neighbors=z",
            "output_schema: grounded_action_v1", "rules: original",
        ]),
    }


def test_exact_evidence_including_reserved_looking_lines_preserved():
    original = row()
    before = deepcopy(original)
    output, removed = remove_reasoning_context(original)
    assert original["evidence_brief"] in output and output.endswith("rules: original")
    assert "not evidence" not in output and "x-相关-y" not in output
    assert original == before and removed["hints_line"].startswith("minirag_hints:")


@pytest.mark.parametrize("old,new", [
    ("question: q", "question: changed"),
    ("round: 1/2", "round: 2/2"),
    ("task: conclusion_generation", "task: other"),
    ("output_schema: grounded_action_v1", "schema: other"),
])
def test_unexpected_structure_rejected(old, new):
    original = row()
    original["user"] = original["user"].replace(old, new)
    with pytest.raises(AssertionError):
        remove_reasoning_context(original)


def test_hypothesis_mismatch_rejected():
    original = row()
    original["hypothesis"]["keywords"] = ["different"]
    with pytest.raises(AssertionError):
        remove_reasoning_context(original)


def test_evidence_mismatch_rejected():
    original = row()
    original["evidence_brief"] = "different"
    with pytest.raises(AssertionError):
        remove_reasoning_context(original)
