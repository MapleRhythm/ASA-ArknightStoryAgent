from dataclasses import dataclass, asdict
import pytest

from entity_projection import project_model_entities


@dataclass
class Hypothesis:
    entities: list
    keywords: list
    question: str = "测试问题"
    dialogue_context: str = ""
    intent: str = "plot_fact"


def test_projection_preserves_all_other_fields_and_input():
    source = Hypothesis(["小大厨", "煌", "挂面", "出煌", "节不足后"], ["出煌", "挂面", "别名"])
    original = asdict(source)
    projected = project_model_entities(source, ["小大厨", "煌", "挂面"])
    assert projected.entities == ["小大厨", "煌", "挂面"]
    assert asdict(source) == original
    actual = asdict(projected)
    actual["entities"] = original["entities"]
    assert actual == original


def test_no_added_entities_is_identical():
    source = Hypothesis(["初雪", "父母"], ["列车"])
    assert asdict(project_model_entities(source, ["初雪", "父母"])) == asdict(source)


def test_model_term_is_not_removed_even_when_it_is_bad():
    source = Hypothesis(["出煌", "挂面", "节不足后"], ["原因"])
    assert project_model_entities(source, ["出煌", "挂面"]).entities == ["出煌", "挂面"]


def test_recall_loss_counterexample_is_preserved_not_hidden():
    # Synthetic risk witness, NOT gold or a new model output:
    # if a valid payload omitted 初雪, final projection loses its heuristic recovery.
    source = Hypothesis(["父母", "初雪"], ["父母", "初雪"], question="初雪的父母发生了什么")
    result = project_model_entities(source, ["父母"])
    assert "初雪" not in result.entities
    assert "初雪" in result.keywords


@pytest.mark.parametrize("entities", [[], ["初雪", "初雪"], [""], [None], list("abcdefghijklmn")])
def test_invalid_input_refused(entities):
    with pytest.raises(ValueError):
        project_model_entities(Hypothesis(["初雪"], []), entities)


def test_prefix_mismatch_refused():
    with pytest.raises(ValueError):
        project_model_entities(Hypothesis(["父母", "初雪"], []), ["初雪"])
