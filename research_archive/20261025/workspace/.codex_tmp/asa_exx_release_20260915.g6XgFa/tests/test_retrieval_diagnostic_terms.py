"""Do not propagate internal grounding diagnostics as story retrieval terms."""
import copy
from dataclasses import asdict

import pytest

from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ModelOutputError
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import (
    build_heuristic_follow_up_hypothesis, merge_hypotheses,
)
from asa_arknight_story_agent.inference.planning.follow_up_query_generation import build_follow_up_hypothesis_queries
from asa_arknight_story_agent.inference.retrieval.query_rendering import build_retrieval_query


QUESTION = "斯第奇为什么离开际崖城？"
NOISE = [
    "answer_directly", "缺少可校验", "quote", "支撑", "missing_supported_facts",
    "supported_fact_1_ref_1_quote_not_found", "inferred_fact_2_missing_evidence_refs",
    "answer_quote_total_over_400:423",
]


def hypothesis(*, polluted=True, question=QUESTION):
    noise = NOISE if polluted else []
    return HypothesisDocument(
        question=question, intent="plot_fact", query_type="reasoning",
        entities=["斯第奇", *noise], keywords=["离开", "原因", *noise],
        expected_answer_type="原因", dialogue_context="",
    )


def assert_clean(document):
    terms = document.entities + document.keywords
    assert "斯第奇" in terms and "离开" in terms
    assert not set(NOISE) & set(terms)


def test_normalized_model_hypothesis_drops_machine_diagnostics():
    raw = asdict(hypothesis())
    result = normalize_hypothesis_payload(raw, question=QUESTION, dialogue_context="")
    assert_clean(result)


def test_heuristic_followup_cleans_existing_hypothesis_not_just_missing_slots():
    result = build_heuristic_follow_up_hypothesis(QUESTION, hypothesis(), [])
    assert_clean(result)


def test_merged_state_cleans_existing_and_new_terms_together():
    base = hypothesis()
    new = hypothesis(polluted=False)
    new.entities.append("quote")
    new.keywords.append("支撑")
    assert_clean(merge_hypotheses(base, new))


def test_merge_does_not_spend_term_budget_on_internal_tokens():
    base = hypothesis()
    follow = hypothesis(polluted=False)
    follow.entities += ["画布", "际崖城"]
    result = merge_hypotheses(base, follow)
    assert_clean(result)
    assert {"画布", "际崖城"} <= set(result.entities)


@pytest.mark.parametrize("build", [
    lambda h: [build_retrieval_query(h)],
    lambda h: build_follow_up_hypothesis_queries(QUESTION, h),
])
def test_query_builders_are_safe_for_already_polluted_external_state(build):
    result = "\n".join(build(hypothesis()))
    assert "斯第奇" in result
    for term in NOISE:
        assert term not in result


@pytest.mark.parametrize("operation", [
    lambda h: normalize_hypothesis_payload(asdict(h), question=QUESTION, dialogue_context=""),
    lambda h: build_heuristic_follow_up_hypothesis(QUESTION, h, []),
    lambda h: merge_hypotheses(h, hypothesis(polluted=False)),
    lambda h: build_retrieval_query(h),
    lambda h: build_follow_up_hypothesis_queries(QUESTION, h),
])
def test_cleaning_never_mutates_the_input_hypothesis(operation):
    source = hypothesis()
    original = copy.deepcopy(asdict(source))
    operation(source)
    assert asdict(source) == original


@pytest.mark.parametrize("name", ["Logos", "W", "Kal'tsit", "S.W.E.E.P.", "AMa-10", "AMa_10"])
def test_normal_english_names_and_identifiers_are_not_diagnostics(name):
    source = hypothesis(polluted=False, question=f"{name} 为什么来到这里？")
    source.entities = [name]
    source.keywords = [name, "原因"]
    result = build_heuristic_follow_up_hypothesis(source.question, source, [])
    assert name in result.entities and name in result.keywords
    assert name in build_retrieval_query(source)


@pytest.mark.parametrize("term", ["quote", "answer_directly", "missing_supported_facts", "支撑"])
def test_explicit_user_question_terms_are_preserved_even_with_other_noise(term):
    question = f"这里的 {term} 是什么意思？"
    source = hypothesis(question=question)
    source.entities.insert(0, term)
    source.keywords.insert(0, term)
    result = build_heuristic_follow_up_hypothesis(question, source, [])
    assert term in result.entities and term in result.keywords


@pytest.mark.parametrize("term", ["quote", "支撑"])
def test_ambiguous_fragments_are_not_removed_without_diagnostic_context(term):
    source = hypothesis(polluted=False)
    source.keywords.append(term)
    result = build_heuristic_follow_up_hypothesis(QUESTION, source, [])
    assert term in result.keywords


@pytest.mark.parametrize("term", ["quote", "支撑"])
def test_unrelated_machine_error_does_not_make_ambiguous_word_noise(term):
    source = hypothesis(polluted=False)
    source.keywords += [term, "supported_fact_1_ref_1_quote_not_found"]
    result = build_heuristic_follow_up_hypothesis(QUESTION, source, [])
    assert term in result.keywords
    assert "supported_fact_1_ref_1_quote_not_found" not in result.keywords


def test_all_diagnostic_model_entities_fail_instead_of_becoming_search_entities():
    raw = asdict(hypothesis())
    raw["entities"] = ["answer_directly", "missing_supported_facts"]
    with pytest.raises(ModelOutputError, match="entities"):
        normalize_hypothesis_payload(raw, question=QUESTION, dialogue_context="")


def test_clean_query_output_is_unchanged():
    clean = hypothesis(polluted=False)
    assert build_retrieval_query(clean) == (
        QUESTION + "\n实体: 斯第奇\n关键词: 离开 原因\n回答类型: 原因"
    )


def test_diagnostic_phrase_is_not_split_into_new_query_tokens():
    source = hypothesis(polluted=False)
    source.keywords.append("answer_directly 缺少可校验 quote 支撑")
    result = build_heuristic_follow_up_hypothesis(QUESTION, source, [])
    assert "answer_directly 缺少可校验 quote 支撑" not in result.keywords
    assert "answer_directly" not in "\n".join(build_follow_up_hypothesis_queries(QUESTION, source))
