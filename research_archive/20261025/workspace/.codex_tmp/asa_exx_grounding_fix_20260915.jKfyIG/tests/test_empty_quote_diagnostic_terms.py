from asa_arknight_story_agent.inference.planning.retrieval_term_cleaning import (
    clean_hypothesis_terms, is_internal_diagnostic_term,
)


def test_new_quote_failure_does_not_become_a_story_search_term():
    terms = [
        "supported_fact_1_ref_1_quote_empty_after_normalization",
        "inferred_fact_2_ref_1_quote_empty_after_normalization",
    ]
    assert all(is_internal_diagnostic_term(term) for term in terms)
    entities, keywords = clean_hypothesis_terms(
        ["角色甲"], ["窗边", *terms], question="角色甲在做什么？",
    )
    assert entities == ["角色甲"] and keywords == ["窗边"]


def test_explicit_user_request_for_diagnostic_name_remains_searchable():
    term = "supported_fact_1_ref_1_quote_empty_after_normalization"
    _, keywords = clean_hypothesis_terms([], [term], question=f"{term}是什么意思？")
    assert keywords == [term]
