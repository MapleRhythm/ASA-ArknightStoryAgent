from copy import deepcopy
import pytest

from quote_pointer_protocol_v1 import (
    annotate, bind, make_user, slices, strict_loads, validate_catalog,
    OLD_COPY_RULE, OLD_SCHEMA_REF,
)


EVIDENCE = "[E1] 甲：不是我说的。 乙：这是我转述丙的说法。\n[E2] 原文也可能包含[Q1]，不能误删。"


@pytest.mark.parametrize("text", [
    "字" * 1000, "甲：一句话。\n乙：另一句话。" * 80,
    " " * 85 + "实质内容", "🙂" * 150, "short", "\t\n", "",
])
def test_slices_are_lossless_and_bounded(text):
    spans = list(slices(text))
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(0 < b - a <= 80 for a, b in spans)
    assert all(a == spans[i - 1][1] for i, (a, b) in enumerate(spans) if i)


def test_annotation_offsets_and_literal_pointer_text():
    annotated, catalog = annotate(EVIDENCE)
    assert "[Q1]，不能误删。" in annotated
    assert set(r["evidence_id"] for r in catalog) == {"E1", "E2"}
    assert all(EVIDENCE[r["start"]:r["end"]] == r["quote"] for r in catalog)
    assert all("[E2]" not in r["quote"] for r in catalog)
    assert annotate(EVIDENCE) == (annotated, catalog)


@pytest.mark.parametrize("text", ["", "preamble\n[E1] text", "[E0] text",
                                 "[E1] a\n[E1] b", "unlabelled"])
def test_malformed_evidence_rejected(text):
    with pytest.raises(ValueError):
        annotate(text)


def value_for(catalog):
    return {"next_action": "answer_directly",
            "supported_facts": [{"fact": "甲是否认者。", "evidence_refs": [
                {"evidence_id": catalog[0]["evidence_id"], "quote_id": catalog[0]["quote_id"]}]}],
            "inferred_facts": [], "final_answer": "甲是否认者。"}


def test_binding_exact_and_input_immutable():
    _, catalog = annotate(EVIDENCE)
    value = value_for(catalog)
    original, original_catalog = deepcopy(value), deepcopy(catalog)
    result = bind(value, EVIDENCE, catalog)
    assert result["supported_facts"][0]["evidence_refs"] == [
        {"evidence_id": "E1", "quote": catalog[0]["quote"]}]
    assert value == original and catalog == original_catalog
    assert result["final_answer"] == value["final_answer"]


@pytest.mark.parametrize("ref", [
    {"evidence_id": "E1", "quote_id": "Q99999"},
    {"evidence_id": "E2", "quote_id": "Q1"},
    {"evidence_id": "E1", "quote_id": 1},
    {"evidence_id": "E1", "quote_id": "Q1", "quote": "invented"},
    {"evidence_id": "E1", "quote": "copied"},
    {"evidence_id": "E1", "quote_id": ["Q1"]},
    {"evidence_id": "E1", "quote_id": " Q1"},
])
def test_invalid_ref_fails_entire_answer(ref):
    _, catalog = annotate(EVIDENCE)
    value = value_for(catalog)
    value["supported_facts"][0]["evidence_refs"].append(ref)
    before = deepcopy(value)
    with pytest.raises(ValueError):
        bind(value, EVIDENCE, catalog)
    assert value == before


def test_inferences_resolved_without_promoting_or_rewriting():
    _, catalog = annotate(EVIDENCE)
    value = value_for(catalog)
    value["inferred_facts"] = [deepcopy(value["supported_facts"][0])]
    value["inferred_facts"][0]["fact"] = "可能由丙说出。"
    result = bind(value, EVIDENCE, catalog)
    assert result["inferred_facts"][0]["fact"] == "可能由丙说出。"
    assert len(result["supported_facts"]) == 1


@pytest.mark.parametrize("change", [
    {"start": 0}, {"end": 99999}, {"quote": "改写"}, {"evidence_id": "E99"}, {"start": True},
])
def test_catalog_tampering_rejected(change):
    _, catalog = annotate(EVIDENCE)
    catalog[0].update(change)
    with pytest.raises(ValueError):
        validate_catalog(EVIDENCE, catalog)


def test_no_automatic_semantic_acceptance():
    _, catalog = annotate(EVIDENCE)
    value = value_for(catalog)
    value["supported_facts"][0]["fact"] = "甲是说话者且肯定此话。"
    # The binder has no semantic knowledge; downstream review must catch this.
    assert bind(value, EVIDENCE, catalog)["supported_facts"][0]["fact"] == value["supported_facts"][0]["fact"]


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":'])
def test_strict_json_rejects_duplicates_nonfinite_and_truncated(text):
    with pytest.raises(ValueError):
        strict_loads(text)


def test_nonanswer_is_not_rewritten():
    _, catalog = annotate(EVIDENCE)
    value = {"next_action": "retrieve_more", "follow_up_hypothesis": {"question": "为何？"}}
    assert bind(value, EVIDENCE, catalog) == value


def test_direct_empty_facts_rejected():
    _, catalog = annotate(EVIDENCE)
    with pytest.raises(ValueError):
        bind({"next_action": "answer_directly", "supported_facts": []}, EVIDENCE, catalog)


def test_only_annotation_schema_and_copy_rule_change():
    user = (
        "task: conclusion_generation\nquestion: 谁说的？\nhypothesis: keep_this\n"
        "evidence_brief:\n" + EVIDENCE + "\nminirag_hints: keep_that\n"
        + OLD_SCHEMA_REF + "\nrules: before;" + OLD_COPY_RULE + "after"
    )
    changed, annotated, catalog = make_user({"user": user, "evidence_brief": EVIDENCE})
    assert "hypothesis: keep_this" in changed and "minirag_hints: keep_that" in changed
    assert annotated in changed and len(catalog) >= 2
    assert "rules: before;" in changed and "after" in changed


def test_ambiguous_prompt_refuses_to_edit():
    with pytest.raises(ValueError):
        make_user({"user": "unrelated", "evidence_brief": EVIDENCE})
