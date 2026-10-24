from copy import deepcopy

import pytest

from quote_pointer_protocol_v1 import OLD_COPY_RULE, OLD_SCHEMA_REF, annotate
from single_quote_pointer_protocol_v1 import (
    OLD_EID_RULE, SINGLE_RULES, SINGLE_SCHEMA_REF, bind, make_user, strict_loads,
)

EVIDENCE = "[E1] 第一份完整的原始证据。\n[E2] 第二份证据包含另一段不同的原文。"


def payload(qid="Q2"):
    return {"next_action": "answer_directly", "final_answer": "未经语义认证的陈述",
            "supported_facts": [{"fact": "未经语义认证的陈述", "evidence_refs": [{"quote_id": qid}]}],
            "inferred_facts": []}


def original():
    return {"evidence_brief": EVIDENCE,
            "user": "task: test\n" + OLD_SCHEMA_REF + "\n" + OLD_EID_RULE + OLD_COPY_RULE
                    + "\nevidence_brief:\n" + EVIDENCE + "\nminirag_hints: 原样提示"}


def test_global_q_resolves_exact_source_and_owning_e_without_semantic_repair():
    _, catalog = annotate(EVIDENCE)
    before = payload()
    after = bind(before, EVIDENCE, catalog)
    assert after["supported_facts"][0]["fact"] == "未经语义认证的陈述"
    assert after["supported_facts"][0]["evidence_refs"] == [
        {"evidence_id": "E2", "quote": catalog[1]["quote"]}]


def test_input_objects_are_not_mutated():
    _, catalog = annotate(EVIDENCE)
    before, cat_before = payload(), deepcopy(catalog)
    value = deepcopy(before)
    bind(value, EVIDENCE, catalog)
    assert before == value and catalog == cat_before


@pytest.mark.parametrize("extra", [{"evidence_id": "E1"}, {"evidence_id": "E2"}, {"quote": "改写原文"}])
def test_dual_pointer_or_quote_text_is_rejected_not_repaired(extra):
    _, catalog = annotate(EVIDENCE)
    value = payload()
    value["supported_facts"][0]["evidence_refs"][0].update(extra)
    with pytest.raises(ValueError, match="exactly quote_id"):
        bind(value, EVIDENCE, catalog)


@pytest.mark.parametrize("qid", ["Q999", "Q0", "[Q2]", "q2", 2, None, ["Q2"]])
def test_unknown_or_malformed_pointers_fail_closed(qid):
    _, catalog = annotate(EVIDENCE)
    with pytest.raises(ValueError, match="unknown or non-string"):
        bind(payload(qid), EVIDENCE, catalog)


def test_no_fact_is_dropped_when_one_ref_is_bad():
    _, catalog = annotate(EVIDENCE)
    value = payload()
    value["supported_facts"].append({"fact": "第二条", "evidence_refs": [{"quote_id": "Q999"}]})
    with pytest.raises(ValueError):
        bind(value, EVIDENCE, catalog)
    assert len(value["supported_facts"]) == 2


def test_inferred_facts_use_same_single_pointer_contract():
    _, catalog = annotate(EVIDENCE)
    value = payload()
    value["inferred_facts"] = [{"fact": "推测", "evidence_refs": [{"quote_id": "Q1"}]}]
    result = bind(value, EVIDENCE, catalog)
    assert result["inferred_facts"][0]["evidence_refs"][0]["evidence_id"] == "E1"


def test_multiple_refs_preserve_order_and_repetition_for_existing_budget_gate():
    _, catalog = annotate(EVIDENCE)
    value = payload()
    value["supported_facts"][0]["evidence_refs"] = [
        {"quote_id": "Q2"}, {"quote_id": "Q1"}, {"quote_id": "Q2"}]
    refs = bind(value, EVIDENCE, catalog)["supported_facts"][0]["evidence_refs"]
    assert [r["evidence_id"] for r in refs] == ["E2", "E1", "E2"]


@pytest.mark.parametrize("refs", [[], None, {}, "Q1"])
def test_missing_or_empty_fact_refs_rejected(refs):
    _, catalog = annotate(EVIDENCE)
    value = payload()
    value["supported_facts"][0]["evidence_refs"] = refs
    with pytest.raises(ValueError, match="nonempty pointer refs"):
        bind(value, EVIDENCE, catalog)


def test_direct_without_facts_rejected():
    _, catalog = annotate(EVIDENCE)
    with pytest.raises(ValueError, match="requires supported_facts"):
        bind({"next_action": "answer_directly", "supported_facts": []}, EVIDENCE, catalog)


def test_abstention_without_facts_is_not_forced_to_answer():
    _, catalog = annotate(EVIDENCE)
    value = {"next_action": "abstain", "supported_facts": [], "inferred_facts": []}
    assert bind(value, EVIDENCE, catalog) == value


def test_corrupt_catalog_fails_before_resolution():
    _, catalog = annotate(EVIDENCE)
    catalog[1]["evidence_id"] = "E1"
    with pytest.raises(ValueError):
        bind(payload(), EVIDENCE, catalog)


def test_new_user_retains_identical_source_annotation():
    row = original()
    frozen = deepcopy(row)
    user, annotated, catalog = make_user(row)
    assert row == frozen
    assert (annotated, catalog) == annotate(EVIDENCE)
    assert SINGLE_SCHEMA_REF in user and SINGLE_RULES in user
    assert OLD_SCHEMA_REF not in user and OLD_COPY_RULE not in user
    assert OLD_EID_RULE not in user
    assert "minirag_hints: 原样提示" in user
    assert "每fact160字、总400字" in user


def test_old_dual_pointer_prompt_not_reinterpreted():
    row = original()
    row["user"] = row["user"].replace(OLD_SCHEMA_REF, '"evidence_refs":[{"evidence_id":"E1","quote_id":"Q1"}]')
    with pytest.raises(ValueError, match="unexpected original"):
        make_user(row)


def test_strict_json_duplicate_key_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_loads('{"quote_id":"Q1","quote_id":"Q2"}')
