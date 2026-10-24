"""Offline tests: query cleanup must not manufacture reviewed KTO positives."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(os.environ.get(
    "ASA_VERIFIER_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "scripts/build_soda_api_verifier_dataset.py"),
))
spec = importlib.util.spec_from_file_location("soda_followup_integrity_target", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

QUESTION = "斯第奇为什么离开际崖城？"
NOISE = [
    "answer_directly", "缺少可校验", "quote", "支撑",
    "supported_fact_123456789_ref_123456789_quote_not_found",
    "inferred_fact_2_missing_evidence_refs", "missing_supported_facts",
]


def followup(question=QUESTION, polluted=False):
    return {
        "question": question, "query_type": "reasoning",
        "entities": ["斯第奇", *(NOISE if polluted else [])],
        "keywords": ["离开", "原因", *(NOISE if polluted else [])],
        "expected_answer_type": "剧情问答", "dialogue_context": "",
    }


def prompt(hypothesis=None, question=QUESTION):
    return (
        "hypothesis: " + json.dumps(hypothesis or followup(question), ensure_ascii=False)
        + f"\nround: 1/2\nquestion: {question}\nevidence_brief:\n[E1] 斯第奇离开了。\n"
        "output_schema: grounded_action_v1\n"
    )


def retrieve(hypothesis=None):
    return {"next_action": "retrieve_more", "follow_up_hypothesis": hypothesis or followup()}


def record(payload=None, *, tag=True):
    return {
        "id": "teacher" if tag else "student", "kto_tag": tag,
        "task_type": "conclusion_generation", "meta": {"prompt_key": "state"},
        "conversations": [
            {"from": "human", "value": prompt()},
            {"from": "gpt", "value": json.dumps(payload or retrieve(), ensure_ascii=False)},
        ],
    }


def relabel(group, **changes):
    verdict = {
        "correct_action": "retrieve_more", "evidence_sufficient": False, "use_for_training": True,
        "teacher_action_error": "none", "student_action_error": "none",
        "teacher_answer_uses_prior_knowledge": False, "missing_slots": ["离开的原因"],
        "supported_answer": "", "label_reason": "fixture",
    }
    verdict.update(changes)
    stats = Counter()
    output = builder.relabel_group("state", group, {"verifier": verdict}, stats)
    return output, stats


def positives(output):
    return [json.loads(builder.response_text(r)) for r in output if r["kto_tag"] is True]


def assert_clean(result):
    terms = result["entities"] + result["keywords"]
    assert {"斯第奇", "离开"} <= set(terms)
    assert not set(NOISE) & set(terms)
    assert not any("supported_fact_" in t or "inferred_fact_" in t for t in terms)


def test_fallback_cleans_existing_hypothesis():
    assert_clean(builder.build_fallback_follow_up(QUESTION, prompt(followup(polluted=True)), []))


def test_normalizer_cleans_model_followup():
    assert_clean(builder.normalize_follow_up_payload(followup(polluted=True), question=QUESTION, prompt=prompt()))


def test_machine_diagnostic_is_removed_before_tokenization():
    result = builder.build_fallback_follow_up(QUESTION, prompt(), [
        "answer_directly 缺少可校验 quote 支撑",
        "supported_fact_123456789_ref_123456789_quote_not_found", "离开的原因",
    ])
    assert_clean(result)
    assert "离开的原因" in result["keywords"]


def test_cleanup_precedes_term_budget():
    source = followup()
    source["entities"] = [f"supported_fact_{i}_missing_quote" for i in range(20)] + ["斯第奇"]
    source["keywords"] = [f"inferred_fact_{i}_missing_quote" for i in range(30)] + ["离开"]
    result = builder.normalize_follow_up_payload(source, question=QUESTION, prompt=prompt())
    assert_clean(result)


def test_polluted_only_fields_get_question_fallback():
    source = followup()
    source["entities"] = source["keywords"] = ["answer_directly", "missing_supported_facts"]
    result = builder.normalize_follow_up_payload(source, question=QUESTION, prompt=prompt())
    assert result["entities"] and result["keywords"]
    assert not set(NOISE) & set(result["entities"] + result["keywords"])


@pytest.mark.parametrize("term", ["Logos", "W", "AMa_10", "S.W.E.E.P."])
def test_preserve_ordinary_names(term):
    source = followup()
    source["entities"].append(term)
    result = builder.normalize_follow_up_payload(source, question=QUESTION, prompt=prompt())
    assert term in result["entities"]


@pytest.mark.parametrize("term", ["answer_directly", "quote", "missing_supported_facts", "支撑"])
def test_preserve_explicit_question_terms(term):
    question = f"{term} 是什么意思？"
    source = followup(question, polluted=True)
    result = builder.normalize_follow_up_payload(source, question=question, prompt=prompt(question=question))
    assert term in result["entities"] and term in result["keywords"]


@pytest.mark.parametrize("term", ["quote", "支撑"])
def test_ambiguous_words_are_not_globally_banned(term):
    source = followup()
    source["keywords"].append(term)
    result = builder.normalize_follow_up_payload(source, question=QUESTION, prompt=prompt())
    assert term in result["keywords"]


def test_clean_followup_stays_identical_and_input_not_mutated():
    source = followup()
    before = deepcopy(source)
    assert builder.normalize_follow_up_payload(source, question=QUESTION, prompt=prompt()) == source
    assert source == before


def test_conclusion_legacy_route_cleans_explicit_followup_too():
    result = builder.normalize_conclusion_payload(
        retrieve(followup(polluted=True)), question=QUESTION, prompt=prompt(),
    )
    assert_clean(result["follow_up_hypothesis"])


def test_reviewed_clean_target_remains_positive():
    output, _ = relabel([record()])
    assert positives(output) == [retrieve()]


def test_cleaned_target_must_not_inherit_old_review():
    original = record(retrieve(followup(polluted=True)))
    before = deepcopy(original)
    output, stats = relabel([original])
    assert positives(output) == []
    assert not output  # unapproved edited output is not a new negative, either
    assert original == before
    assert stats["skip:teacher_follow_up_changed_requires_reaudit"] == 1


def test_synthesized_fallback_is_not_an_approved_positive():
    original = record({"next_action": "answer_directly", "answer": "无依据回答"})
    output, _ = relabel([original], teacher_action_error="premature_answer")
    assert not positives(output)
    assert len(output) == 1 and output[0]["kto_tag"] is False
    assert builder.response_text(output[0]) == builder.response_text(original)


@pytest.mark.parametrize("field,value", [
    ("teacher_action_error", None),
    ("teacher_action_error", ""),
    ("teacher_answer_uses_prior_knowledge", None),
    ("teacher_answer_uses_prior_knowledge", "false"),
])
def test_ambiguous_approval_does_not_approve_retrieve_positive(field, value):
    output, _ = relabel([record()], **{field: value})
    assert not positives(output)


@pytest.mark.parametrize("kind", ["fenced", "duplicate", "missing_followup", "extra_key"])
def test_repaired_or_changed_json_is_not_a_reviewed_target(kind):
    source = record()
    raw = builder.response_text(source)
    if kind == "fenced":
        source["conversations"][-1]["value"] = "```json\n" + raw + "\n```"
    elif kind == "duplicate":
        source["conversations"][-1]["value"] = raw[:-1] + ',"next_action":"retrieve_more"}'
    elif kind == "missing_followup":
        source["conversations"][-1]["value"] = '{"next_action":"retrieve_more"}'
    else:
        source["conversations"][-1]["value"] = json.dumps({**retrieve(), "invented": True})
    output, _ = relabel([source])
    assert not positives(output)


def test_select_unchanged_reviewed_student_when_teacher_needs_cleaning():
    output, _ = relabel([
        record(retrieve(followup(polluted=True))),
        record(tag=False),
    ])
    assert positives(output) == [retrieve()]


def test_rejected_polluted_output_is_kept_verbatim():
    original = record(retrieve(followup(polluted=True)))
    output, _ = relabel([original], teacher_action_error="invalid_output")
    assert not positives(output)
    assert len(output) == 1 and output[0]["kto_tag"] is False
    assert builder.response_text(output[0]) == builder.response_text(original)


@pytest.mark.parametrize("legacy_fields", [
    {"question": QUESTION},
    {"answer": ""},
    {"clarification_question": ""},
    {"missing_slots": []},
    {"question": QUESTION, "answer": "", "clarification_question": "", "missing_slots": []},
])
def test_lossless_empty_legacy_envelope_is_kept_with_provenance(legacy_fields):
    source = record({**retrieve(), **legacy_fields})
    before = deepcopy(source)
    output, _ = relabel([source])
    assert positives(output) == [retrieve()]
    assert source == before
    kept = next(r for r in output if r["kto_tag"] is True)
    trace = kept["meta"]["verifier_target_projection"]
    assert trace["removed_redundant_fields"] == sorted(legacy_fields)
    assert trace["reviewed_source_record_id"] == source["id"]
    assert trace["retrieval_payload_unchanged"] is True


@pytest.mark.parametrize("legacy_fields", [
    {"question": "另一个问题？"},
    {"answer": "实际回答内容"},
    {"clarification_question": "另一个澄清？"},
    {"missing_slots": ["离开原因"]},
    {"answer": None},
    {"missing_slots": ""},
])
def test_nonempty_or_nonstandard_legacy_content_is_not_silently_projected(legacy_fields):
    output, _ = relabel([record({**retrieve(), **legacy_fields})])
    assert not positives(output)


def test_empty_legacy_projection_does_not_allow_polluted_followup():
    source = record({**retrieve(followup(polluted=True)), "question": QUESTION, "answer": ""})
    output, _ = relabel([source])
    assert not positives(output)
