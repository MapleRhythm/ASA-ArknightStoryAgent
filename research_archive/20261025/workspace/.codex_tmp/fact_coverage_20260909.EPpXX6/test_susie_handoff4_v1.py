from copy import deepcopy
import pytest
from run_susie_handoff4_v1 import build_rows, ARMS, pair_dispatch


def render(system, user, assistant_prefix="{"):
    return "<|im_start|>system\n" + system.strip() + "<|im_end|>\n<|im_start|>user\n" + user.strip() + \
        "<|im_end|>\n<|im_start|>assistant\n" + assistant_prefix


def fixtures():
    original = {"source_id": "s", "question": "q", "hypothesis": {}, "current_round": 1,
                "max_retrieval_rounds": 2, "sampling_seed": 71, "system": "system",
                "assistant_prefix": "{", "template": "runtime_raw", "parent_record_id": "p"}
    conditions = []
    for i, (label, reserve) in enumerate(ARMS):
        brief = f"[E1] text{i}"
        conditions.append({
            "scoring_condition": label, "order_only_reserve6": reserve,
            "runtime_prompt": render("system", brief), "evidence_brief": brief,
            "selected": [{"document": {"id": f"new{i}", "clean_text": f"text{i}"}}],
            "packing_trace": [{"status": "included", "evidence_id": "E1", "source_id": f"new{i}"}],
        })
    return original, conditions


def test_every_arm_keeps_its_own_evidence_mapping():
    original, conditions = fixtures()
    rows = build_rows(original, conditions, render)
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert row["sampling_seed"] == 71 and row["evidence_id_to_corpus_id"] == {"E1": f"new{i}"}
        assert row["runtime_prompt"] == conditions[i]["runtime_prompt"]
        assert row["prompt_evidence"] == conditions[i]["selected"]


def test_original_inputs_not_mutated():
    original, conditions = fixtures()
    before = deepcopy((original, conditions))
    build_rows(original, conditions, render)
    assert (original, conditions) == before


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_wrong_condition_count_rejected(change):
    original, conditions = fixtures()
    conditions = conditions[:-1] if change == "missing" else conditions + conditions[:1]
    with pytest.raises(AssertionError, match="missing or duplicate"):
        build_rows(original, conditions, render)


def test_template_drift_rejected():
    original, conditions = fixtures()
    conditions[0]["runtime_prompt"] = "different prefix" + conditions[0]["runtime_prompt"]
    with pytest.raises(AssertionError, match="unexpected prompt template"):
        build_rows(original, conditions, render)


def test_evidence_drift_rejected():
    original, conditions = fixtures()
    conditions[0]["evidence_brief"] = "not shown"
    with pytest.raises(AssertionError, match="evidence/prompt mismatch"):
        build_rows(original, conditions, render)


def test_identical_prompts_not_generated_twice():
    original, conditions = fixtures()
    conditions[1]["runtime_prompt"] = conditions[0]["runtime_prompt"]
    conditions[1]["evidence_brief"] = conditions[0]["evidence_brief"]
    with pytest.raises(AssertionError, match="identical prompts"):
        build_rows(original, conditions, render)


def test_missing_adapter_rejected_before_call():
    original, conditions = fixtures()
    rows = build_rows(original, conditions, render)
    def real(*args, **kwargs):
        raise AssertionError("must not dispatch")
    capture = pair_dispatch(real, rows, [])
    with pytest.raises(AssertionError, match="configured LoRA"):
        capture([rows[0]["runtime_prompt"]], object())
