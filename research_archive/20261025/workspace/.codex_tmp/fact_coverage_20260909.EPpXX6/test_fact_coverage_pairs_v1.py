from copy import deepcopy
from types import SimpleNamespace
import pytest

from build_fact_coverage_pairs_v1 import (
    COVERAGE_RULES, build_pairs, digest, normalized_question, select_states,
)
from run_fact_coverage_pairs_v1 import pair_dispatch


def render(system, user, assistant_prefix):
    return system + "\n" + user + "\n" + assistant_prefix


def sample(rid="a", question="谁执行命令？"):
    row = {
        "id": rid, "question": question, "system": "系统",
        "user": "evidence_brief:\n[E1] 候选甲执行了命令。\nrules: JSON",
        "assistant_prefix": "{", "evidence_brief": "[E1] 候选甲执行了命令。",
        "prompt_evidence": [{"text": "候选甲执行了命令。"}],
        "hypothesis": {"question": question}, "current_round": 1,
    }
    row["runtime_prompt"] = render(row["system"], row["user"], row["assistant_prefix"])
    return row


def test_normalized_questions_match_prior_definition():
    assert normalized_question(" 谁 执行 命令？ ") == "谁执行命令"


def test_selection_independent_of_order_and_candidate_answers():
    rows = [sample("b"), sample("a"), sample("c", "为何？")]
    chosen = select_states(rows)
    changed = deepcopy(rows[::-1])
    for row in changed:
        row["grounded"] = {"answer": "任意结果", "next_action": "abstain"}
        row["evidence_brief"] = "任意证据"
    assert [r["id"] for r in chosen] == [r["id"] for r in select_states(changed)]
    assert len(chosen) == 2


def test_duplicate_source_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_states([sample(), sample()])


def test_empty_question_rejected():
    with pytest.raises(ValueError, match="empty"):
        select_states([sample(question="？")])


def test_pairs_only_add_fixed_instruction_and_do_not_mutate_sources():
    source = [sample("a"), sample("b", "为何？")]
    before = deepcopy(source)
    rows, selection = build_pairs(source, render, {"谁执行命令"})
    assert source == before
    for index in range(2):
        pair = rows[index * 2:index * 2 + 2]
        assert pair[0]["sampling_seed"] == pair[1]["sampling_seed"]
        by_variant = {r["renderer"]: r for r in pair}
        control = by_variant["original_prompt_v1"]
        candidate = by_variant["fact_coverage_prompt_v1"]
        assert control["runtime_prompt"] == source[index]["runtime_prompt"]
        assert candidate["user"] == control["user"] + COVERAGE_RULES
        for key in ("system", "assistant_prefix", "question", "evidence_brief",
                    "prompt_evidence", "hypothesis", "current_round"):
            assert candidate[key] == control[key] == source[index][key]
        assert candidate["runtime_prompt_sha256"] == digest(candidate["runtime_prompt"])
    assert selection[0]["first_variant"] == "original_prompt_v1"
    assert selection[1]["first_variant"] == "fact_coverage_prompt_v1"
    assert [r["prior_six_question_development_set"] for r in selection] == [True, False]


def test_rules_contain_no_story_specific_development_answer():
    assert all(word not in COVERAGE_RULES for word in (
        "行星碎屑", "充能", "部署", "矿石病", "夜刀", "美食节", "娜斯提", "里昂",
    ))
    assert "继续检索/弃答" in COVERAGE_RULES and "否定与可能性" in COVERAGE_RULES


def test_dispatch_checks_prompt_seed_and_adapter():
    rows, _ = build_pairs([sample()], render, set())
    calls, actual = [], []
    adapter = SimpleNamespace(lora_int_id=7)

    def real(prompts, params, **kwargs):
        actual.append((prompts, params.seed, kwargs["lora_request"]))
        return [SimpleNamespace(lora_request=adapter)]

    dispatch = pair_dispatch(real, rows, calls)
    for row in rows:
        dispatch([row["runtime_prompt"]], SimpleNamespace(seed=None), lora_request=adapter)
    assert actual[0][1] == actual[1][1]
    assert all(item[2] is adapter for item in actual)
    assert all(row["returned_lora_echo"] == [7] for row in calls)
    with pytest.raises(AssertionError, match="extra"):
        dispatch(["unexpected"], SimpleNamespace(), lora_request=adapter)


def test_dispatch_rejects_mismatched_prompt_and_missing_adapter():
    rows, _ = build_pairs([sample()], render, set())
    calls = []
    dispatch = pair_dispatch(lambda *a, **kw: [], rows, calls)
    with pytest.raises(AssertionError):
        dispatch(["different"], SimpleNamespace(), lora_request=object())
    with pytest.raises(AssertionError, match="LoRA"):
        dispatch([rows[0]["runtime_prompt"]], SimpleNamespace())
    assert calls == []


def test_dispatch_preserves_failed_call():
    rows, _ = build_pairs([sample()], render, set())
    calls = []

    def failed(*a, **kw):
        raise RuntimeError("failure")

    dispatch = pair_dispatch(failed, rows, calls)
    with pytest.raises(RuntimeError, match="failure"):
        dispatch([rows[0]["runtime_prompt"]], SimpleNamespace(), lora_request=object())
    assert calls[0]["status"] == "error"
    assert calls[0]["error"] == "RuntimeError: failure"
