from __future__ import annotations

from dataclasses import asdict

from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import (
    build_minimal_conclusion_prompt,
)
from asa_arknight_story_agent.inference.generation.exx_prompt import (
    EXX_PROTOCOL,
    EXX_RULES,
    EXX_SYSTEM_PROMPT,
    render_exx_user_prompt,
)
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import (
    validate_evidence_id_grounding,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def _evidence() -> list[dict]:
    return [
        {
            "doc_index": 1,
            "document": {
                "id": "story#1",
                "clean_text": "阿米娅检查了报告，随后同意娜塔莉娅的调岗申请。",
            },
        },
        {
            "doc_index": 2,
            "document": {
                "id": "story#2",
                "clean_text": "杜宾继续安排当天的训练计划。",
            },
        },
    ]


def _hypothesis() -> HypothesisDocument:
    return HypothesisDocument(
        question="阿米娅如何处理调岗申请？",
        intent="plot_fact",
        query_type="fact",
        entities=["阿米娅"],
        keywords=["调岗申请"],
        expected_answer_type="事实",
    )


def _conclusion(evidence_id: str) -> ConclusionResult:
    return ConclusionResult(
        next_action="answer_directly",
        answer="阿米娅检查报告后同意了娜塔莉娅的调岗申请。",
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {
                "fact": "阿米娅检查报告后同意了娜塔莉娅的调岗申请。",
                "evidence_refs": [{"evidence_id": evidence_id}],
            }
        ],
    )


def test_exx_payload_normalizes_compact_evidence_ids_for_runtime_validator() -> None:
    from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload

    conclusion = normalize_conclusion_payload(
        {
            "next_action": "answer_directly",
            "supported_facts": [
                {
                    "fact": "阿米娅检查报告后同意了娜塔莉娅的调岗申请。",
                    "evidence_ids": ["E1"],
                }
            ],
        },
        question="阿米娅如何处理调岗申请？",
        dialogue_context="",
        current_intent="plot_fact",
        current_hypothesis=_hypothesis(),
    )
    assert conclusion.answer == "阿米娅检查报告后同意了娜塔莉娅的调岗申请。"
    assert conclusion.supported_facts == [
        {
            "fact": "阿米娅检查报告后同意了娜塔莉娅的调岗申请。",
            "evidence_refs": [{"evidence_id": "E1"}],
        }
    ]


def test_exx_abstain_reason_is_user_visible_answer() -> None:
    from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload

    conclusion = normalize_conclusion_payload(
        {"next_action": "abstain", "reason": "现有证据不足以确认。"},
        question="测试问题？",
        dialogue_context="",
        current_intent="plot_fact",
        current_hypothesis=_hypothesis(),
        max_round_reached=True,
    )
    assert conclusion.next_action == "abstain"
    assert conclusion.answer == "现有证据不足以确认。"


def test_exx_retrieve_more_accepts_follow_up_query_alias() -> None:
    from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload

    conclusion = normalize_conclusion_payload(
        {
            "next_action": "retrieve_more",
            "follow_up_query": {
                "question": "需要继续查什么？",
                "query_type": "fact",
                "entities": ["阿米娅"],
                "keywords": ["报告"],
                "expected_answer_type": "事实",
                "dialogue_context": "",
            },
        },
        question="测试问题？",
        dialogue_context="",
        current_intent="plot_fact",
        current_hypothesis=_hypothesis(),
    )
    assert conclusion.next_action == "retrieve_more"
    assert conclusion.follow_up_hypothesis is not None
    # The runtime intentionally keeps the original user question as the
    # hypothesis question while importing the model's follow-up entities and
    # keywords as retrieval guidance.
    assert conclusion.follow_up_hypothesis.question == "测试问题？"
    assert "阿米娅" in conclusion.follow_up_hypothesis.entities


def test_evidence_id_mode_keeps_evidence_text_in_input_but_not_output_schema() -> None:
    prompt, evidence_brief = build_minimal_conclusion_prompt(
        question="阿米娅如何处理调岗申请？",
        current_hypothesis=_hypothesis(),
        evidence=_evidence(),
        current_round=1,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=2,
        grounding_mode="evidence_id",
    )
    assert "[E1]\n阿米娅检查了报告" in evidence_brief
    assert f"output_schema: {EXX_PROTOCOL}" in prompt
    assert "evidence_ids" in prompt
    assert "evidence_refs" in prompt  # forbidden-field rule, not output schema
    assert '"quote"' not in prompt
    assert "不要输出evidence_refs、quote" in prompt
    assert prompt.startswith(f"<|im_start|>system\n{EXX_SYSTEM_PROMPT}")
    assert prompt.endswith("<|im_start|>assistant\n")


def test_exx_prompt_drops_whole_lower_rank_evidence_instead_of_truncating() -> None:
    evidence = _evidence()
    evidence[0]["document"]["clean_text"] = "甲" * 40
    evidence[1]["document"]["clean_text"] = "乙" * 40
    _, brief = build_minimal_conclusion_prompt(
        question="测试？",
        current_hypothesis=_hypothesis(),
        evidence=evidence,
        current_round=1,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=2,
        evidence_max_chars_per_doc=10,
        evidence_max_total_chars=48,
        grounding_mode="evidence_id",
    )
    assert "甲" * 40 in brief
    assert "…" not in brief
    assert "[E2]" not in brief


def test_exx_prompt_returns_no_partial_evidence_when_first_item_exceeds_budget() -> None:
    evidence = _evidence()
    evidence[0]["document"]["clean_text"] = "甲" * 40
    _, brief = build_minimal_conclusion_prompt(
        question="测试？",
        current_hypothesis=_hypothesis(),
        evidence=evidence,
        current_round=1,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=2,
        evidence_max_chars_per_doc=10,
        evidence_max_total_chars=20,
        grounding_mode="evidence_id",
    )
    assert brief == ""


def test_evidence_id_validator_accepts_existing_supporting_id() -> None:
    issues, _ = validate_evidence_id_grounding(
        conclusion=_conclusion("E1"), evidence=_evidence(), question="阿米娅如何处理调岗申请？"
    )
    assert issues == []


def test_evidence_id_validator_rejects_unknown_id() -> None:
    issues, _ = validate_evidence_id_grounding(
        conclusion=_conclusion("E99"), evidence=_evidence(), question="阿米娅如何处理调岗申请？"
    )
    assert any("evidence_id_not_found:E99" in issue for issue in issues)


def test_evidence_id_validator_rejects_claim_terms_outside_cited_evidence() -> None:
    conclusion = _conclusion("E1")
    conclusion.supported_facts[0]["fact"] = "阿米娅任命娜塔莉娅为整合运动领袖。"
    issues, _ = validate_evidence_id_grounding(
        conclusion=conclusion, evidence=_evidence(), question="阿米娅如何处理调岗申请？"
    )
    assert any("terms_outside_cited_evidence" in issue for issue in issues)


def test_evidence_id_validator_checks_sensitive_relation_even_if_question_mentions_it() -> None:
    conclusion = _conclusion("E1")
    conclusion.supported_facts[0]["fact"] = "娜塔莉娅是阿米娅的母亲。"
    issues, _ = validate_evidence_id_grounding(
        conclusion=conclusion,
        evidence=_evidence(),
        question="娜塔莉娅是不是阿米娅的母亲？",
    )
    assert any("sensitive_terms_outside_cited_evidence:母亲" in issue for issue in issues)


def test_evidence_id_validator_uses_only_text_visible_in_truncated_prompt() -> None:
    conclusion = _conclusion("E1")
    conclusion.supported_facts[0]["fact"] = "阿米娅任命娜塔莉娅为整合运动领袖。"
    issues, _ = validate_evidence_id_grounding(
        conclusion=conclusion,
        evidence=_evidence(),
        question="阿米娅如何处理调岗申请？",
        evidence_prompt_text="[E1] 阿米娅检查了报告。\n[E2] 杜宾安排训练。",
    )
    assert any("terms_outside_cited_evidence" in issue for issue in issues)


def test_full_prompt_request_uses_compact_schema_in_evidence_id_mode() -> None:
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import (
        build_conclusion_prompt,
    )

    prompt, evidence_text = build_conclusion_prompt(
        question="阿米娅如何处理调岗申请？",
        current_hypothesis=_hypothesis(),
        evidence=_evidence(),
        retrieval_trace=[],
        current_round=1,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=2,
        prompt_mode="full",
        grounding_mode="evidence_id",
    )
    assert evidence_text.startswith("[E1]")
    assert f"output_schema: {EXX_PROTOCOL}" in prompt
    assert "evidence_ids" in prompt
    assert '"quote"' not in prompt


def test_runtime_exx_prompt_body_is_the_canonical_training_prompt() -> None:
    prompt, evidence_text = build_minimal_conclusion_prompt(
        question="阿米娅如何处理调岗申请？",
        current_hypothesis=_hypothesis(),
        evidence=_evidence(),
        current_round=1,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=2,
        grounding_mode="evidence_id",
    )
    expected_body = render_exx_user_prompt(
        question="阿米娅如何处理调岗申请？",
        hypothesis=asdict(_hypothesis()),
        round_value="1/2",
        evidence_text=evidence_text,
    )
    assert expected_body in prompt
    assert f"rules: {EXX_RULES}" in prompt


def test_canonical_exx_prompt_compacts_serialized_hypothesis() -> None:
    prompt = render_exx_user_prompt(
        question="测试？",
        hypothesis='{"entities": ["阿米娅"], "keywords": ["报告"]}',
        round_value="1/2",
        evidence_text="[E1]\n正文",
    )
    assert 'hypothesis: {"entities":["阿米娅"],"keywords":["报告"]}' in prompt


def test_offline_evidence_map_can_match_prompt_truncation_limits() -> None:
    mapping = evidence_id_text_map(_evidence(), max_chars_per_doc=8, max_total_chars=100)
    assert set(mapping) == {"E1", "E2"}
    assert len(mapping["E1"]) <= 8


def test_grounding_disabled_does_not_run_legacy_fallbacks() -> None:
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding

    conclusion = ConclusionResult(
        next_action="abstain",
        answer="原始回答",
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
    )
    actual = validate_conclusion_grounding(
        question="阿米娅如何处理调岗申请？",
        hypothesis=_hypothesis(),
        evidence=_evidence(),
        conclusion=conclusion,
        max_round_reached=True,
        mode="off",
    )
    assert actual is conclusion


def test_evidence_id_mode_discards_untrusted_final_answer_text() -> None:
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding

    conclusion = _conclusion("E1")
    conclusion.answer = "阿米娅同意申请，并秘密任命她为整合运动领袖。"
    actual = validate_conclusion_grounding(
        question="阿米娅如何处理调岗申请？",
        hypothesis=_hypothesis(),
        evidence=_evidence(),
        conclusion=conclusion,
        max_round_reached=True,
        mode="evidence_id",
        evidence_prompt_text="[E1] 阿米娅检查了报告，随后同意娜塔莉娅的调岗申请。",
    )
    assert actual.next_action == "answer_directly"
    assert actual.answer == conclusion.supported_facts[0]["fact"]
    assert "整合运动" not in actual.answer


def test_evidence_id_failure_does_not_surface_unvalidated_retrieval_snippets() -> None:
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding

    conclusion = _conclusion("E99")
    actual = validate_conclusion_grounding(
        question="阿米娅如何处理调岗申请？",
        hypothesis=_hypothesis(),
        evidence=_evidence(),
        conclusion=conclusion,
        max_round_reached=True,
        mode="evidence_id",
        evidence_prompt_text="[E1] 阿米娅检查了报告，随后同意娜塔莉娅的调岗申请。",
    )
    assert actual.next_action == "abstain"
    assert actual.answer == "现有检索证据不足以给出可校验的回答。"
    assert "阿米娅检查了报告" not in actual.answer


def test_valid_evidence_id_answer_is_not_rejected_by_legacy_token_validator() -> None:
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding

    conclusion = _conclusion("E1")
    actual = validate_conclusion_grounding(
        question="阿米娅如何处理调岗申请？",
        hypothesis=_hypothesis(),
        evidence=_evidence(),
        conclusion=conclusion,
        max_round_reached=False,
        mode="evidence_id",
        evidence_prompt_text="[E1] 阿米娅检查了报告，随后同意娜塔莉娅的调岗申请。",
    )
    assert actual.next_action == "answer_directly"
    assert actual.answer == conclusion.supported_facts[0]["fact"]


def test_truncated_exx_recovery_preserves_evidence_ids() -> None:
    from asa_arknight_story_agent.inference.payload.truncated_answer_recovery import (
        recover_truncated_grounded_answer,
    )

    recovered = recover_truncated_grounded_answer(
        '{"next_action":"answer_directly","supported_facts":'
        '[{"fact":"事实甲","evidence_ids":["E1","E2"]},'
        '{"fact":"事实乙","evidence_ids":["E3"]}',
        question="测试？",
    )
    assert recovered is not None
    assert recovered.supported_facts == [
        {"fact": "事实甲", "evidence_refs": [{"evidence_id": "E1"}, {"evidence_id": "E2"}]},
        {"fact": "事实乙", "evidence_refs": [{"evidence_id": "E3"}]},
    ]
