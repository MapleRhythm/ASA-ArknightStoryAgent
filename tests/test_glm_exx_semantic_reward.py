import importlib.util
import io
import json
import math
import sys
import urllib.error
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "glm_exx_semantic_reward.py"
SPEC = importlib.util.spec_from_file_location("glm_exx_semantic_reward", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


PROMPT = """question: 谁批准了申请？
hypothesis: {}
round: 1/2
evidence:
[E1]
阿米娅批准了申请。
[E2]
杜宾安排了训练。
output_schema: grounded_action_exx_v1
rules: omitted
"""


def test_extract_judge_context_excludes_hypothesis_and_rules() -> None:
    context = MODULE.extract_judge_context(PROMPT)

    assert context == {
        "question": "谁批准了申请？",
        "round": "1/2",
        "evidence": "[E1]\n阿米娅批准了申请。\n[E2]\n杜宾安排了训练。",
    }


def test_judge_prompt_encodes_round_action_state_machine() -> None:
    messages = MODULE.build_messages(
        MODULE.extract_judge_context(PROMPT),
        [(0, {"next_action": "abstain", "reason": "证据不足。"})],
    )

    assert "证据不足且仍有后续轮次" in messages[1]["content"]
    assert "不得因当前证据不足就在非末轮提前abstain" in messages[1]["content"]


def test_validate_and_score_answer_judgement() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "阿米娅批准了申请。", "evidence_ids": ["E1"]}],
    }
    value = {
        "protocol": MODULE.PROTOCOL_VERSION,
        "rollouts": [
            {
                "rollout_index": 0,
                "action_appropriateness": "appropriate",
                "facts": [
                    {
                        "fact_index": 0,
                        "support": "entailed",
                        "checked_evidence_ids": ["E1"],
                        "citation_complete": True,
                    }
                ],
                "coverage": "complete",
                "critical_unsupported_claims": 0,
            }
        ],
    }

    validated = MODULE.validate_judgement(value, [(0, payload)])

    assert math.isclose(MODULE.semantic_score(validated["rollouts"][0], payload), 1.0)
    assert MODULE.judgement_counts(validated) == {
        "action:appropriate": 1,
        "citation_complete": 1,
        "coverage:complete": 1,
        "critical_unsupported_claims": 0,
        "facts": 1,
        "support:entailed": 1,
    }


def test_contradiction_caps_semantic_reward() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "杜宾批准了申请。", "evidence_ids": ["E1"]}],
    }
    row = {
        "rollout_index": 0,
        "action_appropriateness": "inappropriate",
        "facts": [
            {
                "fact_index": 0,
                "support": "contradicted",
                "checked_evidence_ids": ["E1"],
                "citation_complete": False,
            }
        ],
        "coverage": "none",
        "critical_unsupported_claims": 1,
    }

    assert MODULE.semantic_score(row, payload) <= -0.5


def test_one_unsupported_fact_cannot_be_hidden_by_supported_facts() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "事实甲", "evidence_ids": ["E1"]},
            {"fact": "事实乙", "evidence_ids": ["E2"]},
        ],
    }
    row = {
        "rollout_index": 0,
        "action_appropriateness": "appropriate",
        "facts": [
            {
                "fact_index": 0,
                "support": "entailed",
                "checked_evidence_ids": ["E1"],
                "citation_complete": True,
            },
            {
                "fact_index": 1,
                "support": "unsupported",
                "checked_evidence_ids": ["E2"],
                "citation_complete": False,
            },
        ],
        "coverage": "complete",
        "critical_unsupported_claims": 1,
    }

    assert MODULE.semantic_score(row, payload) <= 0.25


def test_checked_ids_must_equal_rollout_claim_ids() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1"]}],
    }
    value = {
        "protocol": MODULE.PROTOCOL_VERSION,
        "rollouts": [
            {
                "rollout_index": 0,
                "action_appropriateness": "appropriate",
                "facts": [
                    {
                        "fact_index": 0,
                        "support": "entailed",
                        "checked_evidence_ids": ["E2"],
                        "citation_complete": True,
                    }
                ],
                "coverage": "complete",
                "critical_unsupported_claims": 0,
            }
        ],
    }

    try:
        MODULE.validate_judgement(value, [(0, payload)])
    except MODULE.SemanticJudgeError as exc:
        assert "checked_evidence_ids_mismatch" in str(exc)
    else:
        raise AssertionError("judge was allowed to inspect an uncited passage")


def test_checked_ids_may_return_same_set_in_different_order() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E2", "E1"]}],
    }
    value = {
        "protocol": MODULE.PROTOCOL_VERSION,
        "rollouts": [
            {
                "rollout_index": 0,
                "action_appropriateness": "appropriate",
                "facts": [
                    {
                        "fact_index": 0,
                        "support": "entailed",
                        "checked_evidence_ids": ["E1", "E2"],
                        "citation_complete": True,
                    }
                ],
                "coverage": "complete",
                "critical_unsupported_claims": 0,
            }
        ],
    }

    assert MODULE.validate_judgement(value, [(0, payload)]) == value


def test_cache_reuses_identical_group_without_api(tmp_path: Path) -> None:
    payload = {
        "next_action": "abstain",
        "reason": "证据不足。",
    }
    judgement = {
        "protocol": MODULE.PROTOCOL_VERSION,
        "rollouts": [
            {
                "rollout_index": 0,
                "action_appropriateness": "appropriate",
                "facts": [],
                "coverage": "not_applicable",
                "critical_unsupported_claims": 0,
            }
        ],
    }
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        max_attempts=1,
    )
    calls = 0

    def fake_request(messages):
        nonlocal calls
        calls += 1
        return json.dumps(judgement), {"model": "glm-5.3"}

    judge._request = fake_request
    completion = [[{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]]

    assert judge.score_group(PROMPT, completion) == [1.0]
    assert judge.score_group(PROMPT, completion) == [1.0]
    assert calls == 1
    cached = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8"))
    assert cached["context"]["question"] == "谁批准了申请？"
    assert cached["rollouts"][0]["output"] == payload


def test_invalid_rollout_skips_teacher_call(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
    )

    assert judge.score_group(PROMPT, ["not-json"]) == [0.0]
    assert not (tmp_path / "failures.jsonl").exists()


def test_semantic_gate_makes_invalid_rollout_worse_than_contradiction() -> None:
    assert MODULE.semantic_reward_gate([0.0, -0.5], [False, True]) == [-1.0, -0.5]


def test_gated_reward_marks_invalid_after_batch_scoring(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
    )
    calls = 0

    def fake_score_batch(completions, contexts):
        nonlocal calls
        calls += 1
        return [0.0]

    judge.score_batch = fake_score_batch
    reward = MODULE.make_glm_semantic_reward(judge, gate_invalid=True)

    assert reward(["not-json"], [PROMPT]) == [-1.0]
    assert calls == 1


def test_strict_judge_rejects_fenced_json_before_api(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        strict_json=True,
    )
    payload = {"next_action": "abstain", "reason": "证据不足。"}
    fenced = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"

    assert judge.score_group(PROMPT, [fenced]) == [0.0]
    assert judge.eligibility(PROMPT, [fenced]) == [False]
    assert not (tmp_path / "cache.jsonl").exists()


def test_gated_reward_keeps_provider_failure_neutral(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        max_attempts=1,
        max_consecutive_failures=2,
        strict_json=True,
    )
    judge._request = lambda messages: (_ for _ in ()).throw(
        MODULE.SemanticJudgeError("temporary")
    )
    payload = json.dumps(
        {"next_action": "abstain", "reason": "证据不足。"}, ensure_ascii=False
    )
    reward = MODULE.make_glm_semantic_reward(judge, gate_invalid=True)

    assert reward([payload], [PROMPT]) == [0.0]
    assert (tmp_path / "failures.jsonl").exists()


def test_content_filter_is_neutral_auditable_and_does_not_trip_breaker(
    tmp_path: Path,
) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        max_attempts=3,
        max_consecutive_failures=1,
        strict_json=True,
    )
    calls = 0

    def filtered(messages):
        nonlocal calls
        calls += 1
        raise MODULE.ContentFilteredSemanticJudgeError(
            'http_400:content_filter:{"error":{"code":"1301"}}'
        )

    judge._request = filtered
    payload = json.dumps(
        {"next_action": "abstain", "reason": "证据不足。"}, ensure_ascii=False
    )
    reward = MODULE.make_glm_semantic_reward(judge, gate_invalid=True)

    assert reward([payload], [PROMPT]) == [0.0]
    assert calls == 1
    assert judge.stats["content_filter"] == 1
    assert judge._consecutive_failures == 0
    failure = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["failure_kind"] == "content_filter"
    assert failure["neutral_reward"] is True


def test_bigmodel_1301_http_400_is_classified_as_content_filter(
    tmp_path: Path, monkeypatch
) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
    )
    body = b'{"contentFilter":[{"level":1}],"error":{"code":"1301"}}'
    error = urllib.error.HTTPError(
        judge.endpoint, 400, "Bad Request", hdrs=None, fp=io.BytesIO(body)
    )

    class FailingOpener:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr(MODULE.urllib.request, "build_opener", lambda *args: FailingOpener())

    with pytest.raises(MODULE.ContentFilteredSemanticJudgeError):
        judge._request([{"role": "user", "content": "test"}])


def test_zero_consecutive_failure_limit_never_aborts_transient_failures(
    tmp_path: Path,
) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        max_attempts=1,
        max_consecutive_failures=0,
        strict_json=True,
    )
    judge._request = lambda messages: (_ for _ in ()).throw(
        MODULE.SemanticJudgeError("temporary")
    )
    payload = json.dumps(
        {"next_action": "abstain", "reason": "证据不足。"}, ensure_ascii=False
    )

    for _ in range(4):
        assert judge.score_group(PROMPT, [payload]) == [0.0]

    assert judge._consecutive_failures == 4
    assert len((tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_unknown_evidence_id_is_not_sent_to_judge(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
    )
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E99"]}],
    }

    assert judge.score_group(PROMPT, [json.dumps(payload, ensure_ascii=False)]) == [0.0]
    assert not (tmp_path / "cache.jsonl").exists()


def test_incomplete_retrieve_schema_is_not_judge_eligible() -> None:
    context = MODULE.extract_judge_context(PROMPT)
    payload = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {"question": "还缺什么？"},
    }

    assert not MODULE.payload_is_judge_eligible(payload, context["evidence"])


def test_duplicate_facts_can_be_enabled_for_semantic_audit() -> None:
    context = MODULE.extract_judge_context(PROMPT)
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "阿米娅批准申请。", "evidence_ids": ["E1"]},
            {"fact": "阿米娅批准申请。", "evidence_ids": ["E1"]},
        ],
    }

    assert not MODULE.payload_is_judge_eligible(payload, context["evidence"])
    assert MODULE.payload_is_judge_eligible(
        payload, context["evidence"], allow_duplicate_facts=True
    )


def test_build_ssl_context_rejects_missing_explicit_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CA bundle does not exist"):
        MODULE.build_ssl_context(tmp_path / "missing.pem")


def test_build_ssl_context_accepts_system_bundle() -> None:
    context = MODULE.build_ssl_context("/etc/ssl/certs/ca-certificates.crt")

    assert context.verify_mode != 0
    assert context.check_hostname


def test_invalid_judge_response_is_auditable(tmp_path: Path) -> None:
    payload = {"next_action": "abstain", "reason": "证据不足。"}
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        max_attempts=1,
        max_consecutive_failures=2,
    )
    judge._request = lambda messages: (
        "not-json",
        {"finish_reason": "length", "usage": {"completion_tokens": 8192}},
    )

    assert judge.score_group(PROMPT, [json.dumps(payload, ensure_ascii=False)]) == [0.0]
    failure = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["last_response"]["content_excerpt"] == "not-json"
    assert failure["last_response"]["api"]["finish_reason"] == "length"


def test_prescore_rollouts_preserves_group_order(tmp_path: Path) -> None:
    judge = MODULE.GlmEvidenceJudge(
        endpoint="https://example.invalid",
        api_key="secret",
        model="glm-5.3",
        cache_path=tmp_path / "cache.jsonl",
        failures_path=tmp_path / "failures.jsonl",
        workers=2,
    )
    judge.score_group = lambda prompt, values: [float(len(prompt))] * len(values)

    assert judge.prescore_rollouts(["a", "abcd"], [[1, 2], [3]]) == [[1.0, 1.0], [4.0]]
