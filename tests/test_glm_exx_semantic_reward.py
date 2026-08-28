import importlib.util
import json
import math
import sys
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


def test_build_ssl_context_rejects_missing_explicit_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CA bundle does not exist"):
        MODULE.build_ssl_context(tmp_path / "missing.pem")


def test_build_ssl_context_accepts_system_bundle() -> None:
    context = MODULE.build_ssl_context("/etc/ssl/certs/ca-certificates.crt")

    assert context.verify_mode != 0
    assert context.check_hostname
