import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_grounding_audit.py"
SPEC = importlib.util.spec_from_file_location("summarize_grounding_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summary_excludes_protocol_failures_from_semantic_denominator() -> None:
    payload = {
        "results": [
            {
                "status": "ok",
                "judgement": {
                    "set_support": "partial",
                    "facts": [
                        {
                            "support": "partial",
                            "question_relevance": "direct",
                            "citation_complete": False,
                        }
                    ],
                    "critical_unsupported_claims": 0,
                    "context_sufficiency": "sufficient",
                    "action_appropriateness": "appropriate",
                },
            },
            {"status": "error", "error": "non_json"},
            {"status": "ineligible"},
        ]
    }
    result = MODULE.summarize(payload)
    assert result["rows"] == 3
    assert result["status"] == {"error": 1, "ineligible": 1, "ok": 1}
    assert result["semantic_denominator"] == 1
    assert result["fact_support"] == {"partial": 1}
    assert result["question_relevance"] == {"direct": 1}


def test_summary_reports_answer_level_risk_separately_from_fact_rate() -> None:
    payload = {
        "results": [
            {
                "status": "ok",
                "action": "answer_directly",
                "judgement": {
                    "set_support": "partial",
                    "facts": [
                        {
                            "support": "partial",
                            "question_relevance": "direct",
                            "citation_complete": False,
                        }
                    ],
                    "critical_unsupported_claims": 1,
                    "context_sufficiency": "sufficient",
                    "action_appropriateness": "inappropriate",
                },
            },
            {
                "status": "ok",
                "action": "abstain",
                "judgement": {
                    "set_support": "none",
                    "facts": [],
                    "critical_unsupported_claims": 0,
                    "context_sufficiency": "insufficient",
                    "action_appropriateness": "appropriate",
                },
            },
        ]
    }
    result = MODULE.summarize(payload)
    assert result["answer_level"]["denominator"] == 1
    assert result["answer_level"]["counts"]["any_nonentailed_fact"] == 1
    assert result["answer_level"]["counts"]["any_critical_unsupported_claim"] == 1
    assert result["answer_level"]["rates"]["complete_answer"] == 0.0
    assert result["complete_answers_per_valid_request"] == 0.0
