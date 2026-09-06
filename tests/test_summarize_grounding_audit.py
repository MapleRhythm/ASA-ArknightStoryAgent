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
