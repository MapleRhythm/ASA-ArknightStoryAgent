from types import SimpleNamespace

from judge_visible_answers_v2 import accepted_reuse, expected_key, unavailable
from test_judge_visible_answers import row, verdict


def args():
    return SimpleNamespace(model="model-x", max_tokens=100, endpoint="https://example.test")


def prior(record, config):
    return {
        "id": record["id"], "status": "ok", "input_key": expected_key(record, config),
        "requested_model": config.model, "response_model": config.model,
        "finish_reason": "stop", "judgement": verdict(),
    }


def test_reuses_only_identical_audit_request():
    record, config = row(), args()
    old = prior(record, config)
    assert accepted_reuse(record, config, old)["reused_exact_request"] is True
    assert "reused_exact_request" not in old
    record["grounded"]["answer"] += "另一个断言。"
    assert accepted_reuse(record, config, old) is None


def test_rejects_response_model_mismatch():
    record, config = row(), args()
    old = prior(record, config)
    old["response_model"] = "unexpected-route"
    assert accepted_reuse(record, config, old) is None


def test_pipeline_error_is_retained_without_invented_judgement():
    result = unavailable({"id": "bad", "status": "parse_or_validation_error", "error": "bad JSON"})
    assert result["id"] == "bad"
    assert result["denominator_retained"]
    assert "judgement" not in result
