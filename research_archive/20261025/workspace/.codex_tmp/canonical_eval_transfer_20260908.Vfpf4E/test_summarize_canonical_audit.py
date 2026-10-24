from summarize_canonical_audit import summarize


def test_full_denominator_and_question_macro_do_not_hide_failures():
    rows = [
        {"question": "问题？", "action": "answer_directly", "label": "supported_complete", "context_sufficiency": "sufficient"},
        {"question": "问题?", "action": "answer_directly", "label": "unsupported", "context_sufficiency": "sufficient"},
        {"question": "另一题", "action": "pipeline_error", "label": "not_auditable_pipeline_error", "context_sufficiency": "not_judged"},
    ]
    result = summarize(rows)
    assert result["records"] == 3
    assert result["complete_yield_all_records"] == 1 / 3
    assert result["complete_fraction_among_shown_answers"] == 1 / 2
    assert result["distinct_normalized_questions"] == 2
    assert result["macro_question_mean_complete_fraction"] == 1 / 4
