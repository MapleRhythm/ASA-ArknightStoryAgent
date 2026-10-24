import copy

from build_native_exx_schema_probe import SCHEMA, transform


def test_schema_example_does_not_replace_any_source_content():
    row = {
        "id": "x", "system": "system", "user": "question and full evidence", "assistant_prefix": "",
        "evidence_brief": "[E1] 全部原文", "question": "question", "hypothesis": {},
    }
    before = copy.deepcopy(row)
    result = transform(row, lambda system, user, **kwargs: system + user)
    assert row == before
    assert result["user"] == before["user"] + "\n" + SCHEMA
    assert result["evidence_brief"] == before["evidence_brief"]
    assert result["assistant_prefix"] == before["assistant_prefix"]
    assert result["system"] == before["system"]
