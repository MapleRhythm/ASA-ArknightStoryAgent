import re

import pytest

from build_dialogue_layout_matrix import transform


def parse(text):
    matches = list(re.finditer(r"(?m)^\[(E\d+)\][ \t]*", text))
    return {m[1]: text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def test_restores_only_original_whitespace_and_preserves_ids():
    row = {
        "id": "test", "system": "S", "user": "正文:\n[E1] 甲：我说的是乙。 乙：不对。",
        "assistant_prefix": "{", "evidence_brief": "[E1] 甲：我说的是乙。 乙：不对。",
        "prompt_evidence": [{"text": "甲：我说的是乙。\n乙：不对。"}],
        "packing_trace": [{"status": "included", "evidence_id": "E1", "input_index": 0}],
    }
    render = lambda s, u, **kw: s + u + kw["assistant_prefix"]
    result = transform(row, render=render, parse_blocks=parse, original_text=lambda item: item["text"])
    assert result["evidence_brief"] == "[E1] 甲：我说的是乙。\n乙：不对。"
    assert result["normalized_visible_content_unchanged"]
    assert "\n乙：" not in row["evidence_brief"]
    row["prompt_evidence"][0]["text"] += "\n新断言。"
    with pytest.raises(ValueError, match="beyond whitespace"):
        transform(row, render=render, parse_blocks=parse, original_text=lambda item: item["text"])
