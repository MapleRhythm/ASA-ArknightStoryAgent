from copy import deepcopy
import pytest
from quote_pointer_protocol_v1 import OLD_SCHEMA_REF, OLD_COPY_RULE
from run_quote_pointer8b_v1 import build_row, check_row


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "|".join(m["content"] for m in messages) + "|<think>\n\n</think>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def fixture():
    from build_fact_coverage_pairs_v1 import digest
    evidence = "[E1] 甲：是乙说的。\n[E2] 乙：我的确这样说过。"
    original = {
        "parent_record_id": "p", "id": "p::original_prompt_v1", "template": "runtime_raw",
        "question": "是谁说的？", "system": "sys", "hypothesis": {"question": "是谁说的？"},
        "user": "question: 是谁说的？\nhypothesis: preserved\nevidence_brief:\n" + evidence
                + "\nminirag_hints: unchanged\n" + OLD_SCHEMA_REF + "\nrules: " + OLD_COPY_RULE,
        "evidence_brief": evidence, "evidence_sha256": digest(evidence),
        "current_round": 1, "max_retrieval_rounds": 2, "sampling_seed": 17,
        "prompt_evidence": [{"text": evidence}],
    }
    tokenizer = Tokenizer()
    return original, tokenizer, build_row(original, tokenizer)


def test_exact_expected_protocol_delta():
    original, tokenizer, row = fixture()
    assert check_row(row, original, tokenizer)["source_pointers"] == 2
    assert row["system"] == original["system"] and row["evidence_brief"] == original["evidence_brief"]


@pytest.mark.parametrize("key,value", [
    ("sampling_seed", 18), ("evidence_brief", "changed"), ("user", "gold"),
    ("assistant_prefix", "{"), ("hypothesis", {}), ("quote_catalog", []),
])
def test_changed_inputs_rejected(key, value):
    original, tokenizer, row = fixture()
    row[key] = value
    with pytest.raises(AssertionError):
        check_row(row, original, tokenizer)


def test_overflow_is_not_truncated_or_dropped():
    original, tokenizer, row = fixture()
    tokenizer.encode = lambda *a, **kw: list(range(10241))
    with pytest.raises(AssertionError, match="do not truncate"):
        check_row(row, original, tokenizer)
