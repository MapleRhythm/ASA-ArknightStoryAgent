from copy import deepcopy
from types import SimpleNamespace

import pytest
from run_native8b_v1 import check_row, digest, idle_observation, wrap_dispatch


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "|".join(m["content"] for m in messages) + "|<think>\n\n</think>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def fixture_row():
    tokenizer = Tokenizer()
    original = {"parent_record_id": "p", "question": "q", "system": "sys", "user": "u evidence",
                "evidence_brief": "evidence", "evidence_sha256": digest("evidence"),
                "prompt_evidence": [{"text": "evidence"}], "hypothesis": {"question": "q"},
                "current_round": 1, "max_retrieval_rounds": 2, "sampling_seed": 17}
    row = deepcopy(original)
    row.update(id="candidate", template="runtime_raw", assistant_prefix="",
               native_messages=[{"role": "system", "content": "sys"},
                                {"role": "user", "content": "u evidence"}],
               native_template_enable_thinking=False)
    prompt = tokenizer.apply_chat_template(row["native_messages"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    row.update(runtime_prompt=prompt, runtime_prompt_sha256=digest(prompt))
    return row, original, tokenizer


def test_exact_native_input():
    row, original, tokenizer = fixture_row()
    assert check_row(row, original, tokenizer)["input_tokens"] == len(row["runtime_prompt"])


@pytest.mark.parametrize("key,value", [
    ("user", "different"), ("sampling_seed", 18),
    ("prompt_evidence", [{"text": "changed"}]), ("assistant_prefix", "{"),
])
def test_changed_inputs_rejected(key, value):
    row, original, tokenizer = fixture_row()
    row[key] = value
    with pytest.raises(AssertionError):
        check_row(row, original, tokenizer)


def test_context_overflow_not_truncated():
    row, original, tokenizer = fixture_row()
    tokenizer.encode = lambda *args, **kwargs: list(range(10241))
    with pytest.raises(AssertionError, match="do not truncate"):
        check_row(row, original, tokenizer)


@pytest.mark.parametrize("processes,stats,expected", [
    ("GPU-other, 123, python, 50", "GPU-test, 18, 24064, 0", True),
    ("GPU-test, 123, python, 50", "GPU-test, 18, 24064, 0", False),
    ("", "GPU-test, 18, 24064, 30", False),
    ("", "GPU-test, 900, 23100, 0", False),
])
def test_idle_checks(processes, stats, expected):
    assert idle_observation("GPU-test", processes, stats)["idle"] is expected


def test_dispatch_records_seed_and_no_adapter():
    row, _, _ = fixture_row()
    params = SimpleNamespace(max_tokens=2048, temperature=0.1, top_p=0.8,
                             repetition_penalty=1.05, seed=None)
    def real(prompts, actual, **kwargs):
        assert actual.seed == 17 and kwargs["lora_request"] is None
        return [SimpleNamespace(prompt_token_ids=[1, 2], lora_request=None,
                                outputs=[SimpleNamespace()])]
    receipts = []
    wrapped = wrap_dispatch(real, row, [1, 2], receipts)
    wrapped([row["runtime_prompt"]], params, lora_request=None)
    assert receipts[0]["status"] == "returned" and receipts[0]["seed"] == 17


def test_lora_dispatch_rejected():
    row, _, _ = fixture_row()
    wrapped = wrap_dispatch(lambda *a, **k: [], row, [], [])
    with pytest.raises(AssertionError, match="must not use LoRA"):
        wrapped([row["runtime_prompt"]], None, lora_request=object())


def test_token_mismatch_rejected():
    row, _, _ = fixture_row()
    params = SimpleNamespace(max_tokens=2048, temperature=0.1, top_p=0.8,
                             repetition_penalty=1.05, seed=None)
    wrapped = wrap_dispatch(lambda *a, **k: [
        SimpleNamespace(prompt_token_ids=[1], outputs=[object()])
    ], row, [1, 2], [])
    with pytest.raises(AssertionError, match="input truncation"):
        wrapped([row["runtime_prompt"]], params, lora_request=None)
