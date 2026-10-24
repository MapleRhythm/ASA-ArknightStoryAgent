import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_quote_render_ablation as runner
from build_quote_render_ablation import build_rows, split_runtime_prompt
from run_quote_render_ablation import render_input, text_sha


PROMPT = (
    "<|im_start|>system\n系统<|im_end|>\n"
    "<|im_start|>user\n问题\n[E1] 证据<|im_end|>\n"
    "<|im_start|>assistant\n{"
)


def row():
    system, user = split_runtime_prompt(PROMPT)
    return {
        "system": system, "user": user, "runtime_prompt": PROMPT,
        "runtime_prompt_sha256": text_sha(PROMPT), "evidence_brief": "[E1] 证据",
        "evidence_sha256": text_sha("[E1] 证据"), "template": "runtime_raw",
    }


def test_raw_prompt_preserves_runtime_prefill_without_tokenizer_template():
    assert render_input(row(), None) == PROMPT
    assert split_runtime_prompt(PROMPT) == ("系统", "问题\n[E1] 证据")


def test_unexpected_runtime_template_rejected():
    with pytest.raises(ValueError, match="unexpected runtime"):
        split_runtime_prompt(PROMPT[:-1])


@pytest.mark.parametrize("field", ["runtime_prompt", "evidence_brief"])
def test_stale_prompt_or_evidence_hash_rejected(field):
    value = row()
    value[field] += "污染"
    with pytest.raises(ValueError, match="hash mismatch"):
        render_input(value, None)


def test_canonical_template_has_no_gold_or_json_prefill():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "system", "content": "系统"},
                {"role": "user", "content": "问题\n[E1] 证据"},
            ]
            assert kwargs == {
                "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
            }
            return "canonical\n<think>\n\n</think>\n\n"
    value = row()
    value["template"] = "canonical_non_thinking"
    value["gold"] = "should never be sent"
    before = copy.deepcopy(value)
    assert render_input(value, Tokenizer()).endswith("</think>\n\n")
    assert value == before


def test_gpu_guard_rejects_occupied_target_without_launching_model(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-target")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="GPU-target, 42, someone-else/python, 494 MiB\n")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="compute process"):
        runner.assert_gpu_idle("GPU-target")
    assert len(calls) == 1


def test_gpu_guard_only_checks_selected_gpu_and_returns_snapshot(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-target")
    outputs = iter(["GPU-other, 42, someone-else/python, 494 MiB\n", "idle GPU snapshot"])
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs))
    )
    assert runner.assert_gpu_idle("GPU-target") == "idle GPU snapshot"


def test_gpu_guard_rejects_ambiguous_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="explicit GPU UUID"):
        runner.assert_gpu_idle("GPU-target")


def test_real_matrix_rebuild_is_content_bound_and_gold_free():
    # Small synthetic frozen audit, generated using the real runtime renderer.
    from audit_rendered_evidence_visibility import audit_case
    snapshot = {
        "id": "q-test", "question": "甲做了什么？", "gold_evidence_ids": ["a"],
        "stages": {"rerank": [{
            "id": "a", "clean_text": "甲帮助了乙。", "fusion_score": 1,
            "evidence_chain_text": "场景前缀。甲帮助了乙。",
        }]},
    }
    review = {
        "minimal_sufficient_evidence_ids": ["a"],
        "answer_facts": [{"fact": "机密审稿意见", "support_groups": [["a"]]}],
    }
    profile = {"mode": "quote", "top_k": 10, "per_doc": 1000, "total": 12000}
    audit = {"profiles": [profile], "variants": {}}
    prefix = "runtime_gpu_reranker_qwen35_4b/production/break"
    for rendering in ("runtime", "direct_only"):
        result = audit_case(
            snapshot, review, mode="quote", strategy="production",
            packing="break", top_k=10, per_doc=1000, total=12000, rendering=rendering,
        )
        audit["variants"][prefix + ("/direct_only" if rendering == "direct_only" else "")] = {
            "rows": [result]
        }
    matrix = build_rows([snapshot], audit)
    assert len(matrix) == 4
    assert {item["template"] for item in matrix} == {"runtime_raw", "canonical_non_thinking"}
    assert all("机密审稿意见" not in item["runtime_prompt"] for item in matrix)
    assert all("gold_evidence_ids" not in item for item in matrix)
    audit["variants"][prefix]["rows"][0]["prompt_sha256"] = "stale"
    with pytest.raises(ValueError, match="drifted"):
        build_rows([snapshot], audit)


def test_mechanical_audit_keeps_raw_and_validated_actions_separate():
    import json
    from analyze_quote_render_ablation import analyze_row
    value = row()
    value.update({
        "id": "q::runtime::runtime_raw", "source_id": "q",
        "question": "甲做了什么？", "renderer": "runtime",
    })
    snapshot = {
        "id": "q", "question": "甲做了什么？",
        "stages": {"rerank": [{"id": "a", "clean_text": "甲帮助了乙。", "fusion_score": 1}]},
    }
    prediction = {
        "input_row_sha256": text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True)),
        "actual_prompt_sha256": text_sha(PROMPT), "actual_prompt": PROMPT,
        "raw_output": '"next_action":"answer_directly","supported_facts":'
                      '[{"fact":"甲帮助了乙。","evidence_refs":[{"evidence_id":"E99","quote":"甲帮助了乙。"}]}],'
                      '"inferred_facts":[],"final_answer":"甲帮助了乙。"}',
        "finish_reason": "stop", "prompt_token_count": 100,
        "generated_token_count": 50, "generation_seconds": 1.0,
    }
    result = analyze_row(value, prediction, snapshot)
    assert result["normalized_strict_object"]
    assert result["raw_action"] == "answer_directly"
    assert result["validated"]["next_action"] == "retrieve_more"
    assert result["semantic_review"] == "pending"
    assert any("E99" in issue for issue in result["quote_issues"])
    prediction["input_row_sha256"] = "stale"
    with pytest.raises(ValueError, match="input row hash mismatch"):
        analyze_row(value, prediction, snapshot)


def summary_inputs():
    audit = {
        "matrix_sha256": "matrix", "predictions_sha256": "predictions",
        "rows": [{
            "id": "q", "template": "runtime_raw", "renderer": "runtime",
            "validated": {"next_action": "answer_directly"}, "finish_reason": "stop",
        }],
    }
    review = {
        "matrix_sha256": "matrix", "predictions_sha256": "predictions",
        "rows": [{"id": "q", "raw_content": "complete_supported", "validated_content": "uncertain"}],
    }
    return audit, review


def test_review_summary_does_not_count_uncertain_as_success():
    from summarize_quote_render_review import summarize
    audit, review = summary_inputs()
    result = summarize(audit, review)["runtime_raw/runtime"]
    assert result["strict_complete_supported"] == 0
    assert result["upper_bound_if_uncertain_accepted"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "action", "hash"])
def test_review_summary_rejects_inconsistent_review(problem):
    from summarize_quote_render_review import summarize
    audit, review = summary_inputs()
    if problem == "duplicate":
        review["rows"].append(copy.deepcopy(review["rows"][0]))
    elif problem == "action":
        review["rows"][0]["validated_content"] = "not_answered"
    else:
        review["predictions_sha256"] = "other"
    with pytest.raises(ValueError):
        summarize(audit, review)
