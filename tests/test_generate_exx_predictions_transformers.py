import importlib.util
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_exx_predictions_transformers.py"
SPEC = importlib.util.spec_from_file_location("generate_exx_predictions_transformers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_chat_messages_preserves_system_and_full_user_prompt() -> None:
    row = {
        "system": "只输出 JSON。",
        "conversations": [
            {"from": "human", "value": "问题\n[E1]\n完整证据"},
            {"from": "gpt", "value": "ignored gold"},
        ],
    }

    assert MODULE.chat_messages(row) == [
        {"role": "system", "content": "只输出 JSON。"},
        {"role": "user", "content": "问题\n[E1]\n完整证据"},
    ]


def test_generation_hit_token_limit_distinguishes_eos_from_length_cutoff() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class Completion:
        def __init__(self, values: list[int]) -> None:
            self.values = values
            self.shape = (len(values),)

        def __getitem__(self, index: int) -> Scalar:
            return Scalar(self.values[index])

    assert MODULE.generation_hit_token_limit(
        Completion([10, 11, 12]), max_new_tokens=3, eos_token_id=99
    )
    assert not MODULE.generation_hit_token_limit(
        Completion([10, 11, 99]), max_new_tokens=3, eos_token_id=99
    )
    assert not MODULE.generation_hit_token_limit(
        Completion([10, 11]), max_new_tokens=3, eos_token_id=99
    )


def test_prediction_row_preserves_diagnostic_metadata() -> None:
    row = {
        "id": "q1::actual_prompt",
        "task_type": "grounded_action_generation",
        "system": "只输出 JSON。",
        "conversations": [{"from": "human", "value": "问题与证据"}],
        "diagnostic": {
            "source_id": "q1",
            "condition": "actual_prompt",
        },
    }

    result = MODULE.prediction_row(
        row=row,
        text='{"next_action":"retrieve_more"}',
        prompt_token_count=20,
        generated_token_count=8,
        hit_max_new_tokens=False,
    )

    assert result["diagnostic"] == row["diagnostic"]
    assert result["finish_reason"] == "stop"


def test_model_asset_manifest_hashes_top_level_checkpoint_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    manifest = MODULE.model_asset_manifest(tmp_path)

    assert len(manifest["aggregate_sha256"]) == 64
    assert {item["name"] for item in manifest["files"]} == {
        "config.json",
        "model.safetensors",
    }


def test_prediction_preserves_nontraining_state_provenance():
    row = {
        "id": "probe", "system": "system",
        "conversations": [{"from": "human", "value": "visible only"},
                          {"from": "gpt", "value": "hidden target"}],
        "meta": {"training_eligible": False, "requires_reaudit": True},
        "state_provenance": {"condition": "missing_candidate"},
    }
    result = MODULE.prediction_row(
        row=row, text="prediction", prompt_token_count=10,
        generated_token_count=4, hit_max_new_tokens=False,
    )
    assert result["meta"] == row["meta"]
    assert result["state_provenance"] == row["state_provenance"]
    assert len(result["input_row_sha256"]) == 64
    assert "hidden target" not in str(MODULE.chat_messages(row))


def test_gpu_preflight_rejects_foreign_compute_even_at_zero_utilization(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-target")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout="GPU-target, 123, other-user-python, 100 MiB\n"),
    )
    with pytest.raises(ValueError, match="has a compute process"):
        MODULE.assert_gpu_idle("GPU-target")


def test_gpu_preflight_checks_exact_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-target")
    with pytest.raises(ValueError, match="match the explicit GPU UUID"):
        MODULE.assert_gpu_idle("GPU-other")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="GPU-other, 123, busy, 100 MiB\n" if len(calls) == 1 else "idle")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert MODULE.assert_gpu_idle("GPU-target") == "idle"
    assert len(calls) == 2
