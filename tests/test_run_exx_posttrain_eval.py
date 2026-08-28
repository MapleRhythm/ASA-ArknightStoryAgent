from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exx_posttrain_eval.sh"


def test_posttrain_eval_requires_verified_training_artifacts_before_generation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    artifact_check = text.index('[[ ! -f "$model_dir/adapter_model.safetensors" ]]')
    saved_check = text.index('grep -Fq "SAVED $model_dir"')
    first_generation = text.index('"$python_bin" "$generator"')
    assert artifact_check < first_generation
    assert saved_check < first_generation


def test_posttrain_eval_uses_same_generator_for_sft_and_rlvr() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.count('"$python_bin" "$generator"') == 2
    assert '--adapter "$sft_adapter"' in text
    assert '--adapter "$model_dir"' in text
