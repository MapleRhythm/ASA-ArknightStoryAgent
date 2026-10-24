from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    log = (HERE / "run_v1.log").read_text()
    assert "invalid literal for int() with base 10: 'GPU-90144416-2626-05e7-0b89-a4e45559ddf9'" in log
    assert "pydantic_core._pydantic_core.ValidationError" in log
    assert not (HERE / "calls.progress.jsonl").exists()
    assert not (HERE / "generations.json").exists()
    idle = json.loads((HERE / "idle_samples.json").read_text())
    assert len(idle["samples"]) == 13 and idle["elapsed_seconds"] >= 60
    source = (HERE / "run_planner_capture.py").read_text()
    assert source.index("llm, adapter = runner._ensure_engine()") < source.index('with (HERE / "calls.progress.jsonl").open("x"')
    with (HERE / "startup_failure_v1.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "version": "planner_capture_startup_failure_v37",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_calls": 58, "actual_generation_calls": 0,
            "status": "engine_config_failed_before_generation",
            "cause": "Installed vLLM parses CUDA_VISIBLE_DEVICES entries as integer IDs; UUID entry fails before engine startup.",
            "original_run_preserved": True, "production_changed": False, "training_ready": False,
            "terminal_observed_exit_code": 1,
            "basis_sha256": {str(p): sha(p) for p in (
                HERE / "preflight.json", HERE / "run_v1.log", HERE / "idle_samples.json",
                HERE / "run_planner_capture.py", Path(__file__),
            )},
            "next_attempt": "New directory, same 58 registered calls and protocol, verified numeric physical GPU2 mapping and a fresh 60-second guard.",
        }, stream, ensure_ascii=False, indent=2)
    print({"status": "startup_failure_preserved", "actual_generation_calls": 0})


if __name__ == "__main__":
    main()
