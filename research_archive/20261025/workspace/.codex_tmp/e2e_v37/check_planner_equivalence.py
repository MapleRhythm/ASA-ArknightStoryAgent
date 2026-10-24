"""Replay actual canonical planner method with recorded generator outputs."""
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_planner_capture as experiment

shared = experiment.shared


def main():
    from asa_arknight_story_agent.inference.generation.hypothesis_generation import HypothesisGenerationMixin
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    experiment.verify(pre)
    output = HERE / "planner_method_equivalence_v1.json"
    assert not output.exists()
    checks = []
    for source, item in zip(pre["rows"], gen["rows"], strict=True):
        if "processing" not in item:
            checks.append({"id": source["id"], "status": "not_replayed_generation_failure"})
            continue

        class RecordedGenerator:
            max_tokens = 256

            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                assert prompt == source["prompt"]
                assert kwargs == {
                    "max_tokens": 256, "temperature": .1, "top_p": .8, "repeat_penalty": 1.15,
                }
                return item["sanitized_raw_output"]

        class Pipeline(HypothesisGenerationMixin):
            generator = RecordedGenerator()

        pipeline = Pipeline()
        warnings = StringIO()
        with redirect_stdout(warnings):
            actual = pipeline.build_hypothesis(source["question"], "")
        assert pipeline.generator.calls == 1
        assert asdict(actual) == item["processing"]["control"]
        checks.append({
            "id": source["id"], "status": "canonical_method_matches",
            "original_generation_status": item["status"],
            "normalization_status": item["processing"]["normalization_status"],
            "canonical_method_warning": warnings.getvalue(),
        })
    shared.save(output, {
        "version": "canonical_planner_method_replay_v37",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "production_changed": False, "training_ready": False,
        "checks": checks, "new_model_calls": 0,
        "basis_sha256": {str(p): shared.sha(p) for p in (
            Path(__file__), HERE / "preflight.json", HERE / "generations.json",
            experiment.SRC / "asa_arknight_story_agent/inference/generation/hypothesis_generation.py",
        )},
        "limitation": "Method equivalence does not imply entity correctness, retrieval quality, or answer accuracy.",
    })
    print({"replayed": sum(r["status"] == "canonical_method_matches" for r in checks)})


if __name__ == "__main__":
    main()
