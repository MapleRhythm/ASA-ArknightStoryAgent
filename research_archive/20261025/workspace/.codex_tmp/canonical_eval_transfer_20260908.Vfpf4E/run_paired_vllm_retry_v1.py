"""Run frozen paired prompts with explicit per-case sampling seed.

Wraps the already-verified real-runner harness. Only SamplingParams.seed changes;
the override is explicit in each result, not represented as production behavior.
"""
import hashlib
import json
from pathlib import Path
import sys

def seeded_generate(original, rows, calls):
    def generate(prompts, params, **kwargs):
        assert len(calls) < len(rows), "Unexpected extra generation"
        row = rows[len(calls)]
        assert prompts == [row["runtime_prompt"]], "Generation order or input changed"
        params.seed = row["sampling_seed"]
        calls.append({
            "id": row["id"], "seed": params.seed,
            "prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
        })
        return original(prompts, params, **kwargs)
    return generate


def main():
    import run_canonical_vllm_eval_v3 as harness
    run = Path(__file__).resolve().parent
    source = run / "context_guard_candidate_bytes_v1/src"
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    matrix_path = run / "same_evidence_retry_paired68_v1.matrix.json"
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["rows"]
    assert len(rows) == 68 and matrix["evaluation_only"]
    real_ensure = VllmRunner._ensure_engine
    calls = []

    def ensure_seeded(self):
        engine, adapter = real_ensure(self)
        if not getattr(engine, "_asa_paired_seed_override", False):
            engine.generate = seeded_generate(engine.generate, rows, calls)
            engine._asa_paired_seed_override = True
        return engine, adapter

    VllmRunner._ensure_engine = ensure_seeded
    folder = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    receipt = {
        "explicit_sampling_override": "SamplingParams.seed per case, equal within a control/feedback pair",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "rows": calls, "evaluation_only": True, "production_sampling_changed": False,
    }
    try:
        harness.main()
        predictions = json.loads((folder / "predictions.json").read_text())
        assert len(predictions) == len(calls) == len(rows)
        assert [r["id"] for r in calls] == [r["id"] for r in predictions]
        receipt["predictions_sha256"] = hashlib.sha256((folder / "predictions.json").read_bytes()).hexdigest()
        receipt["completed"] = True
    finally:
        VllmRunner._ensure_engine = real_ensure
        if folder.is_dir():
            with (folder / "sampling_seed_override.json").open("x") as output:
                output.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
