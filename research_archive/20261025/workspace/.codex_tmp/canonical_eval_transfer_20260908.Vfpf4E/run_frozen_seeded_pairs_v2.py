"""Generic explicit-seed wrapper around the verified frozen vLLM harness."""
import hashlib
import json
from pathlib import Path
import sys

from run_paired_vllm_retry_v1 import seeded_generate


def main():
    import run_canonical_vllm_eval_v3 as harness
    source = Path(sys.argv[sys.argv.index("--source-root") + 1])
    matrix_path = Path(sys.argv[sys.argv.index("--input") + 1])
    folder = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["rows"]
    assert matrix["evaluation_only"] and rows and len(rows) == len({r["id"] for r in rows})
    assert all(type(r["sampling_seed"]) is int for r in rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["parent_record_id"], []).append(row)
    assert all(len(pair) == 2 and pair[0]["sampling_seed"] == pair[1]["sampling_seed"] for pair in groups.values())
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    real_ensure = VllmRunner._ensure_engine
    calls = []

    def ensure_seeded(self):
        engine, adapter = real_ensure(self)
        if not getattr(engine, "_asa_paired_seed_override", False):
            engine.generate = seeded_generate(engine.generate, rows, calls)
            engine._asa_paired_seed_override = True
        return engine, adapter

    VllmRunner._ensure_engine = ensure_seeded
    receipt = {
        "explicit_sampling_override": "SamplingParams.seed equal within each pair",
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
