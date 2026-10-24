"""Use the same configured4B harness with exact58 seeded LoRA dispatch capture."""
import json
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, RUN, SOURCE, load, sha
from build_reasoning_context_ablation_v1 import MATRIX
from run_fact_coverage_pairs_v1 import pair_dispatch
from verify_empty_quote_candidate_v1 import save


def main():
    sys.path.insert(0, str(RUN))
    import run_canonical_vllm_eval_v3 as harness
    assert Path(sys.argv[sys.argv.index("--input") + 1]) == MATRIX
    assert Path(sys.argv[sys.argv.index("--source-root") + 1]) == SOURCE
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    matrix = load(MATRIX)
    rows = matrix["rows"]
    assert matrix["evaluation_only"] and len(rows) == len({r["id"] for r in rows}) == 58
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    real_ensure, calls = VllmRunner._ensure_engine, []

    def ensure(self):
        engine, adapter = real_ensure(self)
        assert adapter is not None
        if not getattr(engine, "_asa_reasoning_context_dispatch", False):
            engine.generate = pair_dispatch(engine.generate, rows, calls)
            engine._asa_reasoning_context_dispatch = True
        return engine, adapter

    VllmRunner._ensure_engine = ensure
    receipt = {"evaluation_only": True, "training_ready": False, "completed": False,
               "matrix_sha256": sha(MATRIX), "wrapper_sha256": sha(Path(__file__)), "rows": calls}
    try:
        harness.main()
        predictions = load(output / "predictions.json")
        assert len(predictions) == len(calls) == 58
        assert [r["id"] for r in calls] == [r["id"] for r in predictions]
        assert all(r["status"] == "returned" and r["lora_id"] == 1 and r["returned_lora_echo"] == [1]
                   for r in calls)
        receipt.update(completed=True, predictions_sha256=sha(output / "predictions.json"))
    finally:
        VllmRunner._ensure_engine = real_ensure
        if output.is_dir():
            save(output / "actual_dispatch_v1.json", receipt)


if __name__ == "__main__":
    main()
