"""Capture actual per-request seed and LoRA while using the existing harness."""
import hashlib
import json
from pathlib import Path
import sys


def pair_dispatch(original, rows, calls):
    def generate(prompts, params, **kwargs):
        assert len(calls) < len(rows), "unexpected extra call"
        row = rows[len(calls)]
        assert prompts == [row["runtime_prompt"]]
        assert type(row["sampling_seed"]) is int
        adapter = kwargs.get("lora_request")
        assert adapter is not None, "configured LoRA must remain enabled"
        params.seed = row["sampling_seed"]
        receipt = {
            "id": row["id"], "seed": params.seed,
            "lora_id": getattr(adapter, "lora_int_id", None),
            "prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
            "status": "requested",
        }
        calls.append(receipt)
        try:
            result = original(prompts, params, **kwargs)
            receipt.update(
                status="returned",
                returned_lora_echo=[
                    getattr(getattr(output, "lora_request", None), "lora_int_id", None)
                    for output in result
                ],
            )
            return result
        except Exception as exc:
            receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
    return generate


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import run_canonical_vllm_eval_v3 as harness
    source = Path(sys.argv[sys.argv.index("--source-root") + 1])
    matrix_path = Path(sys.argv[sys.argv.index("--input") + 1])
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["rows"]
    assert matrix["evaluation_only"] and len(rows) == 116
    assert len({row["id"] for row in rows}) == 116
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    real_ensure, calls = VllmRunner._ensure_engine, []

    def ensure(self):
        engine, adapter = real_ensure(self)
        assert adapter is not None
        if not getattr(engine, "_asa_fact_coverage_dispatch", False):
            engine.generate = pair_dispatch(engine.generate, rows, calls)
            engine._asa_fact_coverage_dispatch = True
        return engine, adapter

    VllmRunner._ensure_engine = ensure
    receipt = {
        "evaluation_only": True, "training_ready": False, "completed": False,
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": calls,
        "metadata_clarification": "The wrapper sets actual per-request seeds; harness metadata describes its original defaults.",
    }
    try:
        harness.main()
        predictions = output / "predictions.json"
        data = json.loads(predictions.read_text())
        assert len(data) == len(calls) == len(rows)
        assert [r["id"] for r in calls] == [r["id"] for r in data]
        assert all(r["status"] == "returned" for r in calls)
        receipt["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
        receipt["completed"] = True
    finally:
        VllmRunner._ensure_engine = real_ensure
        if output.is_dir():
            with (output / "actual_dispatch_v1.json").open("x") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
