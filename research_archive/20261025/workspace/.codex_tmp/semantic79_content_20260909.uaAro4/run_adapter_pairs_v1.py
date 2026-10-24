"""Per-request adapter/seed override around the existing verified vLLM harness."""
import hashlib
import json
from pathlib import Path
import sys


def paired_dispatch(original, rows, calls):
    def generate(prompts, params, **kwargs):
        assert len(calls) < len(rows), "Unexpected extra generation"
        row = rows[len(calls)]
        assert prompts == [row["runtime_prompt"]], "Generation input/order differs"
        assert type(row["sampling_seed"]) is int and type(row["use_adapter"]) is bool
        configured = kwargs.get("lora_request")
        assert configured is not None, "The shared engine must use the configured adapter tokenizer"
        effective = configured if row["use_adapter"] else None
        kwargs["lora_request"] = effective
        params.seed = row["sampling_seed"]
        receipt = {
            "id": row["id"], "seed": params.seed, "adapter_active_at_call": effective is not None,
            "configured_lora_id": getattr(configured, "lora_int_id", None),
            "effective_lora_id": getattr(effective, "lora_int_id", None),
            "prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
            "status": "requested",
        }
        calls.append(receipt)
        try:
            result = original(prompts, params, **kwargs)
            receipt["status"] = "returned"
            receipt["returned_lora_echo"] = [
                {
                    "available": hasattr(output, "lora_request"),
                    "lora_id": getattr(getattr(output, "lora_request", None), "lora_int_id", None),
                }
                for output in result
            ]
            return result
        except Exception as exc:
            receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
    return generate


def main():
    run = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(run))
    import run_canonical_vllm_eval_v3 as harness
    source = Path(sys.argv[sys.argv.index("--source-root") + 1])
    matrix_path = Path(sys.argv[sys.argv.index("--input") + 1])
    folder = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["rows"]
    assert matrix["evaluation_only"] and len(rows) == 12
    assert len({r["id"] for r in rows}) == 12
    groups = {}
    for row in rows:
        groups.setdefault(row["parent_record_id"], []).append(row)
    assert len(groups) == 6
    for pair in groups.values():
        assert len(pair) == 2 and pair[0]["sampling_seed"] == pair[1]["sampling_seed"]
        assert pair[0]["runtime_prompt"] == pair[1]["runtime_prompt"]
        assert {row["use_adapter"] for row in pair} == {False, True}
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    real_ensure = VllmRunner._ensure_engine
    calls = []

    def ensure_overridden(self):
        engine, adapter = real_ensure(self)
        assert adapter is not None
        if not getattr(engine, "_asa_adapter_pairs_override", False):
            engine.generate = paired_dispatch(engine.generate, rows, calls)
            engine._asa_adapter_pairs_override = True
        return engine, adapter

    VllmRunner._ensure_engine = ensure_overridden
    receipt = {
        "evaluation_only": True, "production_changed": False, "completed": False,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "rows": calls,
        "metadata_clarification": "metadata.json describes the shared configured engine; this receipt records actual per-request adapter and seed.",
    }
    try:
        harness.main()
        predictions_path = folder / "predictions.json"
        predictions = json.loads(predictions_path.read_text())
        assert len(predictions) == len(calls) == len(rows)
        assert [r["id"] for r in calls] == [r["id"] for r in predictions]
        assert all(r["status"] == "returned" for r in calls)
        receipt["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        receipt["completed"] = True
    finally:
        VllmRunner._ensure_engine = real_ensure
        if folder.is_dir():
            with (folder / "adapter_seed_overrides_v1.json").open("x") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
