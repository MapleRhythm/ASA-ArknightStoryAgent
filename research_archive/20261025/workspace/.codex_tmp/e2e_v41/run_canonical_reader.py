"""Six frozen retrieval pairs, two seeds, actual canonical first-round reader.

This is a staged reader diagnostic, not a full end-to-end run. No new retrieval,
planner, second round, evidence alteration, source-label injection, or promotion.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V40 = HERE.parent / "routing_frozen_retrieval_20260910.v40"
sys.path.insert(0, str(V40))
import run_retrieval_probe as retrieval
shared = retrieval.shared


def runtime_imports():
    sys.path.insert(0, str(SRC))
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    return {**retrieval.imports(), **locals()}


def pipeline_for(pre, helpers, generator):
    pipeline, _ = retrieval.make_pipeline(pre, helpers, retriever=None)
    pipeline.generator = generator
    assert pipeline.self_consistency_samples == 1
    assert pipeline.max_retrieval_rounds == 2
    assert pipeline.conclusion_prompt_mode == "minimal"
    assert pipeline.answer_grounding_mode == "quote"
    return pipeline


def render(pipeline, helpers, source, result):
    hypothesis = helpers["HypothesisDocument"](**source)
    evidence = deepcopy(result["evidence"])
    packed = pipeline.prepare_prompt_evidence(result["question"], hypothesis, evidence)
    assert packed == result["prompt_evidence"]
    trace = []
    prompt, visible = helpers["build_conclusion_prompt"](
        result["question"], hypothesis, evidence, [deepcopy(result["trace"])], 1,
        pipeline.max_retrieval_rounds, pipeline.prompt_evidence_top_k,
        prompt_evidence=packed,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
        prompt_mode=pipeline.conclusion_prompt_mode,
        grounding_mode=pipeline.answer_grounding_mode,
        evidence_packing_trace=trace,
    )
    return {"prompt": prompt, "visible_evidence": visible, "packing_trace": trace}


def prepare():
    assert not (HERE / "preflight.json").exists()
    helpers = runtime_imports()
    old = shared.load(V27 / "preflight.json")
    v40pre = shared.load(V40 / "preflight.json")
    retrieval.verify(v40pre)
    data = shared.load(V40 / "retrieval.json")
    checkpoint = shared.load(V40 / "checkpoint_v40.json")
    assert checkpoint["assistant_visible_source_review_complete"]
    assert data["llm_generation_calls"] == 0 and len(data["conditions"]) == 12
    pre = {
        key: deepcopy(old[key]) for key in
        ("base_model", "adapter", "tokenizer", "runtime", "generator")
    }
    pre.update({key: deepcopy(v40pre[key]) for key in ("query_config", "pipeline_config")})
    assert "soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3" in pre["adapter"]
    pre.update(
        version="routing_frozen_canonical_first_round_reader_v41",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        production_changed=False, training_ready=False,
        verified_nvidia_mapping=retrieval.numeric_mapping(),
        protocol=[
            "All six v40 case pairs, two predefined paired seeds, alternating condition order: 24 first-round attempts.",
            "Use v40's exact retrieved objects and corresponding control/candidate hypotheses; no fresh planner or retrieval.",
            "Call actual canonical pipeline.generate_conclusion at round1/2, preserving configured packing and validators.",
            "Production 4B SODA LoRA, legacy qwen3_nothink chat wrapper, original brace-prefill semantics; no native template.",
            "Actual first-round generation: 2048 tokens, temperature .1, top_p .8, repetition 1.05; context 12288.",
            "Use the frozen runtime 12000-character evidence budget, NOT v40's standardized 5000-character preview.",
            "Use real prepared evidence objects in quote and legacy validation; never pass an empty evidence placeholder.",
            "No custom E-ID repair, no quote relaxation, no output retry, no direct-answer fallback or second round at round1.",
            "Canonical parsing/truncated-recovery behavior, if reached, stays unchanged and is reported; preserve finish_reason length.",
            "Preserve context overflow without truncating or reducing token budgets; each of 24 attempts remains in the record.",
            "017/023 identical-input controls quantify generation variation; same seed is not a determinism guarantee.",
            "049 retains v40 query-count confound (15 control / 16 candidate); all retrieval differences are frozen, not rerun.",
            "Assistant observations/reviews/old answers are never input to the model; observations do not define human gold.",
            "Verify numeric GPU2 UUID/PCI mapping and collect fresh continuous >=60-second idle window inside the existing lock.",
            "No training, release-source changes, default model switches, service restarts, commit or push.",
        ],
        conditions=[], pairs=[],
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    pipeline = pipeline_for(pre, helpers, retrieval.NoGeneration())
    indexed = {(r["id"], r["condition"]): r for r in data["conditions"]}
    for row in v40pre["rows"]:
        for kind in ("control", "candidate"):
            result = {**indexed[(row["id"], kind)], "question": row["question"]}
            item = {
                "id": row["id"], "condition": kind, "question": row["question"],
                "hypothesis": row[kind], "no_intervention_control": row["no_intervention_control"],
                **render(pipeline, helpers, row[kind], result),
            }
            ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
            item.update(
                prompt_sha256=shared.digest(item["prompt"]),
                token_ids_sha256=shared.digest(json.dumps(ids)),
                prompt_tokens=len(ids),
                context_fits=len(ids) + 2048 <= pre["runtime"]["max_model_len"],
                v40_preview_identical=item["prompt"] == result["preview_prompt"],
            )
            assert item["visible_evidence"] in item["prompt"]
            assert "<think>" not in item["prompt"].rsplit("<|im_start|>assistant", 1)[-1]
            pre["conditions"].append(item)
    for repeat in range(2):
        for index, row in enumerate(v40pre["rows"]):
            pre["pairs"].append({
                "id": row["id"], "repeat": repeat, "pair_id": f"{row['id']}:seed{repeat}",
                "sampling_seed": int(shared.digest(f"canonical-reader-v41:{repeat}:{row['id']}")[:8], 16) % (2**31),
                "condition_order": ["control", "candidate"] if (repeat + index) % 2 == 0 else ["candidate", "control"],
            })
    paths = [
        Path(__file__), V40 / "run_retrieval_probe.py", V27 / "run_reader_router.py",
        V27 / "selector_protocol.py", V27 / "runtime.json", V40 / "preflight.json",
        V40 / "retrieval.json", V40 / "checkpoint_v40.json",
        ROOT / "indexes/arknights_story/operator_aliases.json", *sorted(SRC.rglob("*.py")),
    ]
    for folder in (pre["base_model"], pre["adapter"]):
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model", ".tiktoken"}:
                assert str(path) in old["sha256"]
                assert shared.sha(path) == old["sha256"][str(path)]
                paths.append(path)
    pre["sha256"] = {str(p): shared.sha(p) for p in paths}
    shared.save(HERE / "preflight.json", pre)
    print({
        "planned_attempts": 24,
        "longest_input_tokens": max(r["prompt_tokens"] for r in pre["conditions"]),
        "context_overflow_conditions": [f"{r['id']}:{r['condition']}" for r in pre["conditions"] if not r["context_fits"]],
        "preview_identical_conditions": sum(r["v40_preview_identical"] for r in pre["conditions"]),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
    }, flush=True)


def verify(pre):
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())


def sources():
    return {(r["id"], r["condition"]): r for r in shared.load(V40 / "retrieval.json")["conditions"]}


def invoke(pipeline, helpers, item, source):
    hypothesis = helpers["HypothesisDocument"](**item["hypothesis"])
    trace = [deepcopy(source["trace"])]
    result = pipeline.generate_conclusion(
        item["question"], hypothesis, deepcopy(source["evidence"]), trace, current_round=1,
    )
    return asdict(result), trace


def generate_all():
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    assert not (HERE / "generations.progress.jsonl").exists()
    helpers = runtime_imports()
    engine, config = pre["runtime"], pre["generator"]
    runner = helpers["VllmRunner"](
        base_model_path=Path(pre["base_model"]), lora_path=Path(pre["adapter"]),
        tensor_parallel_size=engine["tensor_parallel_size"],
        gpu_memory_utilization=engine["gpu_memory_utilization"], max_model_len=engine["max_model_len"],
        max_tokens=config["max_tokens"], temperature=config["temperature"], top_p=config["top_p"],
        repeat_penalty=config["repeat_penalty"], dtype=engine["dtype"],
        max_num_batched_tokens=engine["max_num_batched_tokens"], enforce_eager=engine["enforce_eager"],
    )
    llm, adapter = runner._ensure_engine()
    assert adapter is not None
    tokenizer = llm.get_tokenizer()
    original_llm_generate, original_runner_generate = llm.generate, runner.generate
    state, calls = {}, []

    def dispatch(prompts, params, **kwargs):
        assert prompts == [state["item"]["prompt"]] and kwargs["lora_request"] is adapter
        assert params.max_tokens == 2048 and params.temperature == 0.1 and params.top_p == 0.8
        assert params.repetition_penalty == 1.05
        params.seed = state["pair"]["sampling_seed"]
        call = {
            "pair_id": state["pair"]["pair_id"], "condition": state["item"]["condition"],
            "prompt_sha256": shared.digest(prompts[0]), "seed": params.seed,
            "max_tokens": params.max_tokens, "temperature": params.temperature, "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty, "lora_id": adapter.lora_int_id,
        }
        calls.append(call)
        output = original_llm_generate(prompts, params, **kwargs)
        assert len(output) == len(output[0].outputs) == 1
        actual = output[0]
        ids = tokenizer.encode(prompts[0], add_special_tokens=True)
        assert list(actual.prompt_token_ids) == ids
        echoed = getattr(actual, "lora_request", None)
        assert echoed is not None and echoed.lora_int_id == adapter.lora_int_id
        completion = actual.outputs[0]
        call.update(
            token_ids_sha256=shared.digest(json.dumps(ids)), prompt_tokens=len(ids),
            returned_lora_id=echoed.lora_int_id, finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason, generated_tokens=len(completion.token_ids),
            raw_output_with_special_tokens=completion.text,
        )
        return output

    def record_generate(prompt, **kwargs):
        assert prompt == state["item"]["prompt"]
        assert not state["runner_requests"], "No retry or final direct fallback at round1"
        state["runner_requests"].append({"prompt_sha256": shared.digest(prompt), "kwargs": kwargs})
        text = original_runner_generate(prompt, **kwargs)
        state["sanitized_raw_output"] = text
        return text

    llm.generate, runner.generate = dispatch, record_generate
    pipeline = pipeline_for(pre, helpers, runner)
    indexed = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    frozen = sources()
    attempts = []
    with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
        for pair in pre["pairs"]:
            for kind in pair["condition_order"]:
                item = indexed[(pair["id"], kind)]
                state.clear()
                state.update(item=item, pair=pair, runner_requests=[])
                before, start = len(calls), time.monotonic()
                result = {"pair_id": pair["pair_id"], "id": pair["id"], "condition": kind, "status": "error"}
                try:
                    conclusion, trace = invoke(pipeline, helpers, item, frozen[(pair["id"], kind)])
                    result.update(status="ok", conclusion=conclusion, retrieval_trace=trace)
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                result.update(
                    calls=deepcopy(calls[before:]), runner_requests=deepcopy(state["runner_requests"]),
                    sanitized_raw_output=state.get("sanitized_raw_output"),
                    runtime_diagnostics=deepcopy(runner.last_generation_diagnostics),
                    seconds=time.monotonic() - start,
                )
                assert len(result["calls"]) <= 1
                attempts.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print({
                    "completed": len(attempts), "total": 24, "pair": result["pair_id"], "condition": kind,
                    "status": result["status"], "action": result.get("conclusion", {}).get("next_action"),
                    "finish": result["calls"][0].get("finish_reason") if result["calls"] else None,
                }, flush=True)
    verify(pre)
    shared.save(HERE / "generations.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "attempts": attempts, "actual_call_count": len(calls),
        "sources_and_models_unchanged": True, "production_changed": False, "training_ready": False,
    })
    print({"saved": len(attempts), "calls": len(calls)}, flush=True)
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
        print("Explicit engine shutdown requested.", flush=True)


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    pre = shared.load(HERE / "preflight.json")
    assert retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    verify(pre)
    assert not (HERE / "idle_samples.json").exists()
    shared.HERE, shared.run = HERE, generate_all
    shared.idle_then_run()


class ReplayGenerator:
    def __init__(self, config, item, result):
        self.max_tokens = config["max_tokens"]
        self.repeat_penalty = config["repeat_penalty"]
        self.item, self.result, self.calls = item, result, 0

    def generate(self, prompt, **kwargs):
        assert self.calls == 0 and prompt == self.item["prompt"]
        assert kwargs == self.result["runner_requests"][0]["kwargs"]
        self.calls += 1
        raw = self.result["sanitized_raw_output"]
        if raw is None:
            raise ValueError("Replay of recorded generation failure")
        return raw


def check():
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    verify(pre)
    assert gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(line) for line in (HERE / "generations.progress.jsonl").read_text().splitlines()]
    assert journal == gen["attempts"] and len(journal) == 24
    indexed = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    frozen, helpers = sources(), runtime_imports()
    expected = [(p, k) for p in pre["pairs"] for k in p["condition_order"]]
    summary = {kind: {"status": Counter(), "action": Counter(), "finish": Counter()} for kind in ("control", "candidate")}
    method_replays = 0
    for result, (pair, kind) in zip(journal, expected, strict=True):
        assert (result["pair_id"], result["condition"]) == (pair["pair_id"], kind)
        item = indexed[(pair["id"], kind)]
        summary[kind]["status"].update([result["status"]])
        if result["calls"]:
            call = result["calls"][0]
            assert call["seed"] == pair["sampling_seed"] and call["max_tokens"] == 2048
            assert call["prompt_sha256"] == item["prompt_sha256"]
            if "token_ids_sha256" in call:
                assert call["token_ids_sha256"] == item["token_ids_sha256"]
                assert call["prompt_tokens"] == item["prompt_tokens"]
                assert call["lora_id"] == call["returned_lora_id"] == 1
                assert call["generated_tokens"] <= 2048
            summary[kind]["finish"].update([call.get("finish_reason", "generation_error")])
        if result["status"] == "ok":
            replay_gen = ReplayGenerator(pre["generator"], item, result)
            pipeline = pipeline_for(pre, helpers, replay_gen)
            conclusion, trace = invoke(pipeline, helpers, item, frozen[(pair["id"], kind)])
            assert conclusion == result["conclusion"] and trace == result["retrieval_trace"]
            assert replay_gen.calls == 1
            method_replays += 1
            summary[kind]["action"].update([conclusion["next_action"]])
    assert sum(len(r["calls"]) for r in journal) == gen["actual_call_count"]
    pairs = []
    for pair in pre["pairs"]:
        a, b = [next(r for r in journal if r["pair_id"] == pair["pair_id"] and r["condition"] == kind)
                for kind in ("control", "candidate")]
        ca, cb = [r.get("conclusion", {}) for r in (a, b)]
        ia, ib = [indexed[(pair["id"], kind)] for kind in ("control", "candidate")]
        pairs.append({
            "pair_id": pair["pair_id"], "same_input_prompt": ia["prompt"] == ib["prompt"],
            "same_visible_evidence": ia["visible_evidence"] == ib["visible_evidence"],
            "same_raw": a["sanitized_raw_output"] == b["sanitized_raw_output"],
            "same_answer": ca.get("answer") == cb.get("answer"),
            "same_action": ca.get("next_action") == cb.get("next_action"),
        })
    shared.save(HERE / "postflight_v1.json", {
        "version": "canonical_first_round_reader_postflight_v41",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "attempts": 24, "actual_calls": gen["actual_call_count"], "canonical_method_replays": method_replays,
        "summary_not_accuracy": summary, "paired_comparisons_not_accuracy": pairs,
        "actual_prompt_tokens_seed_lora_verified": True, "real_prepared_evidence_grounding_replayed": True,
        "production_changed": False, "training_ready": False, "assistant_source_review_complete": False,
    })
    print({"attempts": 24, "calls": gen["actual_call_count"], "canonical_replays": method_replays,
           "summary_not_accuracy": summary, "pairs": pairs}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_canonical_reader.py prepare|run|check")
    operations[sys.argv[1]]()
