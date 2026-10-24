"""Complete all v41/v44 staged endpoints, retaining both canonical reader calls.

Ten first-round terminals carry unchanged. Fourteen unresolved states invoke
the actual round2 method, including its original final-direct fallback.
No fresh benchmark claim, planner, retrieval, custom retry, or production edit.
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
V44 = HERE.parent / "staged_round2_retrieval_20260910.v44"
sys.path.insert(0, str(V44))
import run_second_round_retrieval as staged
reader, shared, retrieval = staged.reader, staged.shared, staged.retrieval
V41 = staged.V41
PARAMS = {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.05}
DRY_RAW = '{"next_action":"abstain","final_answer":"现有证据不足以确认。"}'


def sources():
    return {(r["pair_id"], r["condition"]): r
            for r in shared.load(V44 / "retrieval.json")["records"]}


def invoke(pre, helpers, item, source, generator):
    pipeline = reader.pipeline_for(pre, helpers, generator)
    state = helpers["PipelineRunState"](**deepcopy(source["state_after_retrieval"]))
    hypothesis = helpers["HypothesisDocument"](**item["hypothesis"])
    conclusion = pipeline.generate_conclusion(
        item["question"], hypothesis, state.evidence, state.retrieval_trace, 2,
    )
    helpers["append_conclusion_to_step"](state.retrieval_trace[-1], conclusion)
    terminal = helpers["apply_terminal_conclusion"](
        state=state, conclusion=conclusion, round_index=2, max_retrieval_rounds=2,
    )
    assert terminal
    assert state.evidence == source["state_after_retrieval"]["evidence"]
    return {
        "conclusion": asdict(conclusion), "retrieval_trace": state.retrieval_trace,
        "final_answer": state.final_answer,
        "state_after_reader_sha256": staged.object_digest(asdict(state)),
    }


def planned_prompts(pre, helpers, item, source):
    from asa_arknight_story_agent.inference.generation.direct_answer_prompt import build_answer_prompt
    pipeline = reader.pipeline_for(pre, helpers, retrieval.NoGeneration())
    state = helpers["PipelineRunState"](**deepcopy(source["state_after_retrieval"]))
    hypothesis = helpers["HypothesisDocument"](**item["hypothesis"])
    preview = staged.round2_prompt(pre, helpers, state, hypothesis, item["question"])
    assert all(source[k] == value for k, value in preview.items())
    # The canonical final fallback receives the already prepared first-call
    # objects, and prepares them again. Do not substitute the original list.
    prepared_again = pipeline.prepare_prompt_evidence(
        item["question"], hypothesis, deepcopy(preview["prompt_evidence"]),
    )
    packing = []
    prompt, visible = build_answer_prompt(
        item["question"], hypothesis, preview["prompt_evidence"],
        prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
        prompt_evidence=prepared_again,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
        grounding_mode=pipeline.answer_grounding_mode,
        round_value=f"{pipeline.max_retrieval_rounds}/{pipeline.max_retrieval_rounds}",
        evidence_packing_trace=packing,
    )
    return [
        {"kind": "round2_conclusion", "prompt": preview["planned_canonical_round2_prompt"],
         "visible_evidence": preview["visible_evidence"], "packing": preview["evidence_packing"],
         "prompt_evidence": preview["prompt_evidence"]},
        {"kind": "canonical_final_direct_fallback", "prompt": prompt, "visible_evidence": visible,
         "packing": packing, "prompt_evidence": prepared_again},
    ]


class DryGenerator:
    def __init__(self, config, item):
        self.max_tokens, self.repeat_penalty = config["max_tokens"], config["repeat_penalty"]
        self.item, self.requests = item, []

    def generate(self, prompt, **kwargs):
        index = len(self.requests)
        assert index < 2 and prompt == self.item["planned_requests"][index]["prompt"]
        assert kwargs == PARAMS
        self.requests.append({"prompt_sha256": shared.digest(prompt), "kwargs": kwargs})
        return DRY_RAW


def prepare():
    assert not (HERE / "preflight.json").exists()
    old, pre44 = shared.load(V41 / "preflight.json"), shared.load(V44 / "preflight.json")
    reader.verify(old)
    reader.verify(pre44)
    closed = shared.load(V44 / "checkpoint_v44.json")
    assert closed["status_counts_not_accuracy"] == {
        "terminal_carried_unchanged": 10, "round2_retrieval_ok": 14,
    }
    pre = {key: deepcopy(old[key]) for key in (
        "base_model", "adapter", "tokenizer", "runtime", "generator",
        "query_config", "pipeline_config",
    )}
    pre.update(
        version="staged_canonical_round2_reader_endpoints_v45",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        production_changed=False, training_ready=False,
        verified_nvidia_mapping=retrieval.numeric_mapping(),
        protocol=[
            "Complete the exact v41/v44 development continuation, not a fresh independently sampled E2E benchmark.",
            "Keep all24 endpoints: ten already-terminal v41 outputs unchanged, fourteen v44 round2 states.",
            "Actual canonical generate_conclusion(round2), append_conclusion_to_step, apply_terminal_conclusion; original validators.",
            "Keep the canonical optional final-direct fallback, including its second evidence preparation and exact prompt.",
            "No new planner, retrieval, MiniRAG, web, automatic E-ID rebinding, custom retry, weakened check or subject-reference gate.",
            "Same production4B SODA LoRA and legacy qwen3_nothink brace-prefill template as v41.",
            "2048 output tokens and .1/.8/1.05 sampling for each actual request; original12000-char evidence budget and12288 context.",
            "Use v41 paired seed for round2 conclusion; independently derived paired fallback seed is frozen before generation.",
            "Both primary and fallback prompts/tokens/kwargs are frozen; save both raw requests even if fallback replaces the first.",
            "Synthetic abstain responses used only in CPU protocol dry tests, never injected into actual model input.",
            "No truncation/retry on overflow; preserve errors and length finishes. Same seed does not promise determinism.",
            "Carry first-round mistakes without retry; fourteen unresolved states are not fourteen independent questions.",
            "No assistant observations, expected answers, historical labels, or source-review text in prompts.",
            "Recheck GPU2 numeric UUID/PCI mapping and fresh>=60-second idle window under the existing lock.",
            "No training, production source/config/default model change, restart, commit or push.",
        ],
        records=[], cpu_synthetic_protocol_checks=[],
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    helpers, frozen = staged.imports(), sources()
    pairs = {p["pair_id"]: p for p in old["pairs"]}
    first = {(r["pair_id"], r["condition"]): r
             for r in shared.load(V41 / "generations.json")["attempts"]}
    for record in pre44["records"]:
        key = record["pair_id"], record["condition"]
        source, pair = frozen[key], pairs[key[0]]
        item = {k: deepcopy(record[k]) for k in ("pair_id", "condition", "id", "question", "terminal_after_round1")}
        item["hypothesis"] = deepcopy(record["round2_hypothesis"])
        if item["terminal_after_round1"]:
            assert source["status"] == "terminal_carried_unchanged"
            assert source["terminal_answer"] == first[key]["conclusion"]["answer"]
            item["carried_terminal_answer"] = source["terminal_answer"]
            item["carried_conclusion"] = first[key]["conclusion"]
        else:
            assert source["status"] == "round2_retrieval_ok"
            item["planned_requests"] = planned_prompts(pre, helpers, item, source)
            seeds = [
                pair["sampling_seed"],
                int(shared.digest(f"staged-reader-v45:final-direct:{pair['sampling_seed']}")[:8], 16) % (2**31),
            ]
            for index, request in enumerate(item["planned_requests"]):
                ids = tokenizer.encode(request["prompt"], add_special_tokens=True)
                request.update(
                    prompt_sha256=shared.digest(request["prompt"]),
                    token_ids_sha256=shared.digest(json.dumps(ids)), prompt_tokens=len(ids),
                    context_fits=len(ids) + 2048 <= pre["runtime"]["max_model_len"],
                    seed=seeds[index], kwargs=deepcopy(PARAMS),
                )
                assert request["visible_evidence"] in request["prompt"]
                assert "<think>" not in request["prompt"].rsplit("<|im_start|>assistant", 1)[-1]
            dry = DryGenerator(pre["generator"], item)
            result = invoke(pre, helpers, item, source, dry)
            assert len(dry.requests) == 2
            assert result["conclusion"]["generation_diagnostics"]["direct_answer_fallback"]
            pre["cpu_synthetic_protocol_checks"].append({
                "pair_id": key[0], "condition": key[1], "requests": dry.requests,
                "fake_abstain_only": True, "actual_model_calls": 0,
            })
        pre["records"].append(item)
    assert len(pre["records"]) == 24 and len(pre["cpu_synthetic_protocol_checks"]) == 14
    paths = [
        Path(__file__), V44 / "run_second_round_retrieval.py", V44 / "preflight.json",
        V44 / "retrieval.json", V44 / "postflight_v1.json", V44 / "checkpoint_v44.json",
        V41 / "preflight.json", V41 / "generations.json", V41 / "run_canonical_reader_v2.py",
        V41 / "checkpoint_v41.json",
    ]
    pre["sha256"] = {**old["sha256"], **pre44["sha256"], **{str(p): shared.sha(p) for p in paths}}
    shared.save(HERE / "preflight.json", pre)
    requests = [q for r in pre["records"] for q in r.get("planned_requests", [])]
    print({
        "endpoints": 24, "terminal_carry": 10, "round2_states": 14, "cpu_dry_checks": 14,
        "longest_planned_input": max(q["prompt_tokens"] for q in requests),
        "planned_overflow_prompts": sum(not q["context_fits"] for q in requests),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
    }, flush=True)


def generate_all():
    pre = shared.load(HERE / "preflight.json")
    reader.verify(pre)
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    assert not (HERE / "generations.progress.jsonl").exists()
    helpers = staged.imports()
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
    llm_generate, runner_generate = llm.generate, runner.generate
    active, calls = {}, []

    def dispatch(prompts, params, **kwargs):
        planned = active["item"]["planned_requests"][len(active["requests"]) - 1]
        assert prompts == [planned["prompt"]] and kwargs["lora_request"] is adapter
        assert params.max_tokens == 2048 and params.temperature == .1 and params.top_p == .8
        assert params.repetition_penalty == 1.05
        params.seed = planned["seed"]
        call = {
            "pair_id": active["item"]["pair_id"], "condition": active["item"]["condition"],
            "kind": planned["kind"], "prompt_sha256": planned["prompt_sha256"],
            "seed": params.seed, "max_tokens": params.max_tokens, "temperature": params.temperature,
            "top_p": params.top_p, "repetition_penalty": params.repetition_penalty, "lora_id": adapter.lora_int_id,
        }
        calls.append(call)
        output = llm_generate(prompts, params, **kwargs)
        assert len(output) == len(output[0].outputs) == 1
        actual = output[0]
        ids = tokenizer.encode(prompts[0], add_special_tokens=True)
        assert list(actual.prompt_token_ids) == ids
        assert shared.digest(json.dumps(ids)) == planned["token_ids_sha256"]
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
        index = len(active["requests"])
        assert index < 2, "No custom retry beyond canonical primary/final-direct calls"
        planned = active["item"]["planned_requests"][index]
        assert prompt == planned["prompt"] and kwargs == planned["kwargs"]
        request = {"kind": planned["kind"], "prompt_sha256": shared.digest(prompt), "kwargs": kwargs}
        active["requests"].append(request)
        try:
            text = runner_generate(prompt, **kwargs)
            request.update(status="ok", sanitized_raw_output=text)
            return text
        except Exception as exc:
            request.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            request["runtime_diagnostics"] = deepcopy(runner.last_generation_diagnostics)

    llm.generate, runner.generate = dispatch, record_generate
    frozen, results = sources(), []
    with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
        for item in pre["records"]:
            start, before = time.monotonic(), len(calls)
            result = {k: item[k] for k in ("pair_id", "condition", "id", "terminal_after_round1")}
            active.clear()
            active.update(item=item, requests=[])
            if item["terminal_after_round1"]:
                result.update(
                    status="terminal_carried_unchanged", conclusion=item["carried_conclusion"],
                    final_answer=item["carried_terminal_answer"],
                )
            else:
                result["status"] = "error"
                try:
                    result.update(status="ok", **invoke(
                        pre, helpers, item, frozen[(item["pair_id"], item["condition"])], runner,
                    ))
                except Exception as exc:
                    result.update(error_type=type(exc).__name__, error_message=str(exc))
            result.update(
                requests=deepcopy(active["requests"]), calls=deepcopy(calls[before:]),
                seconds=time.monotonic() - start,
            )
            results.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print({
                "completed": len(results), "total": 24, "pair_id": item["pair_id"], "condition": item["condition"],
                "status": result["status"], "action": result.get("conclusion", {}).get("next_action"),
                "new_calls": len(result["calls"]),
            }, flush=True)
    reader.verify(pre)
    shared.save(HERE / "generations.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "records": results, "actual_new_llm_calls": len(calls),
        "production_changed": False, "training_ready": False,
    })
    print({"saved_endpoints": len(results), "actual_new_llm_calls": len(calls)}, flush=True)
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
        print("Explicit engine shutdown requested.", flush=True)


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    pre = shared.load(HERE / "preflight.json")
    assert retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    reader.verify(pre)
    assert not (HERE / "idle_samples.json").exists()
    shared.HERE, shared.run = HERE, generate_all
    shared.idle_then_run()


class ReplayGenerator:
    def __init__(self, config, item, saved):
        self.max_tokens, self.repeat_penalty = config["max_tokens"], config["repeat_penalty"]
        self.item, self.saved, self.calls = item, saved, 0

    def generate(self, prompt, **kwargs):
        request = self.saved["requests"][self.calls]
        planned = self.item["planned_requests"][self.calls]
        assert prompt == planned["prompt"] and kwargs == request["kwargs"] == planned["kwargs"]
        self.calls += 1
        if request["status"] == "error":
            error_class = type(request["error_type"], (Exception,), {})
            raise error_class(request["error_message"])
        return request["sanitized_raw_output"]


def check():
    pre, result = shared.load(HERE / "preflight.json"), shared.load(HERE / "generations.json")
    reader.verify(pre)
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(line) for line in (HERE / "generations.progress.jsonl").read_text().splitlines()]
    assert journal == result["records"] and len(journal) == 24
    helpers, frozen = staged.imports(), sources()
    summary = {k: {"status": Counter(), "action": Counter(), "new_calls": 0, "fallback_states": 0}
               for k in ("control", "candidate")}
    replays, finishes, carries = 0, Counter(), 0
    for item, saved in zip(pre["records"], journal, strict=True):
        key = item["pair_id"], item["condition"]
        assert key == (saved["pair_id"], saved["condition"])
        stats = summary[item["condition"]]
        stats["status"][saved["status"]] += 1
        stats["new_calls"] += len(saved["calls"])
        stats["fallback_states"] += len(saved["requests"]) == 2
        if item["terminal_after_round1"]:
            assert saved["status"] == "terminal_carried_unchanged"
            assert not saved["calls"] and not saved["requests"]
            assert saved["final_answer"] == item["carried_terminal_answer"]
            assert saved["conclusion"] == item["carried_conclusion"]
            carries += 1
        else:
            planned = planned_prompts(pre, helpers, item, frozen[key])
            assert len(saved["requests"]) in {1, 2}
            for i, request in enumerate(planned):
                assert all(item["planned_requests"][i][k] == v for k, v in request.items())
            generator = ReplayGenerator(pre["generator"], item, saved)
            try:
                replay = invoke(pre, helpers, item, frozen[key], generator)
            except Exception as exc:
                assert saved["status"] == "error"
                assert type(exc).__name__ == saved["error_type"] and str(exc) == saved["error_message"]
            else:
                assert saved["status"] == "ok"
                assert all(saved[k] == v for k, v in replay.items())
            assert generator.calls == len(saved["requests"])
            replays += 1
            assert len(saved["calls"]) <= len(saved["requests"])
            assert len({c["kind"] for c in saved["calls"]}) == len(saved["calls"])
            by_kind = {request["kind"]: request for request in item["planned_requests"]}
            for call in saved["calls"]:
                request = by_kind[call["kind"]]
                assert call["prompt_sha256"] == request["prompt_sha256"] and call["seed"] == request["seed"]
                assert call["max_tokens"] == 2048
                if "token_ids_sha256" in call:
                    assert call["token_ids_sha256"] == request["token_ids_sha256"]
                    assert call["prompt_tokens"] == request["prompt_tokens"]
                    assert call["lora_id"] == call["returned_lora_id"] == 1
                    assert call["generated_tokens"] <= 2048
                finishes[call.get("finish_reason", "generation_error")] += 1
        stats["action"][saved.get("conclusion", {}).get("next_action", "error")] += 1
    assert carries == 10 and replays == 14
    assert sum(len(r["calls"]) for r in journal) == result["actual_new_llm_calls"]
    shared.save(HERE / "postflight_v1.json", {
        "version": "staged_canonical_round2_reader_postflight_v45",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "endpoints": 24, "terminal_carry_unchanged": carries, "round2_canonical_replays": replays,
        "actual_new_llm_calls": result["actual_new_llm_calls"], "finish_counts": finishes,
        "summary_not_accuracy": summary, "production_changed": False, "training_ready": False,
        "assistant_source_review_complete": False, "not_independent_fresh_e2e": True,
    })
    print({"endpoints": 24, "replays": replays, "new_calls": result["actual_new_llm_calls"],
           "finishes": finishes, "summary_not_accuracy": summary}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_second_round_reader.py prepare|run|check")
    operations[sys.argv[1]]()
