"""Frozen development experiment; append-only artifacts, no release writes."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import span_protocol as protocol

HERE = Path(__file__).resolve().parent
V49 = HERE.parent / "fallback_trace_retention_integration_20260910.v49"
V50 = HERE.parent / "relation_claim_sensitivity_20260910.v50"
sys.path[:0] = [str(V49), str(V50)]
import run_trace_integration as integration
import audit_relation_claims as audit

shared = integration.shared
reader = integration.probe.reader
PARAMS = {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.05}


def verify(pre):
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    for path, expected in pre["sha256"].items():
        assert shared.sha(Path(path)) == expected, path


def collect_inputs():
    """Exactly replay v50 with a transparent capture; restore helper afterward."""
    captured = {}
    original = audit.evaluate_projection

    def record(kwargs, fact, label):
        key = shared.digest(kwargs["evidence_prompt_text"])
        captured.setdefault(key, deepcopy(kwargs))
        return original(kwargs, fact, label)

    audit.evaluate_projection = record
    try:
        result = audit.compute()
    finally:
        audit.evaluate_projection = original
    saved = shared.load(V50 / "audit_v1.json")
    assert all(saved[k] == v for k, v in result.items())
    rows = []
    for row in result["rows"]:
        kwargs = captured[row["visible_sha256"]]
        packet = protocol.make_packet(kwargs["evidence_prompt_text"])
        from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
        assert dict(packet.blocks) == evidence_id_text_map_from_prompt(kwargs["evidence_prompt_text"])
        rows.append({
            "family": row["family"], "question": row["question"],
            "packet": protocol.packet_dict(packet),
            "canonical_kwargs": {
                **kwargs, "hypothesis": asdict(kwargs["hypothesis"]),
                "conclusion": asdict(kwargs["conclusion"]),
            },
            "source_row": row,
        })
    return rows


def task(task_id, kind, family, seed, prompt, claim=None):
    return {"task_id": task_id, "kind": kind, "family": family, "seed": seed,
            "prompt": prompt, "prompt_sha256": shared.digest(prompt), "claim": claim}


def make_tasks(rows):
    tasks = []
    for row in rows:
        packet, source = protocol.from_dict(row["packet"]), row["source_row"]
        family = row["family"]
        for index, projection in enumerate(source["single_fact_projections_not_model_answers"]):
            claim = projection["input"]["supported_facts"][0]["fact"]
            claim_id = f"C{index}"
            for repeat in range(2):
                seed = int(shared.digest(f"v53:{family}:{repeat}")[:8], 16) % (2**31)
                item = task(f"{family}:projection{index}:seed{repeat}", "verify_frozen", family, seed,
                            protocol.verify_prompt(packet, row["question"], claim, claim_id), claim)
                item.update(claim_id=claim_id, origin="assistant_authored_v50_contrast",
                            assistant_reference_label=projection["label"])
                tasks.append(item)
        original_fact = source["original_parsed_conclusion"]["supported_facts"][source["fact_index_zero_based"]]["fact"]
        seed = int(shared.digest(f"v53:{family}:historical")[:8], 16) % (2**31)
        item = task(f"{family}:historical", "verify_frozen", family, seed,
                    protocol.verify_prompt(packet, row["question"], original_fact, "H0"), original_fact)
        item.update(claim_id="H0", origin="actual_cached_model_fact",
                    assistant_reference_label="known_erroneous_or_unsupported_composite")
        tasks.append(item)
    for repeat in range(2):
        for row in rows:
            seed = int(shared.digest(f"v53:draft:{row['family']}:{repeat}")[:8], 16) % (2**31)
            packet = protocol.from_dict(row["packet"])
            tasks.append(task(f"{row['family']}:draft:seed{repeat}", "draft", row["family"], seed,
                              protocol.draft_prompt(packet, row["question"])))
    return tasks


def canonical_draft(row, draft):
    """NEW protocol output only; never accepts an old answer's bad citations."""
    from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
    packet = protocol.from_dict(row["packet"])
    protocol.validate_draft(draft, packet)
    kwargs = deepcopy(row["canonical_kwargs"])
    kwargs["hypothesis"] = HypothesisDocument(**kwargs["hypothesis"])
    original = ConclusionResult(**kwargs["conclusion"])
    facts = [{"fact": f["claim"], "evidence_refs": protocol.references(packet, f["span_ids"])}
             for f in draft["facts"]]
    proposed = replace(
        original, next_action="answer_directly" if facts else "abstain",
        answer="\n".join(f["fact"] for f in facts) if facts else "现有证据不足以确认。",
        supported_facts=facts, inferred_facts=[], missing_slots=draft["unresolved"],
        follow_up_hypothesis=None, grounding_warnings=[], generation_diagnostics={},
        clarification_question="",
    )
    kwargs["conclusion"] = deepcopy(proposed)
    issues, warnings = validate_grounded_quotes(
        conclusion=deepcopy(proposed), evidence=deepcopy(kwargs["evidence"]),
        question=kwargs["question"], evidence_prompt_text=kwargs["evidence_prompt_text"],
    ) if facts else ([], [])
    validated = validate_conclusion_grounding(**kwargs)
    return {"constructed_before_validation": asdict(proposed), "canonical_result": asdict(validated),
            "quote_issues": issues, "quote_warnings": warnings,
            "citation_origin": "fresh_draft_selected_fixed_spans_not_old_output_rebinding"}


def analyze(item, raw, row):
    packet = protocol.from_dict(row["packet"])
    value, mode = protocol.parse_prefilled(raw)
    if item["kind"] in ("verify_frozen", "verify_draft"):
        protocol.validate_verdict(value, packet, item["claim_id"])
        return {"parse_mode": mode, "value": value,
                "selected_source_spans": [asdict(s) for s in protocol.select_spans(
                    packet, value["span_ids"], required=False)],
                "shadow_semantic_pass": protocol.semantic_pass(value, item.get("draft_span_ids"))}
    protocol.validate_draft(value, packet)
    return {"parse_mode": mode, "value": value, "canonical": canonical_draft(row, value)}


def prepare():
    assert not (HERE / "preflight.json").exists()
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    checkpoint = shared.load(V50 / "checkpoint_v50.json")
    assert all(shared.sha(Path(p)) == h for p, h in checkpoint["basis_sha256"].items())
    reader.runtime_imports()
    rows = collect_inputs()
    old = shared.load(integration.probe.V41 / "preflight.json")
    pre = {k: deepcopy(old[k]) for k in (
        "base_model", "adapter", "tokenizer", "runtime", "generator", "query_config", "pipeline_config")}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    tasks = make_tasks(rows)
    for item in tasks:
        ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        item.update(prompt_tokens=len(ids), token_ids_sha256=shared.digest(json.dumps(ids)),
                    context_fits=len(ids) + PARAMS["max_tokens"] <= pre["runtime"]["max_model_len"])
    assert len(tasks) == 21 and all(t["context_fits"] for t in tasks)
    shared.save(HERE / "inputs.json", rows)
    paths = [
        HERE / "span_protocol.py", HERE / "test_span_protocol.py", Path(__file__),
        HERE / "README.md", HERE / "inputs.json", HERE / "heldout_review_template.json",
        V49 / "preflight.json", V49 / "checkpoint_v49.json", V49 / "run_trace_integration.py",
        V50 / "preflight.json", V50 / "audit_v1.json", V50 / "checkpoint_v50.json",
        V50 / "audit_relation_claims.py", integration.probe.V41 / "preflight.json",
    ]
    pre.update(
        version="exact_span_semantic_shadow_v53", created_at_utc=datetime.now(timezone.utc).isoformat(),
        sha256={str(p): shared.sha(p) for p in paths}, tasks=tasks,
        verified_nvidia_mapping=reader.retrieval.numeric_mapping(), params=PARAMS,
        max_actual_calls=33, new_protocol=True, production_changed=False, training_ready=False,
        model_independence=False, human_gold=False, independent_holdout_available=False,
        source_full_coverage=True, old_citations_rebound=False,
        max_fixed_prompt_tokens=max(t["prompt_tokens"] for t in tasks),
        evaluation_reference="v50 source-bound assistant observations, NOT human gold",
    )
    shared.save(HERE / "preflight.json", pre)
    print({"prepared_tasks": len(tasks), "max_calls": 33,
           "longest_fixed_prompt": pre["max_fixed_prompt_tokens"],
           "span_counts": {r["family"]: len(r["packet"]["spans"]) for r in rows},
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def child_task(parent, fact, index, row):
    packet = protocol.from_dict(row["packet"])
    claim_id = f"D{index}"
    result = task(parent["task_id"] + f":fact{index}", "verify_draft", row["family"],
                  parent["seed"], protocol.verify_prompt(packet, row["question"], fact["claim"], claim_id),
                  fact["claim"])
    result.update(claim_id=claim_id, parent_task_id=parent["task_id"], draft_span_ids=fact["span_ids"])
    return result


def generate_all():
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    assert reader.retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    shared.gpu_idle()
    assert not (HERE / "generations.progress.jsonl").exists()
    helpers = reader.runtime_imports()
    engine, config = pre["runtime"], pre["generator"]
    runner = helpers["VllmRunner"](
        base_model_path=Path(pre["base_model"]), lora_path=Path(pre["adapter"]),
        tensor_parallel_size=engine["tensor_parallel_size"],
        gpu_memory_utilization=engine["gpu_memory_utilization"], max_model_len=engine["max_model_len"],
        max_tokens=config["max_tokens"], temperature=config["temperature"], top_p=config["top_p"],
        repeat_penalty=config["repeat_penalty"], dtype=engine["dtype"],
        max_num_batched_tokens=engine["max_num_batched_tokens"], enforce_eager=engine["enforce_eager"],
    )
    llm = None
    results, state, all_calls = [], {}, []
    try:
        llm, adapter = runner._ensure_engine()
        assert adapter is not None
        tokenizer = llm.get_tokenizer()
        original = llm.generate

        def dispatch(prompts, params, **kwargs):
            item = state["item"]
            assert prompts == [item["prompt"]] and kwargs["lora_request"] is adapter
            assert params.max_tokens == 2048 and params.temperature == .1 and params.top_p == .8
            assert params.repetition_penalty == 1.05
            params.seed = item["seed"]
            receipt = {"task_id": item["task_id"], "seed": params.seed,
                       "prompt_sha256": shared.digest(prompts[0]), "params": PARAMS,
                       "lora_id": adapter.lora_int_id}
            all_calls.append(receipt)
            outputs = original(prompts, params, **kwargs)
            assert len(outputs) == len(outputs[0].outputs) == 1
            output, completion = outputs[0], outputs[0].outputs[0]
            assert list(output.prompt_token_ids) == state["ids"]
            assert output.lora_request.lora_int_id == adapter.lora_int_id
            receipt.update(
                prompt_tokens=len(state["ids"]), token_ids_sha256=shared.digest(json.dumps(state["ids"])),
                returned_lora_id=output.lora_request.lora_int_id,
                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                generated_tokens=len(completion.token_ids), raw_with_special_tokens=completion.text,
            )
            return outputs

        llm.generate = dispatch
        rows = {r["family"]: r for r in shared.load(HERE / "inputs.json")}
        with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
            def execute(item):
                start, before = time.monotonic(), len(all_calls)
                state.clear()
                ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
                state.update(item=item, ids=ids)
                result = {"task": item, "status": "error", "sanitized_raw_output": None}
                try:
                    if len(ids) + 2048 > engine["max_model_len"]:
                        raise protocol.ProtocolError("context_overflow_no_truncation")
                    if "token_ids_sha256" in item:
                        assert item["token_ids_sha256"] == shared.digest(json.dumps(ids))
                    raw = runner.generate(item["prompt"], **PARAMS)
                    result["sanitized_raw_output"] = raw
                    assert len(all_calls) == before + 1
                    if all_calls[-1].get("finish_reason") != "stop":
                        raise protocol.ProtocolError("non_stop_completion_not_repaired")
                    result["analysis"] = analyze(item, raw, rows[item["family"]])
                    result["status"] = "ok"
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                result.update(seconds=time.monotonic() - start, calls=deepcopy(all_calls[before:]),
                              runtime_diagnostics=deepcopy(runner.last_generation_diagnostics))
                assert len(result["calls"]) <= 1
                results.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print({"completed": len(results), "task": item["task_id"], "status": result["status"],
                       "finish": result["calls"][0].get("finish_reason") if result["calls"] else None,
                       "verdict": result.get("analysis", {}).get("value", {}).get("verdict"),
                       "action": result.get("analysis", {}).get("value", {}).get("action")}, flush=True)
                return result

            for item in pre["tasks"]:
                result = execute(item)
                if item["kind"] == "draft" and result["status"] == "ok":
                    for index, fact in enumerate(result["analysis"]["value"]["facts"]):
                        execute(child_task(item, fact, index, rows[item["family"]]))
        verify(pre)
        assert len(all_calls) <= pre["max_actual_calls"]
        shared.save(HERE / "generations.json", {
            "preflight_sha256": shared.sha(HERE / "preflight.json"),
            "records": results, "actual_call_count": len(all_calls),
            "production_changed": False, "old_outputs_unchanged": True,
        })
        print({"saved_records": len(results), "actual_calls": len(all_calls)}, flush=True)
    finally:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
            print("Explicit engine shutdown requested.", flush=True)


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    assert reader.retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    assert not (HERE / "idle_samples.json").exists()
    before = shared.HERE, shared.run
    shared.HERE, shared.run = HERE, generate_all
    try:
        shared.idle_then_run()
    finally:
        shared.HERE, shared.run = before


def recompute(pre, saved):
    reader.runtime_imports()
    rows = {r["family"]: r for r in shared.load(HERE / "inputs.json")}
    pending = list(pre["tasks"])
    frozen_summary, drafts, replayed = [], [], 0
    for result in saved["records"]:
        item = pending.pop(0)
        assert item == result["task"]
        if result["calls"]:
            receipt = result["calls"][0]
            assert receipt["seed"] == item["seed"] and receipt["prompt_sha256"] == item["prompt_sha256"]
            assert receipt["params"] == PARAMS and receipt["lora_id"] == 1
        if result["status"] == "ok":
            assert result["calls"][0]["finish_reason"] == "stop"
            actual = analyze(item, result["sanitized_raw_output"], rows[item["family"]])
            assert actual == result["analysis"]
            replayed += 1
            if item["kind"] == "draft":
                pending[:0] = [child_task(item, f, i, rows[item["family"]])
                               for i, f in enumerate(actual["value"]["facts"])]
        elif result["sanitized_raw_output"] is not None and result["calls"][0].get("finish_reason") == "stop":
            try:
                analyze(item, result["sanitized_raw_output"], rows[item["family"]])
            except Exception as exc:
                assert result["error"] == f"{type(exc).__name__}: {exc}"
            else:
                raise AssertionError("saved_error_did_not_reproduce")
        if item["kind"] == "verify_frozen":
            value = result.get("analysis", {}).get("value", {})
            frozen_summary.append({
                "task_id": item["task_id"], "claim": item["claim"],
                "assistant_reference_label_not_gold": item["assistant_reference_label"],
                "status": result["status"], "verdict": value.get("verdict"),
                "relevance": value.get("relevance"),
                "shadow_semantic_pass": result.get("analysis", {}).get("shadow_semantic_pass", False),
            })
    assert not pending
    for result in saved["records"]:
        item = result["task"]
        if item["kind"] != "draft":
            continue
        children = [r for r in saved["records"] if r["task"].get("parent_task_id") == item["task_id"]]
        value = result.get("analysis", {}).get("value", {})
        canonical = result.get("analysis", {}).get("canonical", {})
        answer = canonical.get("canonical_result", {}).get("answer")
        passes = (result["status"] == "ok" and value.get("action") == "answer"
                  and canonical["canonical_result"]["next_action"] == "answer_directly"
                  and len(children) == len(value["facts"])
                  and all(c["status"] == "ok" and c["analysis"]["shadow_semantic_pass"] for c in children))
        drafts.append({"task_id": item["task_id"], "status": result["status"],
                       "draft": value, "canonical": canonical, "checker_count": len(children),
                       "shadow_combined_pass": passes, "shadow_display": answer if passes else None})
    return {
        "fresh_cpu_replays_equal": replayed, "actual_call_count": saved["actual_call_count"],
        "record_count": len(saved["records"]),
        "status_counts": dict(Counter(r["status"] for r in saved["records"])),
        "finish_counts": dict(Counter(c.get("finish_reason") for r in saved["records"] for c in r["calls"])),
        "frozen_claims": frozen_summary, "drafts": drafts, "not_accuracy": True,
        "human_gold": False, "production_changed": False, "training_ready": False,
    }


def check():
    assert not (HERE / "postflight_v1.json").exists()
    pre, saved = shared.load(HERE / "preflight.json"), shared.load(HERE / "generations.json")
    verify(pre)
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(s) for s in (HERE / "generations.progress.jsonl").read_text().splitlines()]
    assert journal == saved["records"]
    assert sum(len(r["calls"]) for r in journal) == saved["actual_call_count"] <= 33
    summary = recompute(pre, saved)
    verify(pre)
    shared.save(HERE / "postflight_v1.json", {
        "preflight_sha256": shared.sha(HERE / "preflight.json"), **summary,
    })
    print({k: v for k, v in summary.items() if k not in ("frozen_claims", "drafts")}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_shadow.py prepare|run|check")
    operations[sys.argv[1]]()
