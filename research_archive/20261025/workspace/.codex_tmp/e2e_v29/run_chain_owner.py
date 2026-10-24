"""32 predeclared reader-only calls; no retrieval, labels, API or production edit."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reader_evidence_router_20260909.v27"
sys.path.insert(0, str(PREVIOUS))
import run_reader_router as shared
from selector_protocol import CASE_IDS, reader_prompt
from chain_owner_protocol import with_explicit_owner


def prepare():
    from transformers import AutoTokenizer

    old = shared.load(PREVIOUS / "preflight.json")
    assert shared.load(PREVIOUS / "postflight_v1.json")["rows"] == 8
    tokenizer = AutoTokenizer.from_pretrained(old["tokenizer"], trust_remote_code=True)
    new = {key: deepcopy(old[key]) for key in (
        "base_model", "adapter", "tokenizer", "runtime", "generator", "sha256",
    )}
    new.update(
        version="explicit_chain_owner_reader32_v29",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_only=True, training_ready=False, production_changed=False,
        parent_preflight_sha256=shared.sha(PREVIOUS / "preflight.json"),
        protocol=[
            "Same eight selected development questions, two predeclared seeds per question, two conditions.",
            "Control uses the entire original evidence, unchanged legacy LoRA-compatible reader prompt.",
            "Candidate only replaces CHAIN_MEMBER_n metadata with its enclosing E owner.",
            "All story/header/whitespace spans, E labels, block order and counterevidence retained.",
            "No selector, no previous answer/label/reason, no hypothesis edit, no retrieval.",
            "Both conditions use the same actual LoRA, tokenizer and paired sampling seed.",
            "Alternating condition order; 2048 output tokens, context 12288, no input truncation.",
            "All failures retained; no output repair, quote rebinding, semantic gate or promotion by pass rate.",
            "Compound presentation intervention: removes member ordinal and repeats owner; not a pure ordinal ablation.",
            "Two seeds measure limited robustness, not independent questions or population correctness.",
        ],
        rows=[],
    )
    for repeat in range(2):
        for index, source in enumerate(old["rows"]):
            evidence, receipt = with_explicit_owner(source["evidence_brief"])
            assert evidence != source["evidence_brief"], "no metadata intervention in selected case"
            row = {
                key: deepcopy(source[key]) for key in (
                    "id", "question", "evidence_brief", "hypothesis",
                    "empty_reader_prompt", "control_prompt",
                )
            }
            row.update(
                repeat=repeat, candidate_evidence=evidence,
                metadata_preservation=receipt,
                candidate_prompt=reader_prompt(source["empty_reader_prompt"], evidence),
                sampling_seed=int(shared.digest(
                    f"explicit-owner-v29:{repeat}:{source['id']}"
                )[:8], 16) % (2**31),
                condition_order=(["control", "candidate"] if (repeat + index) % 2 == 0
                                 else ["candidate", "control"]),
            )
            row["pair_id"] = f"{source['id']}:seed{repeat}"
            for kind in ("control", "candidate"):
                prompt = row[f"{kind}_prompt"]
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                assert len(ids) + 2048 <= old["runtime"]["max_model_len"]
                row[f"{kind}_prompt_sha256"] = shared.digest(prompt)
                row[f"{kind}_prompt_tokens"] = len(ids)
                row[f"{kind}_token_ids_sha256"] = shared.digest(json.dumps(ids))
            new["rows"].append(row)
    for name in ("run_chain_owner.py", "chain_owner_protocol.py", "test_chain_owner_protocol.py"):
        new["sha256"][str(HERE / name)] = shared.sha(HERE / name)
    shared.save(HERE / "preflight.json", new)
    print(json.dumps({"prepared_pairs": 16, "planned_calls": 32,
                      "preflight_sha256": shared.sha(HERE / "preflight.json")}), flush=True)


def runtime_imports():
    sys.path.insert(0, str(PREVIOUS / "src"))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    return (VllmRunner, normalize_minimal_conclusion_output, parse_conclusion_output,
            validate_conclusion_grounding, HypothesisDocument)


def process(text, row, evidence, helpers):
    _, normalize, parse, validate, Hypothesis = helpers
    hypothesis = Hypothesis(**row["hypothesis"])
    normalized = normalize(text, "minimal")
    parsed = parse(normalized, question=row["question"], current_hypothesis=hypothesis,
                   max_round_reached=True)
    grounded = validate(question=row["question"], hypothesis=hypothesis, evidence=[],
                        conclusion=parsed, max_round_reached=True, mode="quote",
                        evidence_prompt_text=evidence)
    return {"normalized_output": normalized, "parsed": asdict(parsed), "grounded": asdict(grounded)}


def run():
    if (HERE / "generations.progress.jsonl").exists():
        raise RuntimeError("existing generation journal; refusing duplicate run")
    preflight = shared.load(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in preflight["sha256"].items())
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    helpers = runtime_imports()
    VllmRunner = helpers[0]
    engine = preflight["runtime"]
    runner = VllmRunner(
        base_model_path=Path(preflight["base_model"]), lora_path=Path(preflight["adapter"]),
        tensor_parallel_size=1, gpu_memory_utilization=engine["gpu_memory_utilization"],
        max_model_len=engine["max_model_len"], max_tokens=2048,
        temperature=0.1, top_p=0.8, repeat_penalty=1.05,
        dtype=engine["dtype"], max_num_batched_tokens=engine["max_num_batched_tokens"],
        enforce_eager=engine["enforce_eager"],
    )
    llm, adapter = runner._ensure_engine()
    assert adapter is not None
    tokenizer = llm.get_tokenizer()
    original_generate = llm.generate
    state, calls = {}, []

    def dispatch(prompts, params, **kwargs):
        assert len(prompts) == 1 and kwargs["lora_request"] is adapter
        params.seed = state["seed"]
        receipt = {
            "pair_id": state["pair_id"], "condition": state["condition"], "seed": params.seed,
            "actual_prompt": prompts[0], "prompt_sha256": shared.digest(prompts[0]),
            "lora_id": adapter.lora_int_id, "max_tokens": params.max_tokens,
        }
        calls.append(receipt)
        expected = tokenizer.encode(prompts[0], add_special_tokens=True)
        actual = original_generate(prompts, params, **kwargs)
        assert len(actual) == len(actual[0].outputs) == 1
        assert list(actual[0].prompt_token_ids) == expected
        assert actual[0].lora_request.lora_int_id == adapter.lora_int_id
        result = actual[0].outputs[0]
        receipt.update(
            prompt_tokens=len(expected), prompt_token_ids_sha256=shared.digest(json.dumps(expected)),
            returned_lora_id=actual[0].lora_request.lora_int_id,
            finish_reason=result.finish_reason, generated_tokens=len(result.token_ids),
            raw_output_with_special_tokens=result.text,
        )
        return actual

    llm.generate = dispatch
    results = []
    with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
        for row in preflight["rows"]:
            result = {"id": row["id"], "pair_id": row["pair_id"], "repeat": row["repeat"],
                      "question": row["question"], "conditions": {}}
            before_pair = len(calls)
            for condition in row["condition_order"]:
                state.update(pair_id=row["pair_id"], condition=condition, seed=row["sampling_seed"])
                evidence = row["evidence_brief"] if condition == "control" else row["candidate_evidence"]
                item = {"status": "error", "evidence_sha256": shared.digest(evidence)}
                before = len(calls)
                try:
                    text = runner.generate(row[f"{condition}_prompt"], max_tokens=2048,
                                           temperature=0.1, top_p=0.8, repeat_penalty=1.05)
                    item["raw_output"] = text
                    assert len(calls) == before + 1
                    assert calls[-1]["finish_reason"] == "stop", "non-stop output not accepted"
                    item.update(process(text, row, evidence, helpers), status="ok")
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                result["conditions"][condition] = item
            result["calls"] = calls[before_pair:]
            results.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({
                "completed_pairs": len(results), "total_pairs": 16, "pair_id": row["pair_id"],
                **{key: value.get("grounded", {}).get("next_action", value["status"])
                   for key, value in result["conditions"].items()},
            }, ensure_ascii=False), flush=True)
    changed = [p for p, h in preflight["sha256"].items() if shared.sha(Path(p)) != h]
    import vllm
    import torch
    shared.save(HERE / "generations.json", {
        "version": preflight["version"], "evaluation_only": True, "training_ready": False,
        "rows": results, "actual_call_count": len(calls),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "postflight_files_unchanged": not changed, "changed_paths": changed,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "production_changed": False, "semantic_review_complete": False,
    })
    if changed:
        raise RuntimeError("frozen files changed")


def check():
    from transformers import AutoTokenizer
    preflight = shared.load(HERE / "preflight.json")
    generated = shared.load(HERE / "generations.json")
    assert generated["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert generated["postflight_files_unchanged"]
    assert all(shared.sha(Path(p)) == h for p, h in preflight["sha256"].items())
    assert len(preflight["rows"]) == len(generated["rows"]) == 16
    assert [r["id"] for r in generated["rows"]] == CASE_IDS * 2
    helpers = runtime_imports()
    tokenizer = AutoTokenizer.from_pretrained(preflight["tokenizer"], trust_remote_code=True)
    summary = {k: {"status": Counter(), "action": Counter(), "warnings": Counter()}
               for k in ("control", "candidate")}
    call_count, finish = 0, Counter()
    for source, result in zip(preflight["rows"], generated["rows"], strict=True):
        assert source["pair_id"] == result["pair_id"]
        changed, receipt = with_explicit_owner(source["evidence_brief"])
        assert changed == source["candidate_evidence"] and receipt == source["metadata_preservation"]
        assert source["control_prompt"] == reader_prompt(source["empty_reader_prompt"], source["evidence_brief"])
        assert source["candidate_prompt"] == reader_prompt(source["empty_reader_prompt"], changed)
        assert [c["condition"] for c in result["calls"]] == source["condition_order"]
        for call in result["calls"]:
            kind = call["condition"]
            assert call["actual_prompt"] == source[f"{kind}_prompt"]
            assert call["seed"] == source["sampling_seed"]
            assert call["prompt_sha256"] == shared.digest(call["actual_prompt"])
            ids = tokenizer.encode(call["actual_prompt"], add_special_tokens=True)
            assert len(ids) == call["prompt_tokens"] == source[f"{kind}_prompt_tokens"]
            assert shared.digest(json.dumps(ids)) == call["prompt_token_ids_sha256"] == source[f"{kind}_token_ids_sha256"]
            assert call["lora_id"] == call["returned_lora_id"] == 1
            assert call["max_tokens"] == 2048 and len(ids) + 2048 <= preflight["runtime"]["max_model_len"]
            finish.update([call["finish_reason"]])
            call_count += 1
        for kind, item in result["conditions"].items():
            summary[kind]["status"].update([item["status"]])
            if item["status"] != "ok":
                continue
            evidence = source["evidence_brief"] if kind == "control" else changed
            replay = process(item["raw_output"], source, evidence, helpers)
            assert all(item[key] == value for key, value in replay.items())
            summary[kind]["action"].update([item["grounded"]["next_action"]])
            summary[kind]["warnings"].update(item["grounded"].get("grounding_warnings", []))
    assert call_count == generated["actual_call_count"] == 32
    report = {
        "evaluation_only": True, "training_ready": False, "unique_questions": 8,
        "paired_seeds_per_question": 2, "pairs": 16, "calls": call_count,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "all_non_metadata_spans_preserved": True, "all_evidence_labels_and_order_preserved": True,
        "actual_prompt_token_ids_lora_and_seed_verified": True,
        "grounding_replayed_against_actual_visible_evidence": True,
        "source_and_model_hashes_unchanged": True,
        "finish_reasons": finish, "summary": summary, "semantic_review_complete": False,
    }
    shared.save(HERE / "postflight_v1.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "idle_then_run", "check"):
        raise SystemExit("usage: run_chain_owner.py prepare|idle_then_run|check")
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "check":
        check()
    else:
        shared.HERE = HERE
        shared.run = run
        shared.idle_then_run()
