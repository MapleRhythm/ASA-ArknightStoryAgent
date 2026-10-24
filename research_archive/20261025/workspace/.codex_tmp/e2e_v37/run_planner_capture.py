"""Fresh initial-planner provenance, then paired CPU projection on the SAME raw."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V35 = HERE.parent / "reader_alias_groups_20260910.v35"
RESULTS = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_no_minirag_fixed58_20260909.v23v3/results.jsonl")
sys.path.insert(0, str(V27))
import run_reader_router as shared
sys.path.insert(0, str(SRC))
from entity_projection import project_model_entities


def imports():
    from asa_arknight_story_agent.inference.generation.hypothesis_prompt_rendering import build_hypothesis_prompt
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.payload.json_utils import repair_json_like_output, extract_json_object
    from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
    from asa_arknight_story_agent.inference.payload.utils import normalize_string_list
    from asa_arknight_story_agent.inference.planning.retrieval_term_cleaning import clean_hypothesis_terms
    from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis, extract_entities, build_retrieval_query
    from asa_arknight_story_agent.inference.pipeline.types import ModelOutputError
    return locals()


def process(text, question, helpers):
    repaired = helpers["repair_json_like_output"](text)
    payload = helpers["extract_json_object"](repaired)
    report = {"repaired_parser_input": repaired, "parsed_model_payload": payload}
    try:
        if not payload:
            raise helpers["ModelOutputError"]("no parseable model payload")
        original_payload = deepcopy(payload)
        control = helpers["normalize_hypothesis_payload"](payload, question=question, dialogue_context="")
        assert payload == original_payload
        entities = helpers["normalize_string_list"](payload.get("entities"), limit=12)
        keywords = helpers["normalize_string_list"](payload.get("keywords"), limit=20)
        cleaned, _ = helpers["clean_hypothesis_terms"](entities, keywords, question=question)
        candidate = project_model_entities(control, cleaned)
        control_dict, candidate_dict = asdict(control), asdict(candidate)
        assert {k: v for k, v in control_dict.items() if k != "entities"} == {
            k: v for k, v in candidate_dict.items() if k != "entities"
        }
        report.update(
            normalization_status="valid_model_payload", cleaned_model_entities=cleaned,
            heuristic_entities=helpers["extract_entities"](question),
            code_added_entities=[term for term in control.entities if term not in cleaned],
            control=control_dict, candidate=candidate_dict,
            control_query=helpers["build_retrieval_query"](control),
            candidate_query=helpers["build_retrieval_query"](candidate),
        )
    except helpers["ModelOutputError"] as exc:
        fallback = helpers["build_hypothesis"](question, "")
        report.update(normalization_status="heuristic_fallback", normalization_error=str(exc),
                      control=asdict(fallback), candidate=asdict(fallback),
                      candidate_not_applied=True, code_added_entities=None,
                      control_query=helpers["build_retrieval_query"](fallback),
                      candidate_query=helpers["build_retrieval_query"](fallback))
    return report


def prepare():
    from transformers import AutoTokenizer
    assert not (HERE / "preflight.json").exists()
    helpers = imports()
    previous = shared.load(V35 / "preflight.json")
    original = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    assert len(original) == 58 and len({row["question"] for row in original}) == 58
    tokenizer = AutoTokenizer.from_pretrained(previous["tokenizer"], trust_remote_code=False)
    rows = []
    for index, source in enumerate(original):
        # Only the original question is transmitted. No old hypothesis, answer,
        # retrieval evidence, labels or source-review text is model input.
        question = source["question"]
        prompt = helpers["build_hypothesis_prompt"](question, "")
        assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n{")
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        assert len(ids) + 256 <= previous["runtime"]["max_model_len"]
        case_id = f"fixed58-{index + 1:03d}"
        rows.append({
            "id": case_id, "question": question, "dialogue_context": "",
            "prompt": prompt, "prompt_sha256": shared.digest(prompt),
            "prompt_tokens": len(ids), "token_ids_sha256": shared.digest(json.dumps(ids)),
            "seed": int(shared.digest("planner-provenance-v37:" + case_id)[:8], 16) % (2**31),
        })
    for old in previous["rows"][:8]:
        assert next(r["question"] for r in rows if r["id"] == old["id"]) == old["question"]
    files = [
        *sorted(SRC.rglob("*.py")), ROOT / "indexes/arknights_story/operator_aliases.json",
        ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json",
        ROOT / "configs/runtime_inference.json", RESULTS, V35 / "preflight.json",
        V27 / "run_reader_router.py", V27 / "selector_protocol.py",
        Path(__file__), HERE / "entity_projection.py", HERE / "test_entity_projection.py",
    ]
    manifest = {str(path): shared.sha(path) for path in files}
    # Inherit and verify ONLY the actual frozen model/tokenizer assets; current
    # canonical code has its own hashes, not a claim that it equals v27 code.
    for name, digest in previous["sha256"].items():
        path = Path(name)
        if any(path.is_relative_to(Path(previous[key])) for key in ("base_model", "adapter")):
            assert shared.sha(path) == digest
            manifest[name] = digest
    pre = {key: deepcopy(previous[key]) for key in ("base_model", "adapter", "tokenizer", "runtime")}
    pre.update(
        version="fresh_initial_planner58_entity_projection_v37",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_only=True, training_ready=False, production_changed=False,
        protocol=[
            "58 previously used development questions, one NEW initial planner call each, 256 total tokens.",
            "Original production 4B LoRA and legacy initial-planner prompt; no native thinking template.",
            "Current canonical initial-planner sampling: temperature .1, top_p .8, repetition_penalty 1.15.",
            "No old hypothesis, evidence, answer, review or labels are transmitted; empty dialogue context.",
            "Capture raw engine text, sanitized text, repaired parser input, parsed payload and cleaned model entities separately.",
            "On the SAME captured output, compare canonical normalization with an experimental final entity projection.",
            "Projection retains only cleaned model entity prefix; ALL other normalized fields/keywords/aliases stay byte-identical.",
            "Invalid payload falls back identically; length/generation failures remain failures with no retry.",
            "Projection may lose useful heuristic recovery. Synthetic recall-loss witness is explicitly retained.",
            "No fallback/global tokenizer/grounding/query policy/source/model-default changes.",
            "No retriever, reader, API judge, training or answer correctness claim.",
            "Do not promote by fewer entities; inspect useful additions, omissions and unchanged keyword pollution first.",
        ],
        sha256=manifest, rows=rows,
    )
    shared.save(HERE / "preflight.json", pre)
    print({"planned_calls": 58, "frozen_files": len(manifest), "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def verify(pre):
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())


def generate():
    assert not (HERE / "calls.progress.jsonl").exists()
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    helpers = imports()
    engine = pre["runtime"]
    runner = helpers["VllmRunner"](
        base_model_path=Path(pre["base_model"]), lora_path=Path(pre["adapter"]),
        tensor_parallel_size=1, gpu_memory_utilization=engine["gpu_memory_utilization"],
        max_model_len=engine["max_model_len"], max_tokens=256,
        temperature=.1, top_p=.8, repeat_penalty=1.15,
        dtype=engine["dtype"], max_num_batched_tokens=engine["max_num_batched_tokens"],
        enforce_eager=engine["enforce_eager"],
    )
    llm, adapter = runner._ensure_engine()
    assert adapter is not None
    tokenizer = llm.get_tokenizer()
    original_generate, calls, state = llm.generate, [], {}

    def dispatch(prompts, params, **kwargs):
        source = state["source"]
        assert prompts == [source["prompt"]] and kwargs["lora_request"] is adapter
        params.seed = source["seed"]
        receipt = {
            "id": source["id"], "actual_prompt": prompts[0],
            "seed": params.seed, "temperature": params.temperature,
            "top_p": params.top_p, "top_k": params.top_k,
            "repetition_penalty": params.repetition_penalty,
            "max_tokens": params.max_tokens, "lora_id": adapter.lora_int_id,
        }
        calls.append(receipt)
        actual = original_generate(prompts, params, **kwargs)
        assert len(actual) == len(actual[0].outputs) == 1
        ids = tokenizer.encode(prompts[0], add_special_tokens=True)
        assert actual[0].prompt_token_ids == ids
        assert actual[0].lora_request.lora_int_id == adapter.lora_int_id
        result = actual[0].outputs[0]
        receipt.update(
            prompt_tokens=len(ids), token_ids_sha256=shared.digest(json.dumps(ids)),
            returned_lora_id=actual[0].lora_request.lora_int_id,
            raw_engine_output=result.text, finish_reason=result.finish_reason,
            generated_tokens=len(result.token_ids),
        )
        return actual

    llm.generate = dispatch
    rows = []
    with (HERE / "calls.progress.jsonl").open("x", encoding="utf-8") as journal:
        for source in pre["rows"]:
            state["source"] = source
            before = len(calls)
            item = {"id": source["id"], "status": "error"}
            try:
                text = runner.generate(source["prompt"], max_tokens=256, temperature=.1, top_p=.8, repeat_penalty=1.15)
                item["sanitized_raw_output"] = text
                assert len(calls) == before + 1
                item["processing"] = process(text, source["question"], helpers)
                assert calls[-1]["finish_reason"] == "stop", "non-stop generation is not a valid comparison"
                item["status"] = "ok"
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            item["calls"] = calls[before:]
            rows.append(item)
            journal.write(json.dumps(item, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print({"completed": len(rows), "id": item["id"], "status": item["status"],
                   "added": item.get("processing", {}).get("code_added_entities")}, flush=True)
    verify(pre)
    shared.save(HERE / "generations.json", {
        "version": pre["version"], "rows": rows, "actual_call_count": len(calls),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "files_unchanged": True, "production_changed": False, "training_ready": False,
    })


def check():
    from transformers import AutoTokenizer
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    verify(pre)
    assert gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert gen["files_unchanged"]
    journal = [json.loads(line) for line in (HERE / "calls.progress.jsonl").read_text().splitlines()]
    assert journal == gen["rows"] and len(journal) == 58
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=False)
    helpers = imports()
    from asa_arknight_story_agent.inference.model_runtime.output_cleaning import sanitize_generation_output
    summary, additions, calls = Counter(), [], 0
    for source, item in zip(pre["rows"], gen["rows"], strict=True):
        assert source["id"] == item["id"]
        assert helpers["build_hypothesis_prompt"](source["question"], "") == source["prompt"]
        assert len(item["calls"]) == 1
        call = item["calls"][0]
        calls += 1
        assert call["actual_prompt"] == source["prompt"]
        assert call["seed"] == source["seed"] and call["max_tokens"] == 256
        assert call["temperature"] == .1 and call["top_p"] == .8 and call["repetition_penalty"] == 1.15
        assert call["lora_id"] == 1
        if "raw_engine_output" in call:
            assert call["returned_lora_id"] == call["lora_id"]
            ids = tokenizer.encode(source["prompt"], add_special_tokens=True)
            assert len(ids) == call["prompt_tokens"] == source["prompt_tokens"]
            assert shared.digest(json.dumps(ids)) == call["token_ids_sha256"] == source["token_ids_sha256"]
            assert call["generated_tokens"] <= 256
            assert sanitize_generation_output(call["raw_engine_output"], source["prompt"]) == item["sanitized_raw_output"]
        if "processing" in item:
            assert process(item["sanitized_raw_output"], source["question"], helpers) == item["processing"]
        summary[item["status"]] += 1
        if item["status"] == "ok":
            assert call["finish_reason"] == "stop"
            processed = item["processing"]
            summary[processed["normalization_status"]] += 1
            if processed["normalization_status"] == "valid_model_payload":
                extra = processed["code_added_entities"]
                summary["valid_with_code_added_entities"] += bool(extra)
                summary["valid_changed_query"] += processed["control_query"] != processed["candidate_query"]
                summary["added_entity_occurrences"] += len(extra)
                additions.append({
                    "id": source["id"], "question": source["question"],
                    "cleaned_model_entities": processed["cleaned_model_entities"],
                    "code_added_entities": extra,
                    "keywords_unchanged": processed["control"]["keywords"] == processed["candidate"]["keywords"],
                    "added_terms_still_in_keywords": [t for t in extra if t in processed["candidate"]["keywords"]],
                })
        else:
            assert "error" in item
    assert calls == gen["actual_call_count"] == 58
    shared.save(HERE / "postflight_v1.json", {
        "version": "fresh_initial_planner58_postflight_v37",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "summary_not_accuracy": summary, "valid_model_additions": additions,
        "actual_prompt_token_ids_lora_seed_and_sampling_verified": True,
        "raw_sanitization_parse_and_normalization_replayed": True,
        "same_raw_output_both_normalizations": True,
        "all_non_entity_fields_preserved": True,
        "production_changed": False, "training_ready": False,
        "promotion_decision": "not_promoted_pending_source_review_and_end_to_end_evaluation",
    })
    print(dict(summary), flush=True)


def run():
    assert not (HERE / "calls.progress.jsonl").exists()
    verify(shared.load(HERE / "preflight.json"))
    shared.HERE, shared.run = HERE, generate
    shared.idle_then_run()


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_planner_capture.py prepare|run|check")
    operations[sys.argv[1]]()
