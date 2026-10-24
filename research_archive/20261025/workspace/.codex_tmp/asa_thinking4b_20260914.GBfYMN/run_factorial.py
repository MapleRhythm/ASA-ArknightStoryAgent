"""Bounded 40-generation native thinking x LoRA experiment, isolated from serving."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from protocol_helpers import sha, digest, normalized, answer_after_thinking

HERE = Path(__file__).resolve().parent
REPO = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
BASE = REPO / "model/qwen3.5-4b"
ADAPTER = REPO / "model/lora/teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2"
PIPE = REPO / "src/goldenglow/inference/cpu_pipeline.py"
CFG = REPO / "configs/runtime_inference_gpu.json"
CONDITIONS = ["base_off", "base_on", "lora_off", "lora_on"]
MAX_OUTPUT = 32768
MAX_CONTEXT = 40960
SAMPLING = dict(temperature=1.0, top_p=.95, top_k=20, min_p=0.0,
                presence_penalty=1.5, repetition_penalty=1.0,
                max_tokens=MAX_OUTPUT, skip_special_tokens=False,
                stop=["<|im_end|>", "<|endoftext|>"])
ENGINE = dict(trust_remote_code=True, enable_lora=True, max_lora_rank=16,
              max_loras=1, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=.80, max_model_len=MAX_CONTEXT,
              max_num_seqs=1, max_num_batched_tokens=8192,
              enforce_eager=True, disable_log_stats=True,
              generation_config="vllm")
EXPECTED = {
    PIPE: "29dfbba72879cec2fa58dabb9c7f6ecc7a36359a457f2482e008e399db790fdc",
    CFG: "49dfe81d5a2349f213b556abdee218f24a644ae3ab92bb57a8a460740e854527",
    ADAPTER / "adapter_model.safetensors":
        "20c369a0042478f50b4128c047eb03844bd0e74c8eccdf6139e3344a396ef0ce",
}


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def check_expected():
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, f"Changed source: {path}"


def prepare():
    from transformers import AutoTokenizer
    from goldenglow.inference import cpu_pipeline as module
    check_expected()
    source = json.loads((HERE / "inputs.json").read_text())
    assert len(source["rows"]) == 10
    base_tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    adapter_tok = AutoTokenizer.from_pretrained(ADAPTER, local_files_only=True)
    assets = [
        PIPE, CFG, ADAPTER / "adapter_model.safetensors", ADAPTER / "adapter_config.json",
        BASE / "config.json", BASE / "chat_template.jinja", BASE / "tokenizer.json",
        BASE / "tokenizer_config.json", BASE / "README.md",
        ADAPTER / "tokenizer.json", ADAPTER / "tokenizer_config.json",
        ADAPTER / "chat_template.jinja", HERE / "inputs.json", HERE / "PROTOCOL.md",
        HERE / "run_factorial.py", HERE / "protocol_helpers.py",
    ] + sorted(BASE.glob("*.safetensors"))
    pre = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "engine": ENGINE, "sampling": SAMPLING,
        "mode": source["mode"], "condition_names": CONDITIONS,
        "base_model": str(BASE), "adapter": str(ADAPTER),
        "common_tokenizer": str(BASE),
        "planned_generations": 40, "semantic_review_complete": False,
        "versions": {x: importlib.metadata.version(x) for x in ("torch", "vllm", "transformers")},
        "sha256": {str(path): sha(path) for path in assets},
        "grounding_mode": json.loads(CFG.read_text())["inference"]["answer_grounding_mode"],
        "rows": [],
    }
    for i, row in enumerate(source["rows"]):
        item = dict(row)
        item["order"] = CONDITIONS[i % 4:] + CONDITIONS[:i % 4]
        item["prompts"] = {}
        bodies = [normalized(module.strip_internal_evidence_meta(x["clean_text"]))
                  for x in row["documents"]]
        assert len(bodies) == 20 and all(bodies)
        for kind, enabled in (("off", False), ("on", True)):
            prompt = base_tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=enabled,
            )
            other = adapter_tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=enabled,
            )
            assert prompt == other, "Base/adapter template mismatch"
            ids = base_tok.encode(prompt, add_special_tokens=True)
            assert ids == adapter_tok.encode(prompt, add_special_tokens=True), "Tokenizer mismatch"
            assert len(ids) + MAX_OUTPUT <= MAX_CONTEXT
            checks = [body in normalized(prompt) for body in bodies]
            assert all(checks), "Incomplete evidence"
            item["prompts"][kind] = {
                "text": prompt, "tokens": len(ids), "token_ids_sha256": digest(ids),
                "sha256": digest(prompt), "body_checks": checks,
            }
        off, on = item["prompts"]["off"]["text"], item["prompts"]["on"]["text"]
        assert off.endswith("<think>\n\n</think>\n\n") and on.endswith("<think>\n")
        assert off == on + "\n</think>\n\n"
        pre["rows"].append(item)
    write_json(HERE / "preflight.json", pre)
    print(json.dumps({"preflight_ok": True, "questions": 10, "planned_calls": 40,
                      "max_input_tokens": max(x["tokens"] for r in pre["rows"]
                                              for x in r["prompts"].values()),
                      "total_output_budget": MAX_OUTPUT,
                      "base_adapter_tokenizers_equal": True,
                      "preflight_sha256": sha(HERE / "preflight.json")}), flush=True)


def verify(pre):
    for name, expected in pre["sha256"].items():
        assert sha(Path(name)) == expected, f"Changed frozen asset: {name}"


def gpu_snapshot():
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).strip()
    target = next(x for x in gpu.splitlines() if x.startswith("0,"))
    fields = [x.strip() for x in target.split(",")]
    assert fields[1] == "GPU-a3cea8b6-c06d-a974-38ef-19ba853d586d"
    return {"utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "apps": apps,
            "target_used_mib": int(fields[2]), "target_util_pct": int(fields[4]),
            "target_has_apps": fields[1] in apps}


def parse_final(final, row, pre):
    from goldenglow.inference import cpu_pipeline as module
    hypothesis = module.HypothesisDocument(**row["hypothesis"])
    fixed = module.repair_json_like_output(final)
    payload = module.extract_json_object(fixed)
    if payload:
        conclusion = module.normalize_conclusion_payload(
            payload, question=row["question"], dialogue_context=hypothesis.dialogue_context,
            current_intent=hypothesis.intent, current_hypothesis=hypothesis,
            max_round_reached=False,
        )
    else:
        conclusion = module.parse_conclusion_json_like_output(
            fixed, question=row["question"], dialogue_context=hypothesis.dialogue_context,
            current_intent=hypothesis.intent, current_hypothesis=hypothesis,
            max_round_reached=False,
        )
        if not conclusion:
            raise ValueError("Canonical parser found no conclusion")
    before = asdict(conclusion)
    evidence = [{"document": doc, "score": 0.0} for doc in row["documents"]]
    grounded = module.validate_conclusion_grounding(
        question=row["question"], hypothesis=hypothesis, evidence=evidence,
        conclusion=conclusion, max_round_reached=False,
        mode=pre["grounding_mode"],
        evidence_prompt_text="\n".join(x["content"] for x in row["messages"]),
    )
    return {"payload": payload, "normalized": before, "grounded": asdict(grounded)}


def run():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    pre = json.loads((HERE / "preflight.json").read_text())
    verify(pre)
    out = HERE / "attempt1"
    out.mkdir(exist_ok=False)
    with Path("/tmp/asa-think4b-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        samples = []
        idle_started = time.monotonic()
        while True:
            sample = gpu_snapshot()
            samples.append(sample)
            assert not sample["target_has_apps"] and sample["target_used_mib"] < 256, (
                "GPU0 no longer idle; no process will be stopped"
            )
            print(json.dumps({"event": "idle_check", "sample": len(samples),
                              "used_mib": sample["target_used_mib"]}), flush=True)
            if time.monotonic() - idle_started >= 60:
                break
            time.sleep(5)
        write_json(out / "idle_samples.json", samples)
        load_started = time.monotonic()
        print(json.dumps({"event": "engine_loading", "pid": os.getpid()}), flush=True)
        llm = LLM(model=str(BASE), tokenizer=str(BASE), **ENGINE)
        tokenizer = llm.get_tokenizer()
        adapter_request = LoRARequest("current_soda", 1, str(ADAPTER))
        write_json(out / "manifest.json", {
            "preflight_sha256": sha(HERE / "preflight.json"),
            "pid": os.getpid(), "engine_load_sec": time.monotonic() - load_started,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "effective_engine": {
                "max_model_len": llm.llm_engine.vllm_config.model_config.max_model_len,
                "dtype": str(llm.llm_engine.vllm_config.model_config.dtype),
                "max_num_seqs": llm.llm_engine.vllm_config.scheduler_config.max_num_seqs,
            },
        })
        count = 0
        with (out / "events.jsonl").open("x") as journal, (out / "results.jsonl").open("x") as results:
            for row in pre["rows"]:
                for condition in row["order"]:
                    count += 1
                    thinking = condition.endswith("_on")
                    use_lora = condition.startswith("lora_")
                    item = row["prompts"]["on" if thinking else "off"]
                    ids = tokenizer.encode(item["text"], add_special_tokens=True)
                    assert digest(ids) == item["token_ids_sha256"]
                    assert len(ids) + MAX_OUTPUT <= MAX_CONTEXT
                    record = {
                        "request_index": count, "question_id": row["question_id"],
                        "question": row["question"], "condition": condition,
                        "thinking": thinking, "requested_lora_id": 1 if use_lora else None,
                        "seed": row["seed"], "prompt_sha256": item["sha256"],
                        "input_tokens": len(ids), "input_ids_sha256": digest(ids),
                        "max_output_tokens": MAX_OUTPUT, "status": "generation_error",
                    }
                    emit(journal, {"event": "request_started", **record})
                    print(json.dumps({"event": "request_started", "n": count,
                                      "q": row["question_id"], "condition": condition}), flush=True)
                    started = time.monotonic()
                    try:
                        params = SamplingParams(**SAMPLING, seed=row["seed"])
                        outputs = llm.generate(
                            [{"prompt_token_ids": ids}], params, use_tqdm=False,
                            lora_request=adapter_request if use_lora else None,
                        )
                        record["elapsed_sec"] = time.monotonic() - started
                        output = outputs[0]
                        returned_id = output.lora_request.lora_int_id if output.lora_request else None
                        record["returned_lora_id"] = returned_id
                        record["input_token_ids_exact_match"] = list(output.prompt_token_ids) == ids
                        assert record["input_token_ids_exact_match"]
                        assert returned_id == record["requested_lora_id"]
                        assert len(output.outputs) == 1
                        choice = output.outputs[0]
                        record.update(
                            raw_output=choice.text, finish_reason=choice.finish_reason,
                            stop_reason=choice.stop_reason, output_tokens=len(choice.token_ids),
                            output_token_ids_sha256=digest(list(choice.token_ids)),
                        )
                        if choice.finish_reason != "stop":
                            record["status"] = "budget_exhausted"
                        else:
                            record["status"] = "final_extraction_error"
                            final, reasoning = answer_after_thinking(choice.text, thinking)
                            record.update(final_text=final, reasoning_chars=len(reasoning),
                                          reasoning_tokens_reencoded=len(tokenizer.encode(
                                              reasoning, add_special_tokens=False)) if reasoning else 0)
                            record["status"] = "protocol_error"
                            record.update(parse_final(final, row, pre), status="ok")
                    except Exception as exc:
                        record["error_type"] = type(exc).__name__
                        record["error"] = str(exc)
                        if "raw_output" not in record:
                            # Do not silently continue after an engine/integrity failure.
                            record["elapsed_sec"] = time.monotonic() - started
                            emit(results, record)
                            emit(journal, {"event": "fatal", **record})
                            raise
                    record.setdefault("elapsed_sec", time.monotonic() - started)
                    emit(results, record)
                    emit(journal, {"event": "request_finished",
                                   **{k: record[k] for k in (
                                       "request_index", "question_id", "condition", "status", "elapsed_sec")}})
                    print(json.dumps({"event": "request_finished", "n": count,
                                      "q": row["question_id"], "condition": condition,
                                      "status": record["status"], "tokens": record.get("output_tokens"),
                                      "seconds": round(record["elapsed_sec"], 2),
                                      "action": record.get("grounded", {}).get("next_action")}),
                          flush=True)
        verify(pre)
        write_json(out / "completion.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "generations": count, "expected_generations": 40,
            "preflight_sha256": sha(HERE / "preflight.json"),
            "results_sha256": sha(out / "results.jsonl"),
            "frozen_assets_unchanged": True,
        })
        print(json.dumps({"event": "complete", "generations": count}), flush=True)


def check():
    pre = json.loads((HERE / "preflight.json").read_text())
    verify(pre)
    out = HERE / "attempt1"
    completion = json.loads((out / "completion.json").read_text())
    assert completion["results_sha256"] == sha(out / "results.jsonl")
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 40
    assert len({(x["question_id"], x["condition"]) for x in rows}) == 40
    source = {x["question_id"]: x for x in pre["rows"]}
    replayed = 0
    for row in rows:
        assert row["input_token_ids_exact_match"]
        assert row["returned_lora_id"] == row["requested_lora_id"]
        assert row["seed"] == source[row["question_id"]]["seed"]
        if row["status"] == "ok":
            final, reasoning = answer_after_thinking(row["raw_output"], row["thinking"])
            assert final == row["final_text"] and len(reasoning) == row["reasoning_chars"]
            replay = parse_final(final, source[row["question_id"]], pre)
            assert all(row[k] == v for k, v in replay.items())
            replayed += 1
    write_json(out / "postflight.json", {
        "complete": True, "generation_count": 40, "canonical_replays": replayed,
        "input_token_ids_and_lora_request_verified": True,
        "semantic_review_complete": False,
    })
    print(json.dumps({"checked": 40, "canonical_replays": replayed}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "check"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "check": check}[args.action]()
