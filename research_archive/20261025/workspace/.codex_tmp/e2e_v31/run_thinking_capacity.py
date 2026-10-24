"""Paired 9B thinking/non-thinking, same 4096-token total output budget."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reader_native_qwen35_20260910.v30"
sys.path.insert(0, str(PREVIOUS))
import run_native_capacity as native
from thinking_protocol import answer_after_thinking
shared = native.shared


def prepare():
    from transformers import AutoTokenizer
    old = shared.load(PREVIOUS / "preflight.json")
    tokenizer = AutoTokenizer.from_pretrained(old["models"]["native9b"], trust_remote_code=False)
    new = {
        "version": "native9b_thinking_budget16_v31",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "model": old["models"]["native9b"],
        "engine": {**old["engine"], "gpu_memory_utilization": old["memory_fraction"]["native9b"]},
        "sampling": {**old["sampling"], "max_tokens": 4096},
        "sha256": deepcopy(old["sha256"]),
        "protocol": [
            "Same eight selected development questions; both conditions use native9b with no LoRA.",
            "Exactly the same structured system/user, full original evidence and hypothesis.",
            "Only native enable_thinking changes within each pair; total output budget is 4096 for BOTH.",
            "Thinking tokens and answer share that 4096 budget, no hidden extra generation.",
            "Same paired seed and sampling; alternating order; context 12288; no input truncation.",
            "Strict thinking termination required; missing terminator or length stop stays a failure.",
            "Canonical quote normalization and validation operate on final answer only, with unchanged evidence.",
            "No deletion, ID/quote repair, relaxed validation, training labels or production change.",
            "Old 2048 nonthinking run is descriptive, not this comparison's paired control.",
        ],
        "rows": [],
    }
    for index, source in enumerate(old["rows"]):
        row = {key: deepcopy(source[key]) for key in ("id", "question", "hypothesis", "evidence_brief")}
        row.update(
            messages=deepcopy(source["messages"]["quote"]),
            sampling_seed=int(shared.digest("thinking-v31:" + source["id"])[:8], 16) % (2**31),
            order=["nothink", "think"] if index % 2 == 0 else ["think", "nothink"],
            prompts={},
        )
        for condition, enabled in (("nothink", False), ("think", True)):
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=enabled,
            )
            assert source["evidence_brief"] in prompt
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            assert len(ids) + 4096 <= 12288
            row["prompts"][condition] = {
                "text": prompt, "sha256": shared.digest(prompt), "tokens": len(ids),
                "token_ids_sha256": shared.digest(json.dumps(ids)),
            }
        assert row["prompts"]["nothink"]["text"].endswith("<think>\n\n</think>\n\n")
        assert row["prompts"]["think"]["text"] == row["prompts"]["nothink"]["text"][:-len("\n</think>\n\n")]
        new["rows"].append(row)
    for path in (Path(__file__), HERE / "thinking_protocol.py", HERE / "test_thinking_protocol.py",
                 PREVIOUS / "preflight.json"):
        new["sha256"][str(path)] = shared.sha(path)
    shared.save(HERE / "preflight.json", new)
    print(json.dumps({"prepared": 8, "planned_calls": 16, "max_tokens_both": 4096,
                      "preflight_sha256": shared.sha(HERE / "preflight.json")}), flush=True)


def verify(preflight):
    assert all(shared.sha(Path(p)) == h for p, h in preflight["sha256"].items())


def run():
    import fcntl
    assert not (HERE / "generations.progress.jsonl").exists()
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    lock = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/locks/gpu2_asa_e2e.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, samples = time.monotonic(), []
        while True:
            samples.append(shared.gpu_idle())
            print(json.dumps({"idle_sample": len(samples), **samples[-1]}), flush=True)
            if len(samples) >= 13 and time.monotonic() - start >= 60:
                break
            time.sleep(5)
        shared.save(HERE / "idle_samples.json",
                    {"samples": samples, "elapsed_seconds": time.monotonic() - start})
        generate(pre)


def generate(pre):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    helpers = native.runtime_imports()
    llm = LLM(model=pre["model"], tokenizer=pre["model"], trust_remote_code=False, **pre["engine"])
    assert llm.llm_engine.vllm_config.lora_config is None
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    rows = []
    with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
        for row in pre["rows"]:
            result = {"id": row["id"], "question": row["question"], "conditions": {}}
            for condition in row["order"]:
                prompt = row["prompts"][condition]
                record = {"status": "error", "actual_prompt": prompt["text"],
                          "seed": row["sampling_seed"], "lora_id": None}
                started = time.monotonic()
                try:
                    params = SamplingParams(**pre["sampling"], seed=row["sampling_seed"])
                    output = llm.generate([prompt["text"]], params, use_tqdm=False)[0]
                    assert len(output.outputs) == 1 and output.lora_request is None
                    ids = tokenizer.encode(prompt["text"], add_special_tokens=True)
                    assert list(output.prompt_token_ids) == ids
                    choice = output.outputs[0]
                    record.update(
                        raw_output=choice.text, prompt_tokens=len(ids),
                        prompt_token_ids_sha256=shared.digest(json.dumps(ids)),
                        generated_tokens=len(choice.token_ids), finish_reason=choice.finish_reason,
                        seconds=time.monotonic() - started, returned_lora_id=None,
                    )
                    assert choice.finish_reason == "stop", "total budget exhausted; not accepted"
                    answer, reasoning_chars = answer_after_thinking(choice.text, condition == "think")
                    record.update(answer_text=answer, reasoning_chars=reasoning_chars)
                    record.update(native.process(answer, row, row["evidence_brief"], helpers), status="ok")
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                result["conditions"][condition] = record
            rows.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({"completed": len(rows), "total": 8, "id": row["id"],
                              **{k: v.get("grounded", {}).get("next_action", v["status"])
                                 for k, v in result["conditions"].items()}}, ensure_ascii=False), flush=True)
    verify(pre)
    shared.save(HERE / "generations.json", {
        "version": pre["version"], "rows": rows, "evaluation_only": True, "training_ready": False,
        "production_changed": False, "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "source_and_assets_unchanged": True, "actual_no_lora_verified": True,
    })


def check():
    from transformers import AutoTokenizer
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    assert gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert len(pre["rows"]) == len(gen["rows"]) == 8
    verify(pre)
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    helpers = native.runtime_imports()
    summary = {k: {"status": Counter(), "action": Counter(), "finish": Counter()}
               for k in ("nothink", "think")}
    for source, result in zip(pre["rows"], gen["rows"], strict=True):
        assert source["id"] == result["id"]
        for kind in ("nothink", "think"):
            item = result["conditions"][kind]
            assert item["actual_prompt"] == source["prompts"][kind]["text"]
            assert item["seed"] == source["sampling_seed"]
            summary[kind]["status"].update([item["status"]])
            summary[kind]["finish"].update([item.get("finish_reason", "generation_error")])
            if "prompt_tokens" in item:
                ids = tokenizer.encode(item["actual_prompt"], add_special_tokens=True)
                assert len(ids) == item["prompt_tokens"] == source["prompts"][kind]["tokens"]
                assert shared.digest(json.dumps(ids)) == item["prompt_token_ids_sha256"] == source["prompts"][kind]["token_ids_sha256"]
                assert item["lora_id"] is item["returned_lora_id"] is None
                assert len(ids) + 4096 <= 12288 and item["generated_tokens"] <= 4096
            if item["status"] != "ok":
                continue
            answer, chars = answer_after_thinking(item["raw_output"], kind == "think")
            assert answer == item["answer_text"] and chars == item["reasoning_chars"]
            replay = native.process(answer, source, source["evidence_brief"], helpers)
            assert all(item[key] == value for key, value in replay.items())
            summary[kind]["action"].update([item["grounded"]["next_action"]])
    report = {
        "evaluation_only": True, "training_ready": False, "questions": 8, "calls": 16,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "summary": summary, "actual_prompts_tokens_seed_and_no_lora_verified": True,
        "total_4096_budget_both_verified": True, "canonical_grounding_replayed": True,
        "semantic_review_complete": False,
    }
    shared.save(HERE / "postflight_v1.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["run"]:
        run()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        raise SystemExit("usage: run_thinking_capacity.py prepare|run|check")
