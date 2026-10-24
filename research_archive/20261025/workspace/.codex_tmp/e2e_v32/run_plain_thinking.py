"""Native 9B prose thinking/nothinking: paired 4096-token total budgets."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
V30 = HERE.parent / "reader_native_qwen35_20260910.v30"
V31 = HERE.parent / "reader_thinking_budget_20260910.v31"
sys.path.insert(0, str(V30))
import run_native_capacity as native
sys.path.insert(0, str(V31))
from thinking_protocol import answer_after_thinking
from plain_protocol import messages_for, verify_messages

shared = native.shared
KINDS = ("nothink", "think")


def prepare():
    from transformers import AutoTokenizer
    old = shared.load(V31 / "preflight.json")
    tokenizer = AutoTokenizer.from_pretrained(old["model"], trust_remote_code=False)
    pre = {
        "version": "native9b_plain_thinking16_v32",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "model": old["model"], "engine": deepcopy(old["engine"]),
        "sampling": deepcopy(old["sampling"]), "sha256": deepcopy(old["sha256"]),
        "protocol": [
            "Eight repeatedly inspected development questions, no independent accuracy estimate.",
            "Both conditions preserve question, hypothesis, full evidence and original evidence IDs.",
            "Within each pair only native enable_thinking differs; same seed and total output budget 4096.",
            "Same seed values as v31, but changed protocol; v31 outputs are not a paired control.",
            "System replaces quote/schema task with concise Chinese prose; this is not a pure JSON-only ablation.",
            "Unlike v30 plain, hypothesis is retained; neither outputs nor annotations are model input.",
            "No extra identity mappings, source selection, changed ordering, evidence deletion or truncation.",
            "Missing thinking terminator, empty answer, or length finish stays a failure; no retries.",
            "Prose is NOT production quote-validated and must not be promoted by this diagnostic.",
            "No LoRA, model pointer, training labels, verifier thresholds, production code or default changes.",
            "Pre-registered after inspecting first five v31 pairs; no independent or blind test claim.",
        ],
        "rows": [],
    }
    assert pre["sampling"]["max_tokens"] == 4096
    for source in old["rows"]:
        row = {key: deepcopy(source[key]) for key in
               ("id", "question", "hypothesis", "evidence_brief", "sampling_seed", "order")}
        row["messages"] = messages_for(row)
        verify_messages(row["messages"], source)
        row["prompts"] = {}
        for kind in KINDS:
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=kind == "think",
            )
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            assert len(ids) + 4096 <= pre["engine"]["max_model_len"]
            row["prompts"][kind] = {
                "text": prompt, "sha256": shared.digest(prompt), "tokens": len(ids),
                "token_ids_sha256": shared.digest(json.dumps(ids)),
            }
        assert row["prompts"]["nothink"]["text"].endswith("<think>\n\n</think>\n\n")
        assert row["prompts"]["think"]["text"] == row["prompts"]["nothink"]["text"][:-len("\n</think>\n\n")]
        pre["rows"].append(row)
    for path in (Path(__file__), HERE / "plain_protocol.py", HERE / "test_plain_protocol.py",
                 V31 / "preflight.json"):
        pre["sha256"][str(path)] = shared.sha(path)
    shared.save(HERE / "preflight.json", pre)
    print(json.dumps({"planned_calls": 16, "preflight_sha256": shared.sha(HERE / "preflight.json"),
                      "max_tokens_both": 4096, "no_production_validation": True}), flush=True)


def verify(pre):
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())


def run():
    import fcntl
    assert not (HERE / "calls.progress.jsonl").exists(), "refuse duplicate generation"
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
        shared.gpu_idle()
        generate(pre)


def generate(pre):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    import vllm
    llm = LLM(model=pre["model"], tokenizer=pre["model"], trust_remote_code=False, **pre["engine"])
    assert llm.llm_engine.vllm_config.lora_config is None
    assert llm.llm_engine.vllm_config.model_config.multimodal_config.language_model_only
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    rows = []
    with (HERE / "calls.progress.jsonl").open("x", encoding="utf-8") as journal:
        for source in pre["rows"]:
            result = {"id": source["id"], "question": source["question"], "conditions": {}}
            for kind in source["order"]:
                prompt = source["prompts"][kind]
                item = {
                    "status": "error", "actual_prompt": prompt["text"],
                    "seed": source["sampling_seed"], "lora_id": None,
                    "unvalidated_capacity_probe_only": True,
                }
                started = time.monotonic()
                try:
                    params = SamplingParams(**pre["sampling"], seed=source["sampling_seed"])
                    out = llm.generate([prompt["text"]], params, use_tqdm=False)[0]
                    assert out.lora_request is None and len(out.outputs) == 1
                    ids = tokenizer.encode(prompt["text"], add_special_tokens=True)
                    assert list(out.prompt_token_ids) == ids
                    choice = out.outputs[0]
                    item.update(
                        raw_output=choice.text, prompt_tokens=len(ids),
                        prompt_token_ids_sha256=shared.digest(json.dumps(ids)),
                        generated_tokens=len(choice.token_ids), finish_reason=choice.finish_reason,
                        returned_lora_id=None, seconds=time.monotonic() - started,
                    )
                    assert choice.finish_reason == "stop", "total budget exhausted; not accepted"
                    answer, chars = answer_after_thinking(choice.text, kind == "think")
                    assert answer.strip(), "empty final answer"
                    item.update(display_answer=answer.strip(), reasoning_chars=chars, status="ok")
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                result["conditions"][kind] = item
                journal.write(json.dumps({"id": source["id"], "condition": kind, "record": item},
                                         ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"id": source["id"], "condition": kind, "status": item["status"],
                                  "tokens": item.get("generated_tokens"),
                                  "finish": item.get("finish_reason")}), flush=True)
            rows.append(result)
    verify(pre)
    shared.save(HERE / "generations.json", {
        "version": pre["version"], "rows": rows, "evaluation_only": True, "training_ready": False,
        "production_changed": False, "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "source_and_assets_unchanged": True, "actual_no_lora_verified": True,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
    })


def check():
    from transformers import AutoTokenizer
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    assert gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    verify(pre)
    assert len(pre["rows"]) == len(gen["rows"]) == 8
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    summary = {k: {"status": Counter(), "finish": Counter(), "generated_tokens": 0} for k in KINDS}
    journal = [json.loads(line) for line in (HERE / "calls.progress.jsonl").read_text().splitlines()]
    assert len(journal) == 16
    counter = 0
    for source, result in zip(pre["rows"], gen["rows"], strict=True):
        assert source["id"] == result["id"] and set(result["conditions"]) == set(KINDS)
        verify_messages(source["messages"], source)
        for kind in source["order"]:
            item = result["conditions"][kind]
            assert journal[counter] == {"id": source["id"], "condition": kind, "record": item}
            counter += 1
            prompt = tokenizer.apply_chat_template(
                source["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=kind == "think",
            )
            expected = source["prompts"][kind]
            assert item["actual_prompt"] == expected["text"] == prompt
            assert shared.digest(prompt) == expected["sha256"]
            assert item["seed"] == source["sampling_seed"]
            assert item["unvalidated_capacity_probe_only"]
            summary[kind]["status"].update([item["status"]])
            summary[kind]["finish"].update([item.get("finish_reason", "generation_error")])
            summary[kind]["generated_tokens"] += item.get("generated_tokens", 0)
            if "prompt_tokens" in item:
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                assert len(ids) == item["prompt_tokens"] == expected["tokens"]
                assert shared.digest(json.dumps(ids)) == item["prompt_token_ids_sha256"] == expected["token_ids_sha256"]
                assert item["lora_id"] is item["returned_lora_id"] is None
                assert len(ids) + 4096 <= pre["engine"]["max_model_len"]
                assert item["generated_tokens"] <= 4096
            if item["status"] == "ok":
                assert item["finish_reason"] == "stop"
                answer, chars = answer_after_thinking(item["raw_output"], kind == "think")
                assert answer.strip() == item["display_answer"] and chars == item["reasoning_chars"]
                assert answer.strip()
            else:
                assert "display_answer" not in item and item["error"]
    shared.save(HERE / "postflight_v1.json", {
        "evaluation_only": True, "training_ready": False, "questions": 8, "calls": 16,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "summary": summary, "actual_prompts_tokens_seed_and_no_lora_verified": True,
        "total_4096_budget_both_verified": True, "all_calls_match_durable_journal": True,
        "full_evidence_and_hypothesis_preserved": True,
        "prose_NOT_production_quote_validated": True, "semantic_review_complete": False,
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_plain_thinking.py prepare|run|check")
    operations[sys.argv[1]]()
