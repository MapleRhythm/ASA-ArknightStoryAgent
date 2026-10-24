"""Four development questions: native quote-thinking, paired sampler profiles."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
V31 = HERE.parent / "reader_thinking_budget_20260910.v31"
V32 = HERE.parent / "reader_plain_thinking_20260910.v32"
sys.path.insert(0, str(V32))
import run_plain_thinking as harness
native = harness.native
shared = harness.shared
answer_after_thinking = harness.answer_after_thinking
KINDS = ("legacy", "model_card")
CASE_IDS = ["fixed58-015", "fixed58-017", "fixed58-023", "fixed58-031"]


def prepare():
    from transformers import AutoTokenizer
    old = shared.load(V31 / "preflight.json")
    tokenizer = AutoTokenizer.from_pretrained(old["model"], trust_remote_code=False)
    card = Path(old["model"]) / "README.md"
    card_line = (
        "Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, "
        "min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`"
    )
    assert card_line in card.read_text()
    pre = {
        "version": "native9b_quote_thinking_sampler8_v34",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "model": old["model"], "engine": deepcopy(old["engine"]),
        "sampling": {
            "legacy": {**deepcopy(old["sampling"]), "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0},
            "model_card": {
                "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
            },
        },
        "sampling_basis": {"path": str(card), "sha256": shared.sha(card), "exact_excerpt": card_line},
        "sha256": deepcopy(old["sha256"]), "rows": [],
        "protocol": [
            "Four already-inspected development questions: crisis, entity identity, positive control, future motivation.",
            "Both use native9b with enable_thinking=True, same full quote-system/user/evidence/hypothesis and seed.",
            "Only the sampling profile differs. Both total output budgets remain 4096, including reasoning.",
            "Model-card profile follows the locally frozen official 9B README general-thinking entry.",
            "This is a multi-parameter profile intervention, not a temperature-only or presence-penalty-only claim.",
            "Fresh paired legacy control, alternating order; v31 is descriptive, not the concurrent control.",
            "No input truncation, source selection, entity hint, answer repair, E-ID rebinding or extra budget.",
            "Missing terminator, empty final, length stop and other errors are retained, not salvaged.",
            "Completed final answers use the same frozen v27-source quote validation as v31.",
            "v33 canonical layout change is not injected into these frozen controls; it changes presentation only.",
            "No production switch, service restart, training, labels, or independent human-gold claim.",
        ],
    }
    source_map = {r["id"]: r for r in old["rows"]}
    for index, identity in enumerate(CASE_IDS):
        source = source_map[identity]
        row = {key: deepcopy(source[key]) for key in
               ("id", "question", "evidence_brief", "hypothesis", "messages", "sampling_seed")}
        row["prompt"] = deepcopy(source["prompts"]["think"])
        row["order"] = list(KINDS) if index % 2 == 0 else list(reversed(KINDS))
        actual = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        assert row["prompt"]["text"] == actual
        ids = tokenizer.encode(actual, add_special_tokens=True)
        assert shared.digest(json.dumps(ids)) == row["prompt"]["token_ids_sha256"]
        assert source["evidence_brief"] in actual and len(ids) + 4096 <= 12288
        pre["rows"].append(row)
    for path in (Path(__file__), V31 / "preflight.json", V32 / "run_plain_thinking.py",
                 V32 / "plain_protocol.py"):
        pre["sha256"][str(path)] = shared.sha(path)
    shared.save(HERE / "preflight.json", pre)
    print(json.dumps({"planned_calls": 8, "questions": 4,
                      "preflight_sha256": shared.sha(HERE / "preflight.json")}), flush=True)


def run():
    # Reuse only the immutable v32 exclusive GPU lock, asset hash checks,
    # continuous 60s idle gate and durable-run existence guard.
    harness.HERE = HERE
    harness.generate = generate
    harness.run()


def generate(pre):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    import vllm
    helpers = native.runtime_imports()
    llm = LLM(model=pre["model"], tokenizer=pre["model"], trust_remote_code=False, **pre["engine"])
    assert llm.llm_engine.vllm_config.lora_config is None
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    rows = []
    with (HERE / "calls.progress.jsonl").open("x", encoding="utf-8") as journal:
        for row in pre["rows"]:
            result = {"id": row["id"], "question": row["question"], "conditions": {}}
            for kind in row["order"]:
                item = {
                    "status": "error", "actual_prompt": row["prompt"]["text"],
                    "seed": row["sampling_seed"], "lora_id": None,
                    "sampling": deepcopy(pre["sampling"][kind]),
                }
                started = time.monotonic()
                try:
                    params = SamplingParams(**pre["sampling"][kind], seed=row["sampling_seed"])
                    actual_params = {k: getattr(params, k) for k in pre["sampling"][kind]}
                    assert actual_params == pre["sampling"][kind]
                    item["actual_sampling"] = actual_params
                    output = llm.generate([row["prompt"]["text"]], params, use_tqdm=False)[0]
                    assert output.lora_request is None and len(output.outputs) == 1
                    ids = tokenizer.encode(row["prompt"]["text"], add_special_tokens=True)
                    assert list(output.prompt_token_ids) == ids
                    choice = output.outputs[0]
                    item.update(
                        raw_output=choice.text, prompt_tokens=len(ids),
                        prompt_token_ids_sha256=shared.digest(json.dumps(ids)),
                        generated_tokens=len(choice.token_ids), finish_reason=choice.finish_reason,
                        returned_lora_id=None, seconds=time.monotonic() - started,
                    )
                    assert choice.finish_reason == "stop", "total budget exhausted; not accepted"
                    answer, chars = answer_after_thinking(choice.text, True)
                    item.update(answer_text=answer, reasoning_chars=chars)
                    item.update(native.process(answer, row, row["evidence_brief"], helpers), status="ok")
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                result["conditions"][kind] = item
                journal.write(json.dumps({"id": row["id"], "condition": kind, "record": item},
                                         ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"id": row["id"], "condition": kind, "status": item["status"],
                                  "tokens": item.get("generated_tokens"),
                                  "finish": item.get("finish_reason"),
                                  "action": item.get("grounded", {}).get("next_action")}), flush=True)
            rows.append(result)
    harness.verify(pre)
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
    harness.verify(pre)
    assert [r["id"] for r in gen["rows"]] == CASE_IDS
    tokenizer = AutoTokenizer.from_pretrained(pre["model"], trust_remote_code=False)
    helpers = native.runtime_imports()
    summary = {k: {"status": Counter(), "finish": Counter(), "action": Counter(),
                   "generated_tokens": 0} for k in KINDS}
    journal = [json.loads(line) for line in (HERE / "calls.progress.jsonl").read_text().splitlines()]
    assert len(journal) == 8
    index = 0
    for source, result in zip(pre["rows"], gen["rows"], strict=True):
        assert source["id"] == result["id"] and set(result["conditions"]) == set(KINDS)
        for kind in source["order"]:
            item = result["conditions"][kind]
            assert journal[index] == {"id": source["id"], "condition": kind, "record": item}
            index += 1
            assert item["actual_prompt"] == source["prompt"]["text"]
            assert item["seed"] == source["sampling_seed"]
            assert item["sampling"] == pre["sampling"][kind]
            summary[kind]["status"].update([item["status"]])
            summary[kind]["finish"].update([item.get("finish_reason", "generation_error")])
            summary[kind]["generated_tokens"] += item.get("generated_tokens", 0)
            if "prompt_tokens" in item:
                assert item["actual_sampling"] == pre["sampling"][kind]
                ids = tokenizer.encode(item["actual_prompt"], add_special_tokens=True)
                assert len(ids) == item["prompt_tokens"] == source["prompt"]["tokens"]
                assert shared.digest(json.dumps(ids)) == item["prompt_token_ids_sha256"] == source["prompt"]["token_ids_sha256"]
                assert len(ids) + 4096 <= 12288 and item["generated_tokens"] <= 4096
                assert item["lora_id"] is item["returned_lora_id"] is None
            if item["status"] == "ok":
                assert item["finish_reason"] == "stop"
                answer, chars = answer_after_thinking(item["raw_output"], True)
                assert answer == item["answer_text"] and chars == item["reasoning_chars"]
                replay = native.process(answer, source, source["evidence_brief"], helpers)
                assert all(item[key] == value for key, value in replay.items())
                summary[kind]["action"].update([item["grounded"]["next_action"]])
            elif item.get("finish_reason") == "length":
                assert item["generated_tokens"] == 4096 and "grounded" not in item
    shared.save(HERE / "postflight_v1.json", {
        "evaluation_only": True, "training_ready": False, "questions": 4, "calls": 8,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "summary": summary, "actual_prompts_tokens_seed_sampling_and_no_lora_verified": True,
        "total_4096_budget_both_verified": True, "all_calls_match_durable_journal": True,
        "frozen_quote_grounding_replayed": True, "semantic_review_complete": False,
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_sampler_probe.py prepare|run|check")
    operations[sys.argv[1]]()
