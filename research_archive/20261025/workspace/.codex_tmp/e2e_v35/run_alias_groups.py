"""Frozen production-LoRA reader, 8 questions x 2 seeds x 2 alias conditions."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V29 = HERE.parent / "reader_chain_owner_20260910.v29"
V32 = HERE.parent / "reader_plain_thinking_20260910.v32"
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
CATALOG = ROOT / "indexes/arknights_story/operator_aliases.json"
sys.path.insert(0, str(V29))
import run_chain_owner as producer
sys.path.insert(0, str(V32))
import run_plain_thinking as harness
from alias_group_protocol import groups_for, inject, verify
from selector_protocol import CASE_IDS
shared = producer.shared


def prepare():
    from transformers import AutoTokenizer
    old = shared.load(V27 / "preflight.json")
    catalog = shared.load(CATALOG)
    tokenizer = AutoTokenizer.from_pretrained(old["tokenizer"], trust_remote_code=False)
    pre = {key: deepcopy(old[key]) for key in
           ("base_model", "adapter", "tokenizer", "runtime", "generator", "sha256")}
    pre.update(
        version="production_lora_alias_groups32_v35",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_only=True, training_ready=False, production_changed=False,
        catalog_path=str(CATALOG), catalog_sha256=shared.sha(CATALOG),
        protocol=[
            "Eight already-inspected development questions, two predeclared seeds each, two conditions: 32 calls.",
            "Frozen production 4B LoRA and original legacy chat wrapper, including opening-brace prefill.",
            "No native-thinking template or 9B/model-card sampler is transferred into the production LoRA.",
            "Candidate adds only catalog-defined identity links among names already present in the hypothesis.",
            "No new names, family links, story facts, expected answer, review or old output is provided.",
            "Original complete prompt is preserved around one insertion at the end of the user message.",
            "Full evidence, IDs, member markers, order, hypothesis and quotation policy stay unchanged.",
            "Questions without at least two linked present names have byte-identical prompts as noise controls.",
            "Paired seeds, alternating order, 2048 total output tokens, context 12288, no input truncation.",
            "All failures and unchanged-prompt variation retained; no deterministic-by-seed assumption.",
            "Same immutable v29 generation harness and v27 quote validation; only experimental globals bind a new output directory.",
            "No production alias prompt change, model switch, training or labels; no human-gold accuracy claim.",
        ],
        rows=[],
    )
    for repeat in range(2):
        for index, source in enumerate(old["rows"]):
            groups = groups_for(catalog, source["hypothesis"])
            row = {key: deepcopy(source[key]) for key in
                   ("id", "question", "hypothesis", "evidence_brief", "empty_reader_prompt", "control_prompt")}
            candidate = inject(source["control_prompt"], groups)
            verify(source["control_prompt"], candidate, groups)
            row.update(
                repeat=repeat, pair_id=f"{source['id']}:seed{repeat}", alias_groups=groups,
                candidate_evidence=source["evidence_brief"], candidate_prompt=candidate,
                sampling_seed=int(shared.digest(f"alias-groups-v35:{repeat}:{source['id']}")[:8], 16) % (2**31),
                condition_order=["control", "candidate"] if (repeat + index) % 2 == 0 else ["candidate", "control"],
            )
            for kind in ("control", "candidate"):
                prompt = row[f"{kind}_prompt"]
                assert row["evidence_brief"] in prompt
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                assert len(ids) + 2048 <= pre["runtime"]["max_model_len"]
                row[f"{kind}_prompt_sha256"] = shared.digest(prompt)
                row[f"{kind}_prompt_tokens"] = len(ids)
                row[f"{kind}_token_ids_sha256"] = shared.digest(json.dumps(ids))
            pre["rows"].append(row)
    for path in (CATALOG, Path(__file__), HERE / "alias_group_protocol.py",
                 HERE / "test_alias_group_protocol.py", V29 / "run_chain_owner.py",
                 V29 / "chain_owner_protocol.py", V32 / "run_plain_thinking.py", V32 / "plain_protocol.py"):
        pre["sha256"][str(path)] = shared.sha(path)
    pre["unique_question_alias_groups"] = {r["id"]: r["alias_groups"] for r in pre["rows"][:8]}
    shared.save(HERE / "preflight.json", pre)
    print(json.dumps({
        "planned_calls": 32, "unique_questions": 8,
        "alias_groups": pre["unique_question_alias_groups"],
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
    }, ensure_ascii=False), flush=True)


def run():
    # Keep old experiment scripts immutable; reuse the GPU guard and the
    # already-audited production-LoRA generation recorder in this process only.
    producer.HERE = HERE
    harness.HERE = HERE
    harness.generate = lambda pre: producer.run()
    harness.run()


def check():
    from transformers import AutoTokenizer
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    assert gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert gen["postflight_files_unchanged"] and gen["actual_call_count"] == 32
    harness.verify(pre)
    assert [r["id"] for r in gen["rows"]] == CASE_IDS * 2
    catalog = shared.load(CATALOG)
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=False)
    helpers = producer.runtime_imports()
    summary = {k: {"status": Counter(), "action": Counter()} for k in ("control", "candidate")}
    noise = []
    count = 0
    for source, result in zip(pre["rows"], gen["rows"], strict=True):
        assert source["pair_id"] == result["pair_id"]
        assert source["alias_groups"] == groups_for(catalog, source["hypothesis"])
        verify(source["control_prompt"], source["candidate_prompt"], source["alias_groups"])
        assert source["candidate_evidence"] == source["evidence_brief"]
        assert len(result["calls"]) == 2
        for kind, call in zip(source["condition_order"], result["calls"], strict=True):
            count += 1
            item = result["conditions"][kind]
            assert call["condition"] == kind and call["pair_id"] == source["pair_id"]
            assert call["seed"] == source["sampling_seed"] and call["max_tokens"] == 2048
            prompt = source[f"{kind}_prompt"]
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            assert call["actual_prompt"] == prompt
            assert call["prompt_sha256"] == source[f"{kind}_prompt_sha256"] == shared.digest(prompt)
            if "prompt_tokens" in call:
                assert len(ids) == call["prompt_tokens"] == source[f"{kind}_prompt_tokens"]
                assert shared.digest(json.dumps(ids)) == call["prompt_token_ids_sha256"] == source[f"{kind}_token_ids_sha256"]
                assert call["lora_id"] == call["returned_lora_id"] and call["lora_id"] is not None
                assert call["generated_tokens"] <= 2048 and len(ids) + 2048 <= pre["runtime"]["max_model_len"]
            summary[kind]["status"].update([item["status"]])
            if item["status"] == "ok":
                assert call["finish_reason"] == "stop"
                replay = producer.process(item["raw_output"], source, source["evidence_brief"], helpers)
                assert all(item[key] == value for key, value in replay.items())
                summary[kind]["action"].update([item["grounded"]["next_action"]])
        if not source["alias_groups"]:
            control, candidate = result["conditions"]["control"], result["conditions"]["candidate"]
            noise.append({
                "pair_id": source["pair_id"],
                "same_raw": control.get("raw_output") == candidate.get("raw_output"),
                "same_display": control.get("grounded", {}).get("answer") == candidate.get("grounded", {}).get("answer"),
                "same_action": control.get("grounded", {}).get("next_action") == candidate.get("grounded", {}).get("next_action"),
            })
    assert count == 32
    report = {
        "version": "production_lora_alias_groups_postflight_v35", "evaluation_only": True,
        "training_ready": False, "production_changed": False, "questions": 8, "calls": 32,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "summary": summary, "no_new_names_or_story_facts_added": True,
        "actual_prompts_token_ids_seed_lora_budget_verified": True,
        "complete_evidence_and_frozen_grounding_replayed": True,
        "byte_identical_prompt_noise_controls": noise, "semantic_review_complete": False,
    }
    shared.save(HERE / "postflight_v1.json", report)
    print(json.dumps({"summary": summary, "noise_controls": noise}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_alias_groups.py prepare|run|check")
    operations[sys.argv[1]]()
