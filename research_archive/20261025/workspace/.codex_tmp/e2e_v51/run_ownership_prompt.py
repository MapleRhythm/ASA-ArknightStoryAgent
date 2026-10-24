"""Generic relation/speaker instruction, isolated from production prompts.

Reuse the already audited v41 generation/recording implementation. Adaptations
are temporary Python callbacks in this experiment process, never source edits.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V49 = HERE.parent / "fallback_trace_retention_integration_20260910.v49"
V50 = HERE.parent / "relation_claim_sensitivity_20260910.v50"
sys.path.insert(0, str(V49))
import run_trace_integration as integration

reader = integration.probe.reader
shared = integration.shared
ORIGINAL_INVOKE, ORIGINAL_SOURCES = reader.invoke, reader.sources
KINDS = ("baseline", "ownership_rules")
PARAMS = {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.05}
RULES = (
    "\n逐条核对事实的主体和关系：引用对话时辨认实际说话者及代词所指，"
    "不要把同一证据中另一个人物的经历归给问题中的人物。"
    "亲属、制造者或因果关系的方向必须有原文依据，关系词或人名出现本身不算支持。"
    "每条事实只保留来源能够确认的内容，不补充未给出的主体、关系或动机；"
    "不能确认核心问题时按原 action_set 继续检索或弃答。"
)


def augment(prompt, kind):
    marker = "<|im_end|>\n<|im_start|>user\n"
    assert prompt.count(marker) == 1 and RULES not in prompt
    if kind == "baseline":
        return prompt
    assert kind == "ownership_rules"
    before, after = prompt.split(marker, 1)
    result = before + RULES + marker + after
    assert result.replace(RULES, "", 1) == prompt
    assert result.split(marker, 1)[1] == after
    return result


def frozen_sources():
    original = ORIGINAL_SOURCES()
    return {(case, kind): value for (case, condition), value in original.items()
            if condition == "control" for kind in KINDS}


def bridged_invoke(pipeline, helpers, item, source):
    """Canonical pipeline supplies its unmodified prompt; bridge one send."""
    generate = pipeline.generator.generate
    calls = 0

    def send(prompt, **kwargs):
        nonlocal calls
        assert prompt == item["canonical_prompt"] and kwargs == PARAMS
        assert calls == 0, "No retry or final-direct fallback at round1"
        calls += 1
        return generate(item["prompt"], **kwargs)

    pipeline.generator.generate = send
    try:
        return ORIGINAL_INVOKE(pipeline, helpers, item, source)
    finally:
        pipeline.generator.generate = generate


@contextmanager
def adapted_reader():
    names = ("HERE", "verify", "sources", "invoke")
    original = {name: getattr(reader, name) for name in names}
    reader.HERE, reader.verify, reader.sources, reader.invoke = HERE, verify, frozen_sources, bridged_invoke
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(reader, name, value)


def verify(pre):
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    assert all(shared.sha(Path(p)) == digest for p, digest in pre["sha256"].items())


def prepare():
    assert not (HERE / "preflight.json").exists()
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    closed = shared.load(V50 / "checkpoint_v50.json")
    assert closed["independent_fresh_process_replay_equal"] and not closed["production_changed"]
    assert all(shared.sha(Path(p)) == h for p, h in closed["basis_sha256"].items())
    old = shared.load(integration.probe.V41 / "preflight.json")
    pre = {k: deepcopy(old[k]) for k in (
        "base_model", "adapter", "tokenizer", "runtime", "generator", "query_config", "pipeline_config",
    )}
    assert "soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3" in pre["adapter"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    helpers = reader.runtime_imports()
    pipeline = reader.pipeline_for(pre, helpers, reader.retrieval.NoGeneration())
    sources = ORIGINAL_SOURCES()
    originals = [r for r in old["conditions"] if r["condition"] == "control"]
    assert len(originals) == 6
    pre.update(
        version="isolated_speaker_relation_system_prompt_v51",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        production_changed=False, training_ready=False, prompt_rule=RULES,
        verified_nvidia_mapping=reader.retrieval.numeric_mapping(),
        conditions=[], pairs=[], cpu_synthetic_protocol_checks=[],
        protocol=[
            "Only six v41 CONTROL hypotheses/retrieval inputs; no v39 routing recovery or new retrieval.",
            "Two new paired seeds, alternating baseline/ownership rule condition order, 24 fresh first-round calls.",
            "One generic system-message addition only; full user message/evidence/schema and legacy assistant brace prefill unchanged.",
            "No answers, named-story examples, historical raw responses or source-review annotations added to prompts.",
            "Actual current canonical first-round reader and validators; generator bridge changes only the experimental sent prompt.",
            "Same production4B SODA LoRA; 2048 output, .1/.8/1.05 sampling, context12288, original12000-char evidence packing.",
            "CPU synthetic abstain is protocol-only and is never substituted for real generation.",
            "No retry, overflow truncation, second round, final-direct fallback, automatic E-ID repair or weakened checks.",
            "Development comparison, not independent evaluation or E2E; action counts are not accuracy.",
            "Fresh >=60-second continuous GPU2 idle interval under original lock; preserve all failures and raw outputs.",
            "No production source/config/default model/prompt changes, training, service restart, commit or push.",
        ],
    )
    for source in originals:
        rendered = reader.render(pipeline, helpers, source["hypothesis"], {
            **sources[(source["id"], "control")], "question": source["question"],
        })
        assert rendered["prompt"] == source["prompt"]
        assert rendered["visible_evidence"] == source["visible_evidence"]
        for kind in KINDS:
            prompt = augment(source["prompt"], kind)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            item = {
                "id": source["id"], "condition": kind, "source_condition": "control",
                "question": source["question"], "hypothesis": deepcopy(source["hypothesis"]),
                "canonical_prompt": source["prompt"], "canonical_prompt_sha256": source["prompt_sha256"],
                "visible_evidence": source["visible_evidence"],
                "visible_evidence_sha256": shared.digest(source["visible_evidence"]),
                "prompt": prompt, "prompt_sha256": shared.digest(prompt),
                "token_ids_sha256": shared.digest(json.dumps(ids)), "prompt_tokens": len(ids),
                "context_fits": len(ids) + 2048 <= pre["runtime"]["max_model_len"],
            }
            pre["conditions"].append(item)
            dry = reader.ReplayGenerator(pre["generator"], item, {
                "runner_requests": [{"kwargs": PARAMS}],
                "sanitized_raw_output": '{"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
            })
            dry_pipeline = reader.pipeline_for(pre, helpers, dry)
            conclusion, trace = bridged_invoke(dry_pipeline, helpers, item, sources[(source["id"], "control")])
            assert dry.calls == 1 and conclusion["next_action"] == "abstain"
            pre["cpu_synthetic_protocol_checks"].append({
                "id": item["id"], "condition": kind, "synthetic_only": True,
                "calls": dry.calls, "trace_sha256": integration.probe.staged.object_digest(trace),
            })
    for repeat in range(2):
        for index, source in enumerate(originals):
            pre["pairs"].append({
                "id": source["id"], "repeat": repeat, "pair_id": f"{source['id']}:seed{repeat}",
                "sampling_seed": int(shared.digest(f"ownership-rules-v51:{repeat}:{source['id']}")[:8], 16) % (2**31),
                "condition_order": list(KINDS if (repeat + index) % 2 == 0 else KINDS[::-1]),
            })
    assert all(r["context_fits"] for r in pre["conditions"])
    paths = [
        Path(__file__), HERE / "README.md", V49 / "preflight.json", V49 / "checkpoint_v49.json",
        V49 / "run_trace_integration.py", V50 / "checkpoint_v50.json",
        integration.probe.V41 / "preflight.json", integration.probe.V41 / "run_canonical_reader_v2.py",
        reader.V40 / "retrieval.json", reader.V40 / "run_retrieval_probe.py",
        reader.V27 / "run_reader_router.py",
    ]
    pre["sha256"] = {str(p): shared.sha(p) for p in paths}
    shared.save(HERE / "preflight.json", pre)
    print({"planned_calls": 24, "cpu_protocol_checks": 12,
           "longest_input": max(r["prompt_tokens"] for r in pre["conditions"]),
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def generate_all():
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    assert reader.retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    with adapted_reader():
        reader.generate_all()


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
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


def check():
    assert not (HERE / "postflight_v1.json").exists()
    pre, saved = shared.load(HERE / "preflight.json"), shared.load(HERE / "generations.json")
    verify(pre)
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(line) for line in (HERE / "generations.progress.jsonl").read_text().splitlines()]
    assert journal == saved["attempts"] and len(journal) == 24
    helpers, frozen = reader.runtime_imports(), frozen_sources()
    indexed = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    expected = [(pair, kind) for pair in pre["pairs"] for kind in pair["condition_order"]]
    summary = {kind: {"status": Counter(), "actions": Counter(), "finish": Counter()} for kind in KINDS}
    replayed = 0
    for result, (pair, kind) in zip(journal, expected, strict=True):
        assert (result["pair_id"], result["condition"]) == (pair["pair_id"], kind)
        item = indexed[(pair["id"], kind)]
        summary[kind]["status"].update([result["status"]])
        assert len(result["calls"]) <= 1
        if result["calls"]:
            call = result["calls"][0]
            assert call["seed"] == pair["sampling_seed"] and call["prompt_sha256"] == item["prompt_sha256"]
            assert call["max_tokens"] == 2048
            if "token_ids_sha256" in call:
                assert call["token_ids_sha256"] == item["token_ids_sha256"]
                assert call["prompt_tokens"] == item["prompt_tokens"]
                assert call["lora_id"] == call["returned_lora_id"] == 1
            summary[kind]["finish"].update([call.get("finish_reason", "generation_error")])
        if result["status"] == "ok":
            generator = reader.ReplayGenerator(pre["generator"], item, result)
            pipeline = reader.pipeline_for(pre, helpers, generator)
            conclusion, trace = bridged_invoke(pipeline, helpers, item, frozen[(pair["id"], kind)])
            assert conclusion == result["conclusion"] and trace == result["retrieval_trace"]
            assert generator.calls == 1
            replayed += 1
            summary[kind]["actions"].update([conclusion["next_action"]])
    assert sum(len(r["calls"]) for r in journal) == saved["actual_call_count"]
    verify(pre)
    shared.save(HERE / "postflight_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "actual_call_count": saved["actual_call_count"], "canonical_replays": replayed,
        "summary_not_accuracy": summary, "production_changed": False, "training_ready": False,
        "requires_visible_source_review_before_any_promotion": True,
    })
    print({"canonical_replays": replayed, "summary_not_accuracy": summary}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_ownership_prompt.py prepare|run|check")
    operations[sys.argv[1]]()
