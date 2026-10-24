"""Four controlled4B generations from an actual canonical prompt suffix probe."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from run_fact_coverage_pairs_v1 import pair_dispatch
from run_native8b_v1 import GPU, observe

MATRIX = HERE / "susie_handoff4_v1.matrix.json"
PREFLIGHT = HERE / "susie_handoff4_preflight_v1.json"
OUTPUT = HERE / "susie_handoff4_ctx12288_v1"
LOG = HERE / "susie_handoff4_generation_v1.log"
RECEIPT = HERE / "susie_handoff4_launch_receipt_v1.json"
VALIDATION = HERE / "susie_handoff4_validation_v1.json"
CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
ARMS = [
    ("original_builder_prefix1024", False),
    ("original_builder_complete3072", False),
    ("seed_retained_complete3072", False),
    ("seed_retained_complete3072", True),
]


def build_rows(original, conditions, renderer):
    rows = []
    for label, reserve in ARMS:
        matches = [c for c in conditions
                   if c["scoring_condition"] == label and c["order_only_reserve6"] is reserve]
        assert len(matches) == 1, "missing or duplicate handoff condition"
        condition = matches[0]
        prompt = condition["runtime_prompt"]
        prefix = "<|im_start|>system\n" + original["system"].strip() + "<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n" + original["assistant_prefix"]
        assert prompt.startswith(prefix) and prompt.endswith(suffix), "unexpected prompt template"
        user = prompt[len(prefix):-len(suffix)]
        assert renderer(original["system"], user, assistant_prefix=original["assistant_prefix"]) == prompt
        assert condition["evidence_brief"] in prompt, "evidence/prompt mismatch"
        row = {k: deepcopy(original[k]) for k in (
            "source_id", "question", "hypothesis", "current_round", "max_retrieval_rounds",
            "sampling_seed", "system", "assistant_prefix", "template", "parent_record_id")}
        variant = "handoff_" + label + ("_order_reserve6" if reserve else "")
        row.update(
            id=original["parent_record_id"] + "::" + variant,
            renderer=variant, user=user, runtime_prompt=prompt, runtime_prompt_sha256=digest(prompt),
            evidence_brief=condition["evidence_brief"], evidence_sha256=digest(condition["evidence_brief"]),
            prompt_evidence=deepcopy(condition["selected"]), packing_trace=deepcopy(condition["packing_trace"]),
            evidence_id_to_corpus_id={p["evidence_id"]: p["source_id"]
                                      for p in condition["packing_trace"] if p["status"] == "included"},
            evaluation_only=True, training_ready=False,
        )
        rows.append(row)
    assert len({r["id"] for r in rows}) == 4
    assert len({r["runtime_prompt"] for r in rows}) == 4, "identical prompts must not be regenerated"
    return rows


def manifest(path):
    sys.path.insert(0, str(RUN))
    from generate_exx_predictions_transformers import model_asset_manifest
    return model_asset_manifest(path)


def dependencies():
    return {str(p): sha(p) for p in [
        Path(__file__), HERE / "test_susie_handoff4_v1.py", HERE / "run_fact_coverage_pairs_v1.py",
        HERE / "run_native8b_v1.py", HERE / "build_fact_coverage_pairs_v1.py",
        HERE / "verify_empty_quote_candidate_v1.py", RUN / "run_canonical_vllm_eval_v3.py",
        RUN / "generate_exx_predictions_transformers.py", RUN / "run_canonical_quote_eval_v2.py",
        RUN / "analyze_canonical_grounding_v4.py",
    ]}


def unchanged(pre, assets=False):
    assert sha(MATRIX) == pre["matrix_sha256"] and sha(CONFIG) == pre["canonical_config_sha256"]
    assert snapshot(SOURCE) == pre["source_hashes"] and dependencies() == pre["dependencies_sha256"]
    assert all(sha(Path(p)) == h for p, h in pre["source_artifacts_sha256"].items())
    if assets:
        assert manifest(Path(pre["base_path"])) == pre["base_assets"]
        assert manifest(Path(pre["adapter_path"])) == pre["adapter_assets"]


def prepare():
    assert not MATRIX.exists() and not PREFLIGHT.exists()
    assert counts(HERE / "susie_handoff4_tests_v1.xml") == {
        "tests": 8, "failures": 0, "errors": 0, "skipped": 0}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    from transformers import AutoTokenizer
    handoff_path = HERE / "susie_prompt_handoff_v1.json"
    handoff = load(handoff_path)
    handoff_pre = load(HERE / "susie_prompt_handoff_preflight_v1.json")
    assert handoff["preflight_sha256"] == sha(HERE / "susie_prompt_handoff_preflight_v1.json")
    assert all(sha(Path(p)) == h for p, h in handoff_pre["artifacts_sha256"].items())
    assert snapshot(SOURCE) == handoff_pre["runtime_source_sha256"]
    original = next(r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                    if r["question"] == handoff_pre["question"] and r["renderer"] == "original_prompt_v1")
    rows = build_rows(original, handoff["conditions"], render_qwen_chat_prompt)
    effective = deepcopy(load(RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json")["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == load(CONFIG)["generator"]
    release = ROOT / "release/ASA-ArknightStoryAgent"
    base, adapter = [release / effective["vllm"][k] for k in ("base_model_path", "lora_path")]
    base_assets, adapter_assets = manifest(base), manifest(adapter)
    previous = load(HERE / "paired116_ctx12288_v1/metadata.json")
    assert base_assets == previous["base_model_assets"] and adapter_assets == previous["adapter_assets"]
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    checked = []
    for row in rows:
        ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        assert len(ids) + 2048 <= 12288, "never truncate or drop a condition"
        checked.append({"id": row["id"], "input_tokens": len(ids), "seed": row["sampling_seed"],
                        "input_token_ids_sha256": digest(json.dumps(ids))})
    save(MATRIX, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True, "training_ready": False,
        "version": "susie_handoff4_v1", "rows": rows, "max_context": 12288, "max_new_tokens": 2048,
        "source_handoff_sha256": sha(handoff_path), "builder_sha256": sha(Path(__file__)),
        "protocol": [
            "One diagnosed development question, four explicit pipeline-suffix conditions; not four independent test questions.",
            "Use actual canonical renderer prompts unchanged, including the evidence each suffix selects and packs.",
            "No target/gold/source-review/old-answer text added to the model; target was used only in separate diagnostics.",
            "Same4B/LoRA/tokenizer/per-question seed/sampling; four unique prompts generated once each with no repair/retry.",
            "First condition reuses earlier1024 chain scores; no new truncated retrieval scoring is performed.",
            "Candidates change chain inputs/order and possibly hints; this is a pipeline-candidate contrast, not a single-token causal ablation.",
            "All output/parse/grounding failures retained; no default changes, new training or claim of E2E accuracy.",
        ],
    })
    pre = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True, "training_ready": False,
        "matrix_sha256": sha(MATRIX), "canonical_config_sha256": sha(CONFIG), "source_hashes": snapshot(SOURCE),
        "dependencies_sha256": dependencies(), "base_path": str(base), "adapter_path": str(adapter),
        "base_assets": base_assets, "adapter_assets": adapter_assets, "rows": checked, "gpu_uuid": GPU,
        "source_artifacts_sha256": {str(p): sha(p) for p in [
            handoff_path, HERE / "susie_prompt_handoff_preflight_v1.json",
            HERE / "susie_handoff4_tests_v1.xml", HERE / "paired116_ctx12288_v1/metadata.json",
            RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json",
        ]},
        "decoding": {"max_tokens": 2048, "temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.05},
    }
    save(PREFLIGHT, pre)
    print(json.dumps({"preflight_rows": 4, "max_input_tokens": max(r["input_tokens"] for r in checked),
                      "model_loaded": False}), flush=True)


def run():
    pre = load(PREFLIGHT)
    unchanged(pre)
    sys.path.insert(0, str(RUN))
    sys.path.insert(0, str(SOURCE))
    import run_canonical_vllm_eval_v3 as harness
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    import vllm
    rows, calls = load(MATRIX)["rows"], []
    real_ensure, real_llm = VllmRunner._ensure_engine, vllm.LLM
    constructors = []
    def tracked_llm(**kwargs):
        constructors.append(dict(kwargs))
        assert kwargs["enable_lora"] and kwargs["max_model_len"] == 12288
        assert Path(kwargs["model"]).resolve() == Path(pre["base_path"]).resolve()
        return real_llm(**kwargs)
    def ensure(self):
        engine, adapter = real_ensure(self)
        assert adapter is not None
        if not getattr(engine, "_asa_handoff4_dispatch", False):
            engine.generate = pair_dispatch(engine.generate, rows, calls)
            engine._asa_handoff4_dispatch = True
        return engine, adapter
    receipt = {"evaluation_only": True, "training_ready": False, "completed": False,
               "preflight_sha256": sha(PREFLIGHT), "rows": calls, "engine_constructor": constructors}
    old_argv = sys.argv
    sys.argv = [str(RUN / "run_canonical_vllm_eval_v3.py"),
                "--input", str(MATRIX), "--output-dir", str(OUTPUT), "--gpu-uuid", GPU,
                "--source-root", str(SOURCE), "--max-new-tokens", "2048", "--max-context", "12288"]
    VllmRunner._ensure_engine, vllm.LLM = ensure, tracked_llm
    try:
        harness.main()
        predictions = load(OUTPUT / "predictions.json")
        assert len(predictions) == len(calls) == 4 and len(constructors) == 1
        for row, pred, call, checked in zip(rows, predictions, calls, pre["rows"]):
            assert row["id"] == pred["id"] == call["id"] == checked["id"]
            assert call["lora_id"] == 1 and call["returned_lora_echo"] == [1]
            assert call["seed"] == checked["seed"] and call["status"] == "returned"
            assert pred["prompt_token_ids_sha256"] == checked["input_token_ids_sha256"]
            diagnostics = pred["runtime_generation_diagnostics"]
            assert diagnostics["input_truncated"] is diagnostics["output_budget_reduced"] is False
        unchanged(pre)
        receipt.update(completed=True, predictions_sha256=sha(OUTPUT / "predictions.json"))
    finally:
        sys.argv = old_argv
        VllmRunner._ensure_engine, vllm.LLM = real_ensure, real_llm
        if OUTPUT.is_dir():
            save(OUTPUT / "actual_dispatch_v1.json", receipt)


def launch():
    assert not any(p.exists() for p in (OUTPUT, LOG, RECEIPT, VALIDATION))
    pre = load(PREFLIGHT)
    unchanged(pre, assets=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    observations = []
    with Path("/tmp/asa_slot_feedback_" + GPU + ".lock").open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, idle_since = time.monotonic(), None
        while time.monotonic() - start < 180:
            observation = observe()
            now = time.monotonic()
            idle_since = None if not observation["idle"] else (now if idle_since is None else idle_since)
            observation.update(at_utc=datetime.now(timezone.utc).isoformat(),
                               idle_seconds=0 if idle_since is None else round(now - idle_since, 2))
            observations.append(observation)
            log.write(json.dumps(observation) + "\n")
            print(json.dumps(observation), flush=True)
            if idle_since is not None and now - idle_since >= 60:
                unchanged(pre)
                child = subprocess.Popen([sys.executable, str(Path(__file__)), "run"],
                                         cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                receipt = {"returncode": code, "child_pid": child.pid, "gpu_uuid": GPU,
                           "preflight_sha256": sha(PREFLIGHT), "idle_observations": observations,
                           "base_assets_after_equal": manifest(Path(pre["base_path"])) == pre["base_assets"],
                           "adapter_assets_after_equal": manifest(Path(pre["adapter_path"])) == pre["adapter_assets"]}
                save(RECEIPT, receipt)
                print(json.dumps({"generation_returncode": code}), flush=True)
                if code:
                    return code
                assert receipt["base_assets_after_equal"] and receipt["adapter_assets_after_equal"]
                unchanged(pre)
                subprocess.run([
                    sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
                    "--validator", "after", "--source-root", str(SOURCE), "--matrix", str(MATRIX),
                    "--predictions", str(OUTPUT / "predictions.json"), "--output", str(VALIDATION),
                ], check=True, env=dict(env, CUDA_VISIBLE_DEVICES=""))
                return 0
            time.sleep(10)
        save(RECEIPT, {"launched": False, "reason": "no_sustained_idle_window",
                       "idle_observations": observations, "preflight_sha256": sha(PREFLIGHT)})
        return 2


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in {"prepare", "launch", "run"}
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run":
        run()
    else:
        raise SystemExit(launch())
