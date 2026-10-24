"""New4B single-Q protocol on all four previously frozen handoff prompts."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_susie_handoff4_v1 as base
from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from single_quote_pointer_protocol_v1 import make_user

MATRIX = HERE / "single_pointer_handoff4_v1.matrix.json"
PREFLIGHT = HERE / "single_pointer_handoff4_preflight_v1.json"
OUTPUT = HERE / "single_pointer_handoff4_ctx12288_v1"
LOG = HERE / "single_pointer_handoff4_generation_v1.log"
RECEIPT = HERE / "single_pointer_handoff4_launch_receipt_v1.json"
VALIDATION = HERE / "single_pointer_handoff4_validation_v1.json"
TEST_XML = HERE / "single_pointer_handoff4_tests_v1.xml"
CONTROL = HERE / "susie_handoff4_v1.matrix.json"
CONTROL_PREFLIGHT = HERE / "susie_handoff4_preflight_v1.json"
CONTROL_POSTFLIGHT = HERE / "susie_handoff4_postflight_v1.json"
ORIGINAL_DEPENDENCIES = base.dependencies


def dependencies():
    return {
        **ORIGINAL_DEPENDENCIES(),
        **{str(p): sha(p) for p in [
            Path(__file__), HERE / "single_quote_pointer_protocol_v1.py",
            HERE / "quote_pointer_protocol_v1.py", HERE / "test_single_quote_pointer_v1.py",
            HERE / "test_single_pointer_handoff4_v1.py", HERE / "analyze_single_pointer_handoff4_v1.py",
            HERE / "analyze_quote_pointer8b_v1.py", HERE / "check_single_pointer_handoff4_v1.py",
        ]},
    }


def build_row(original, renderer):
    user, annotated, catalog = make_user(original)
    row = deepcopy(original)
    prompt = renderer(original["system"], user, assistant_prefix=original["assistant_prefix"])
    row.update(
        id=original["id"] + "::single_pointer_v1",
        renderer=original["renderer"] + "_single_pointer_v1",
        control_record_id=original["id"], user=user, annotated_evidence=annotated,
        quote_catalog=catalog, runtime_prompt=prompt, runtime_prompt_sha256=digest(prompt))
    return row


def configure():
    for key in ("MATRIX", "PREFLIGHT", "OUTPUT", "LOG", "RECEIPT", "VALIDATION"):
        setattr(base, key, globals()[key])
    base.dependencies = dependencies


def prepare():
    assert not any(p.exists() for p in (MATRIX, PREFLIGHT, OUTPUT))
    test_counts = counts(TEST_XML)
    assert test_counts == {"tests": 32, "failures": 0, "errors": 0, "skipped": 0}
    source = snapshot(SOURCE)
    assert source == load(HERE / "checkpoint_v21_state.json")["runtime_source_sha256"]
    oldpre, oldpost = load(CONTROL_PREFLIGHT), load(CONTROL_POSTFLIGHT)
    assert oldpost["protocol_verified"] and oldpost["states"] == 4
    assert oldpre["matrix_sha256"] == sha(CONTROL)
    assert oldpre["source_hashes"] == source and oldpre["canonical_config_sha256"] == sha(base.CONFIG)
    assert all(sha(Path(p)) == h for p, h in oldpost["artifacts_sha256"].items())
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    from transformers import AutoTokenizer
    controls = load(CONTROL)["rows"]
    assert len(controls) == 4 and len({r["question"] for r in controls}) == 1
    rows = [build_row(r, render_qwen_chat_prompt) for r in controls]
    assert len({r["runtime_prompt"] for r in rows}) == 4
    assets_base = base.manifest(Path(oldpre["base_path"]))
    assets_adapter = base.manifest(Path(oldpre["adapter_path"]))
    assert assets_base == oldpre["base_assets"] and assets_adapter == oldpre["adapter_assets"]
    adapter = Path(oldpre["adapter_path"])
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").is_file() else Path(oldpre["base_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    checks = []
    for row, control in zip(rows, controls):
        for field in ("question", "system", "hypothesis", "sampling_seed", "assistant_prefix",
                      "evidence_brief", "evidence_sha256", "prompt_evidence", "packing_trace",
                      "evidence_id_to_corpus_id", "current_round", "max_retrieval_rounds"):
            assert row[field] == control[field]
        assert row["assistant_prefix"] == "{"
        assert row["annotated_evidence"] in row["runtime_prompt"]
        ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        assert len(ids) + 2048 <= 12288, "never drop or truncate an overflowing condition"
        checks.append({"id": row["id"], "seed": row["sampling_seed"], "input_tokens": len(ids),
                       "input_token_ids_sha256": digest(json.dumps(ids)),
                       "quote_pointers": len(row["quote_catalog"])})
    save(MATRIX, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "version": "single_pointer_handoff4_v1",
        "rows": rows, "max_context": 12288, "max_new_tokens": 2048,
        "control_matrix_sha256": sha(CONTROL),
        "protocol": [
            "One diagnosed development question, all four frozen suffix states; not four independent test questions.",
            "Only add lossless Q markers and change the quote reference schema/instructions.",
            "Each Q already uniquely identifies both exact visible text and E ownership; output only Q, never repair dual-pointer output.",
            "Fresh four generations, same4B/LoRA/seed/decoding/context as controls, no selective regeneration.",
            "Retain same hypothesis, hints and every source character; no prior answer/judge/source-review sent.",
            "Assemble the known assistant prefill with the raw continuation before strict duplicate-key JSON parsing.",
            "All existing quote80/fact160/total400 and semantic-source limitations remain; no acceptance-as-accuracy.",
            "No production source/config/model changes, training, label rewrite or service restart.",
        ],
    })
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": sha(MATRIX), "canonical_config_sha256": sha(base.CONFIG),
        "source_hashes": source, "dependencies_sha256": dependencies(),
        "base_path": oldpre["base_path"], "adapter_path": oldpre["adapter_path"],
        "base_assets": assets_base, "adapter_assets": assets_adapter, "rows": checks,
        "decoding": oldpre["decoding"], "gpu_uuid": base.GPU,
        "source_artifacts_sha256": {str(p): sha(p) for p in [
            CONTROL, CONTROL_PREFLIGHT, CONTROL_POSTFLIGHT, TEST_XML, HERE / "checkpoint_v21_state.json"]},
        "evaluation_only": True, "training_ready": False,
    })
    assert snapshot(SOURCE) == source
    print(json.dumps({"states": 4, "questions": 1, "gpu_loaded": False, "rows": checks}), flush=True)


def launch():
    assert not any(p.exists() for p in (OUTPUT, LOG, RECEIPT, VALIDATION))
    pre = load(PREFLIGHT)
    base.unchanged(pre, assets=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=base.GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    observations = []
    with Path("/tmp/asa_slot_feedback_" + base.GPU + ".lock").open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, idle_since = time.monotonic(), None
        while time.monotonic() - start < 180:
            observation = base.observe()
            now = time.monotonic()
            idle_since = None if not observation["idle"] else (now if idle_since is None else idle_since)
            observation.update(at_utc=datetime.now(timezone.utc).isoformat(),
                               idle_seconds=0 if idle_since is None else round(now - idle_since, 2))
            observations.append(observation)
            log.write(json.dumps(observation) + "\n")
            print(json.dumps(observation), flush=True)
            if idle_since is not None and now - idle_since >= 60:
                base.unchanged(pre)
                child = subprocess.Popen([sys.executable, str(Path(__file__)), "run"],
                                         cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                save(RECEIPT, {
                    "returncode": code, "child_pid": child.pid, "gpu_uuid": base.GPU,
                    "preflight_sha256": sha(PREFLIGHT), "idle_observations": observations,
                    "base_assets_after_equal": base.manifest(Path(pre["base_path"])) == pre["base_assets"],
                    "adapter_assets_after_equal": base.manifest(Path(pre["adapter_path"])) == pre["adapter_assets"],
                })
                print(json.dumps({"generation_returncode": code}), flush=True)
                return code
            time.sleep(10)
        save(RECEIPT, {"launched": False, "reason": "no_sustained_idle_window",
                       "idle_observations": observations, "preflight_sha256": sha(PREFLIGHT)})
        return 2


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in {"prepare", "launch", "run"}
    configure()
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run":
        base.run()
    else:
        raise SystemExit(launch())
