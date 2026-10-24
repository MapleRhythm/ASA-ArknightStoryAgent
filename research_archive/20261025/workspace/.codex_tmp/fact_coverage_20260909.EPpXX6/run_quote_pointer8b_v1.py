"""Isolated native8B source-pointer experiment using the verified8B dispatch.

Only annotation/output reference protocol changes versus the saved native8B
control. Never edits live config/model/source or old predictions.
"""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_native8b_v1 as base
from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from quote_pointer_protocol_v1 import make_user

MATRIX = HERE / "quote_pointer8b58_v1.matrix.json"
PREFLIGHT = HERE / "quote_pointer8b58_execution_preflight_v1.json"
OUTPUT = HERE / "quote_pointer8b58_ctx12288_v1"
LOG = HERE / "quote_pointer8b58_generation_v1.log"
RECEIPT = HERE / "quote_pointer8b58_launch_receipt_v1.json"
VALIDATION = HERE / "quote_pointer8b58_validation_v1.json"
TEST_XML = HERE / "quote_pointer_combined_tests_v1.xml"
PROTOCOL_TESTS = HERE / "quote_pointer_protocol_tests_v1.xml"
CONTROL = HERE / "qwen3_8b_native58_v1.matrix.json"
CONTROL_REPLAY = HERE / "question_echo_native8b58_replay_v1.json"


def dependencies():
    files = [
        Path(__file__), HERE / "run_native8b_v1.py", HERE / "quote_pointer_protocol_v1.py",
        HERE / "test_quote_pointer_protocol_v1.py", HERE / "test_quote_pointer_harness_v1.py",
        HERE / "analyze_quote_pointer8b_v1.py", HERE / "check_quote_pointer8b_v1.py",
        HERE / "build_fact_coverage_pairs_v1.py", HERE / "verify_empty_quote_candidate_v1.py",
        HERE / "test_native8b_harness_v1.py",
        RUN / "analyze_canonical_grounding_v4.py", RUN / "generate_exx_predictions_transformers.py",
    ]
    return {str(p): sha(p) for p in files}


def build_row(original, tokenizer):
    user, annotated, catalog = make_user(original)
    row = deepcopy(original)
    messages = [{"role": "system", "content": original["system"]},
                {"role": "user", "content": user}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    row.update(
        id=original["parent_record_id"] + "::quote_pointer8b_v1",
        renderer="quote_pointer8b_v1", user=user, annotated_evidence=annotated,
        quote_catalog=catalog, native_messages=messages, native_template_enable_thinking=False,
        runtime_prompt=rendered, runtime_prompt_sha256=digest(rendered), assistant_prefix="",
    )
    return row


def check_row(row, original, tokenizer):
    expected = build_row(original, tokenizer)
    assert row == expected, "changed question/evidence/seed or non-protocol prompt delta"
    assert row["template"] == "runtime_raw"
    assert digest(row["evidence_brief"]) == row["evidence_sha256"]
    assert type(row["sampling_seed"]) is int
    ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
    assert ids == tokenizer.encode(row["runtime_prompt"], add_special_tokens=False)
    assert len(ids) + 2048 <= 12288, "do not truncate or drop an overflowing row"
    assert row["annotated_evidence"] in row["runtime_prompt"]
    return {"id": row["id"], "input_tokens": len(ids),
            "prompt_token_ids_sha256": digest(json.dumps(ids)),
            "source_pointers": len(row["quote_catalog"])}


def configure_base():
    # Process-local bindings only; the original source file and control run
    # remain byte-for-byte unchanged.
    for key in ("MATRIX", "PREFLIGHT", "OUTPUT", "LOG", "RECEIPT", "VALIDATION", "TEST_XML"):
        setattr(base, key, globals()[key])
    base.check_row = check_row
    base.dependencies = dependencies


def preflight():
    assert not any(p.exists() for p in (MATRIX, PREFLIGHT, OUTPUT))
    assert counts(PROTOCOL_TESTS) == {"tests": 36, "failures": 0, "errors": 0, "skipped": 0}
    test_counts = counts(TEST_XML)
    assert test_counts == {"tests": 44, "failures": 0, "errors": 0, "skipped": 0}
    assert snapshot(SOURCE) == load(HERE / "checkpoint_v19_state.json")["runtime_source_sha256"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base.MODEL), local_files_only=True,
                                              trust_remote_code=False)
    originpath = HERE / "fact_coverage_pairs116_v1.matrix.json"
    originals = [r for r in load(originpath)["rows"] if r["renderer"] == "original_prompt_v1"]
    controls = load(CONTROL)["rows"]
    assert len(originals) == len(controls) == 58
    # The current guard has already replayed all native8B controls unchanged.
    replay = load(CONTROL_REPLAY)
    assert len(replay["results"]) == 58
    oldpre = load(HERE / "qwen3_8b_execution_preflight_v1.json")
    assert sha(CONTROL) == oldpre["matrix_sha256"]
    assets = base.assets()
    assert assets == oldpre["model_assets"]
    rows, checked = [], []
    for original, control in zip(originals, controls):
        for key in ("question", "system", "user", "evidence_brief", "evidence_sha256",
                    "prompt_evidence", "hypothesis", "current_round", "max_retrieval_rounds", "sampling_seed"):
            assert original[key] == control[key], "native8B control input mismatch"
        row = build_row(original, tokenizer)
        rows.append(row)
        checked.append(check_row(row, original, tokenizer))
    assert len({r["id"] for r in rows}) == 58
    limitations = [
        "Development frozen-evidence experiment, not heldout answer accuracy or RAG recall.",
        "Same native8B model/template/no-LoRA and seeds as saved control; no fresh interleaving.",
        "Only lossless source annotation and pointer-output instructions/schema change.",
        "Every source character preserved; quote pointers are visible annotations, not hidden evidence.",
        "Binding selects no text for the model, makes no semantic decision and never repairs bad IDs.",
        "Existing per-quote80, per-fact160, total400 and grounding conditions are not relaxed.",
        "All parse/bind/grounding failures stay in58; no retries or selective regeneration.",
        "More accepted answers do not prove more correct answers; source review remains required.",
        "No gold/reference answer/prior judgement/source-review correction supplied to generation.",
        "No training, runtime source edit, default switch or server restart.",
    ]
    matrix = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "quote_pointer8b58_v1", "evaluation_only": True, "training_ready": False,
        "rows": rows, "source_matrix_sha256": sha(originpath), "native_control_sha256": sha(CONTROL),
        "max_context": 12288, "max_new_tokens": 2048,
        "builder_sha256": sha(Path(__file__)), "predeclared_protocol": limitations,
    }
    # Reject overflow before writing any matrix/preflight, never drop cases.
    assert all(r["input_tokens"] <= 10240 for r in checked)
    save(MATRIX, matrix)
    kwargs = base.requested_runner_kwargs()
    assert {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()} == oldpre["runner_kwargs"]
    frozen = {str(p): sha(p) for p in [originpath, CONTROL, CONTROL_REPLAY,
                                     HERE / "checkpoint_v19_state.json", PROTOCOL_TESTS]}
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "matrix_sha256": sha(MATRIX),
        "rows": checked, "canonical_config_sha256": sha(base.CONFIG),
        "source_hashes": snapshot(SOURCE), "model_assets": assets,
        "dependencies_sha256": dependencies(), "source_artifacts_sha256": frozen,
        "harness_tests": test_counts, "harness_test_xml_sha256": sha(TEST_XML),
        "runner_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
        "native_template_sha256": digest(tokenizer.chat_template),
        "limitations": limitations,
    })
    print(json.dumps({"rows": 58, "max_input_tokens": max(r["input_tokens"] for r in checked),
                      "quote_pointers": sum(r["source_pointers"] for r in checked),
                      "gpu_loaded": False}), flush=True)


def ensure_frozen(pre):
    base.ensure_frozen(pre)
    assert all(sha(Path(p)) == h for p, h in pre["source_artifacts_sha256"].items())
    assert sha(TEST_XML) == pre["harness_test_xml_sha256"]


def launch():
    assert not any(p.exists() for p in (OUTPUT, LOG, RECEIPT, VALIDATION))
    pre = load(PREFLIGHT)
    ensure_frozen(pre)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=base.GPU, VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    observations = []
    lockpath = Path("/tmp/asa_slot_feedback_" + base.GPU + ".lock")
    with lockpath.open("a") as lock, LOG.open("x", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert base.assets() == pre["model_assets"]
        idle_since, start = None, time.monotonic()
        while time.monotonic() - start < 180:
            observation = base.observe()
            now = time.monotonic()
            idle_since = (now if idle_since is None else idle_since) if observation["idle"] else None
            observation.update(at_utc=datetime.now(timezone.utc).isoformat(),
                               idle_seconds=round(now - idle_since, 2) if idle_since is not None else 0)
            observations.append(observation)
            log.write(json.dumps(observation) + "\n")
            print(json.dumps(observation), flush=True)
            if idle_since is not None and now - idle_since >= 60:
                ensure_frozen(pre)
                child = subprocess.Popen([sys.executable, str(Path(__file__)), "run"],
                                         cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
                save(RECEIPT, {"child_pid": child.pid, "returncode": code, "gpu_uuid": base.GPU,
                               "idle_observations": observations, "preflight_sha256": sha(PREFLIGHT),
                               "ended_at_utc": datetime.now(timezone.utc).isoformat()})
                print(json.dumps({"generation_returncode": code}), flush=True)
                return code
            time.sleep(10)
        save(RECEIPT, {"launched": False, "reason": "no_sustained_idle_window",
                       "idle_observations": observations, "evaluation_only": True})
        return 2


if __name__ == "__main__":
    configure_base()
    assert len(sys.argv) == 2 and sys.argv[1] in {"preflight", "run", "launch"}
    if sys.argv[1] == "preflight":
        preflight()
    elif sys.argv[1] == "run":
        ensure_frozen(load(PREFLIGHT))
        base.run()
        ensure_frozen(load(PREFLIGHT))
    else:
        raise SystemExit(launch())
