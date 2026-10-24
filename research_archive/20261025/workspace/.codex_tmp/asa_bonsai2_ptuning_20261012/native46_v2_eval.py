"""Evaluate the completed v2 encoder against immutable native46 tasks.

Only v2 is generated. Baseline answers and semantic decisions remain historical.
The unchanged native46 worker receives no answer labels or baseline outputs.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import native46_eval as n

TASK_SHA = "370ed18099030240f054f6ca6f260b7e028d31570d6deaae9fdd4eb8d3ca1ec9"
ADAPTER_SHA = "b79a2f0ba824cb42e021f24afefcd4bd3b032e52a9889651cdbab422d02f787a"
REVIEW_SHA = "e37cc5f76088ece25dc812f4e53e59cc6ba80590eb281891785e9d7b7fda0211"
BASELINE_SHAS = {
    "fourb.results.jsonl": "3993aa4359e53c78895c4ca3f2f3c6b09bc6568b3278839639d1af7e63788f60",
    "bonsai_shard0.results.jsonl": "a1842582b77e1ab7cbddaa5e3393afac0e546522f97dfba120b1a604a4d2916b",
    "bonsai_shard1.results.jsonl": "3f16156ff2159f9296be6ba97f0ccec935682ed4ff192fce6623c00b3ee0e5f2",
}
CACHE_DIRS = {
    "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp",
}


def match_validation(tasks, validation):
    assert len(tasks) == len(validation) == 46
    assert [t["position"] for t in tasks] == list(range(1, 47))
    assert len({t["id"] for t in tasks}) == 46
    for task, row in zip(tasks, validation):
        n.check_task(task)
        assert row["split"] == "val" and not row["truncated"]
        for key in ("id", "question", "round", "documents"):
            assert task[key] == row[key], f"validation mismatch: {key}"
        assert task["bonsai_prompt"] == row["prompt"]


def prepare(args):
    from audit_native46 import audit as audit_baseline
    from targeted_pt_v2 import verify_merged

    root = Path(__file__).resolve().parent
    data = args.experiment / "merged_data"
    run_dir = args.experiment / "trained_v2"
    data_manifest = verify_merged(data)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["training_complete"] and run["optimizer_steps"] == 750
    assert run["backbone_frozen"] and run["backbone_bytewise_unchanged"]
    assert run["adapter_reload_embeddings_exact"] and run["no_truncation"]
    assert run["data_sha256"] == data_manifest["data_sha256"]
    assert run["targeted_merged_preflight"]["manifest_sha256"] == n.file_sha(data / "manifest.json")
    assert n.file_sha(run_dir / "final_adapter/prompt_encoder.safetensors") == ADAPTER_SHA
    baseline_manifest = json.loads((args.baseline / "manifest.json").read_text())
    assert n.file_sha(args.baseline / "tasks.jsonl") == TASK_SHA == baseline_manifest["tasks_sha256"]
    assert n.file_sha(root / "native46_eval.py") == baseline_manifest["script_sha256"]
    assert n.file_sha(root / "backend.py") == baseline_manifest["backend_sha256"]
    assert n.file_sha(args.review) == REVIEW_SHA
    for filename, expected in BASELINE_SHAS.items():
        assert n.file_sha(args.baseline / filename) == expected
    audit_baseline(args.baseline, require_complete=True)
    tasks = n.rows(args.baseline / "tasks.jsonl")
    match_validation(tasks, n.rows(data / "val.jsonl"))
    assert baseline_manifest["data_val_sha256"] == n.file_sha(data / "val.jsonl")
    manifest = dict(baseline_manifest)
    # Do not carry v1 provenance fields over under their previous meanings.
    manifest.pop("reviewed_dataset_audit_sha256", None)
    manifest.update(
        kind="native46_v2_generation_with_hash_bound_historical_baselines",
        prepared_utc=n.now(), expected_generations=46,
        bonsai_run=str(run_dir), bonsai_run_sha256=n.file_sha(run_dir / "run.json"),
        bonsai_adapter_sha256=ADAPTER_SHA,
        bonsai_adapter_config_sha256=n.file_sha(run_dir / "final_adapter/adapter_config.json"),
        merged_data_manifest_sha256=n.file_sha(data / "manifest.json"),
        wrapper_sha256=n.file_sha(__file__),
        baseline_run=str(args.baseline),
        baseline_manifest_sha256=n.file_sha(args.baseline / "manifest.json"),
        historical_baseline_outputs=BASELINE_SHAS,
        historical_content_review=str(args.review),
        historical_content_review_sha256=REVIEW_SHA,
        reuse_previous_outputs=True, reuse_previous_outputs_for=["fourb", "bonsai_pt_v1"],
        newly_generated_arm="bonsai_pt_v2",
        semantic_adjudication_complete=False,
    )
    args.out.mkdir(exist_ok=False)
    shutil.copyfile(args.baseline / "tasks.jsonl", args.out / "tasks.jsonl")
    n.save_json(args.out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(args.out), "new_generations": 46,
                      "adapter_sha256": ADAPTER_SHA, "tasks_sha256": TASK_SHA}), flush=True)


def cache_environment(out):
    assert out.stat().st_dev != Path.home().stat().st_dev, "caches must use data disk"
    assert shutil.disk_usage(out).free >= 20 * 1024 ** 3
    cache = out / "runtime_cache"
    assert not cache.is_symlink()
    cache.mkdir(mode=0o700, exist_ok=True)
    result = {}
    for variable, name in CACHE_DIRS.items():
        directory = cache / name
        assert not directory.is_symlink()
        directory.mkdir(mode=0o700, exist_ok=True)
        assert directory.resolve().parent == cache.resolve()
        assert directory.stat().st_dev == out.stat().st_dev
        result[variable] = str(directory.resolve())
    return result


def launch(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert n.file_sha(__file__) == manifest["wrapper_sha256"]
    assert n.file_sha(args.out / "tasks.jsonl") == TASK_SHA
    assert n.file_sha(Path(manifest["bonsai_run"]) / "run.json") == manifest["bonsai_run_sha256"]
    assert n.file_sha(Path(manifest["bonsai_run"]) / "final_adapter/prompt_encoder.safetensors") == ADAPTER_SHA
    path = args.out / "launch.json"
    assert not path.exists(), "refuse duplicate launch"
    for shard in (0, 1):
        assert not (args.out / f"bonsai_shard{shard}.status.json").exists()
        assert not (args.out / f"bonsai_shard{shard}.results.jsonl").exists()
        assert not (args.out / f"bonsai_shard{shard}.log").exists()
    for gpu in (1, 2):
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True).strip())
        assert free >= 22000, f"GPU {gpu}: {free} MiB free, need 22000"
    caches = cache_environment(args.out)
    record = dict(started_utc=n.now(), workers={}, cache_environment=caches,
                  only_new_v2_generated=True, historical_baselines_reused=True,
                  production_modified=False, other_jobs_stopped=False)
    # Reserve the launch before spawning; never silently restart failed output.
    n.save_json(path, record)
    for shard, gpu in enumerate((1, 2)):
        env = dict(os.environ, **caches, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, "-u", str(Path(n.__file__).resolve()), "worker",
                   "--out", str(args.out), "--arm", "bonsai", "--shard", str(shard)]
        with (args.out / f"bonsai_shard{shard}.log").open("x") as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["workers"][str(shard)] = dict(pid=child.pid, gpu=gpu, command=command)
        n.save_json(path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def audit_records(tasks, values, shard):
    expected = n.shard_tasks(tasks, shard)
    assert len(values) <= len(expected)
    assert [v["id"] for v in values] == [t["id"] for t in expected[:len(values)]]
    for task, value in zip(expected, values):
        for key in ("position", "id", "question", "round", "question_group"):
            assert value[key] == task[key]
        assert value["arm"] == "bonsai" and value["shard"] == shard
        assert value["evidence_count"] == 20 and value["input_truncated"] is False
        assert value["answer_validation"] is False and value["protocol_prefill"] == ""
        assert value["prompt_sha256"] == task["bonsai_prompt_sha256"]
        assert 0 < value["input_tokens"] <= 32768 - 16 - 1024
        assert len(value["output_token_ids"]) <= 1024
        assert value["finish_reason"] in ("stop", "length")
        assert value["output_reached_length_limit"] == (value["finish_reason"] == "length")
        assert n.parse_raw(value["raw_output"], "") == value["parsed_output_for_offline_inspection_only"]


def audit(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert n.file_sha(args.out / "tasks.jsonl") == TASK_SHA == manifest["tasks_sha256"]
    tasks = n.rows(args.out / "tasks.jsonl")
    assert len(tasks) == 46 and [t["id"] for t in tasks] == manifest["task_order"]
    for task in tasks:
        n.check_task(task)
    states, values = {}, []
    for shard in (0, 1):
        name = f"bonsai_shard{shard}"
        state_path = args.out / f"{name}.status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"stage": "not_started"}
        states[name] = state
        path = args.out / f"{name}.results.jsonl"
        records = n.rows(path) if path.exists() else []
        audit_records(tasks, records, shard)
        if state["stage"] == "completed":
            assert len(records) == state["completed"] == state["expected"] == 23
            assert n.file_sha(path) == state["results_sha256"]
            assert state["source_sha256"] == "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
        values.extend(records)
    complete = all(s["stage"] == "completed" for s in states.values())
    if args.require_complete:
        assert complete and len(values) == 46, "generation not complete"
    # These are protocol diagnostics only, never answer accuracy.
    result = dict(checked_utc=n.now(), completed=len(values), expected=46, workers=states,
                  all_generation_complete=complete, semantic_adjudication_complete=False,
                  valid_json=sum(isinstance(v["parsed_output_for_offline_inspection_only"], dict) for v in values),
                  output_length_limits=sum(v["output_reached_length_limit"] for v in values),
                  parsed_actions=dict(Counter(
                      v["parsed_output_for_offline_inspection_only"].get("next_action", "missing")
                      if isinstance(v["parsed_output_for_offline_inspection_only"], dict) else "invalid_json"
                      for v in values)))
    if args.require_complete:
        n.save_json(args.out / "generation_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for name in ("experiment", "baseline", "review", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("launch")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
