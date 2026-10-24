"""Versioned canonical snapshot and CPU-only training preflight."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RUN_ID = "asa_historical_sft_20260911_v55_attempt02"
CANON = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage")
CANON_RUN = CANON / "experiments" / RUN_ID
TRAIN_RUN = Path("${REMOTE_STORE}/exx_grounding_v1/runs") / RUN_ID
HERE = Path(__file__).resolve().parent
DATA_NAME = "exx_grounding_v3_sft_minimal559_glm53_20260903"
PARENT_NAME = "exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827"
BASE248 = Path("${REMOTE_STORE}/ASA-ArknightStoryAgent/model/qwen3.5-4b")
PARENT248 = Path("${REMOTE_STORE}/exx_grounding_v1/models") / PARENT_NAME
SOURCE_CONFIG = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/src/config/"
                     "llama_factory_exx_grounding_v3_sft_minimal559_glm53_full_a100_20260903.yaml")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def assets(folder):
    """Root-level inference assets only; never load training_args.bin."""
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in {
            ".json", ".safetensors", ".jinja", ".model", ".tiktoken", ".txt", ".py"
        } and not path.name.endswith("_results.json") and path.name not in {
            "trainer_state.json", "trainer_log.jsonl"
        }:
            result[path.name] = sha(path)
    assert any(name.endswith(".safetensors") for name in result)
    return result


def freeze241():
    import yaml
    assert HERE == CANON_RUN
    assert not (HERE / "snapshot.json").exists()
    source = CANON / "data" / DATA_NAME
    for line in (source / "final_checksums.sha256").read_text().splitlines():
        expected, name = line.split()
        assert Path(name).name == name and sha(source / name) == expected, name
    (HERE / "data").mkdir()
    for path in sorted(source.iterdir()):
        assert path.is_file() and not path.is_symlink()
        shutil.copy2(path, HERE / "data" / path.name)
    shutil.copy2(SOURCE_CONFIG, HERE / "original_config.yaml")
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    assert config["stage"] == "sft" and config["num_train_epochs"] == 1.0
    assert config["cutoff_len"] == 12000 and config["packing"] is False
    overrides = {
        "dataset_dir": str(TRAIN_RUN / "data"),
        "output_dir": str(TRAIN_RUN / "model"),
        "run_name": RUN_ID,
        "logging_dir": str(TRAIN_RUN / "tensorboard_unused"),
        "logging_steps": 1,
        "logging_first_step": True,
        "save_total_limit": None,
        "seed": 20260911,
        "data_seed": 20260911,
        "overwrite_cache": False,
        "disable_tqdm": True,
    }
    config.update(overrides)
    with (HERE / "train.yaml").open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    identities = {
        "base": assets(Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/model/qwen3.5-4b")),
        "parent": assets(CANON / "models" / PARENT_NAME),
    }
    save(HERE / "asset_hashes.json", identities)
    counts = {split: len(load(HERE / "data" / f"{split}.json")) for split in ("train", "val", "test")}
    assert counts == {"train": 910, "val": 95, "test": 0}, counts
    paths = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    save(HERE / "snapshot.json", {
        "run_id": RUN_ID, "created_at_utc": now(), "canonical_host": "${REMOTE_HOST}",
        "training_host": "${REMOTE_HOST}", "counts": counts,
        "config_overrides": overrides,
        "source_dataset": str(source), "source_config": str(SOURCE_CONFIG),
        "asset_source_paths": {
            "base": "${REMOTE_HOME}/ASA-ArknightStoryAgent/model/qwen3.5-4b",
            "parent": str(CANON / "models" / PARENT_NAME),
        },
        "sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted(paths)},
        "training_authorized": True, "shared_gpu_authorized": True,
        "production_changed": False, "human_gold": False, "new_glm_labels": 0,
        "parent_is_current_production_soda": False,
    })
    print(json.dumps({"snapshot_sha256": sha(HERE / "snapshot.json"), "counts": counts}), flush=True)


def verify_snapshot():
    snapshot = load(HERE / "snapshot.json")
    for relative, expected in snapshot["sha256"].items():
        path = HERE / relative
        assert path.resolve().is_relative_to(HERE.resolve())
        assert sha(path) == expected, relative
    return snapshot


def messages(row):
    roles = {"human": "user", "gpt": "assistant"}
    conversations = row["conversations"]
    assert len(conversations) == 2
    assert [c["from"] for c in conversations] == ["human", "gpt"]
    assert all(isinstance(c["value"], str) and c["value"] for c in conversations)
    return [{"role": roles[c["from"]], "content": c["value"]} for c in conversations]


def preflight248():
    import yaml
    assert HERE == TRAIN_RUN
    assert not (HERE / "preflight.json").exists()
    snapshot = verify_snapshot()
    expected = load(HERE / "asset_hashes.json")
    for kind, folder in (("base", BASE248), ("parent", PARENT248)):
        for name, digest in expected[kind].items():
            assert sha(folder / name) == digest, (kind, name)
    config = yaml.safe_load((HERE / "train.yaml").read_text())
    assert not Path(config["output_dir"]).exists(), "fresh output only"
    # These lightweight dataclasses/loaders do not allocate model weights.
    from llamafactory.hparams import ModelArguments, DataArguments
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    model_args = ModelArguments(model_name_or_path=config["model_name_or_path"], trust_remote_code=True)
    data_args = DataArguments(template=config["template"], cutoff_len=config["cutoff_len"])
    module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(module["tokenizer"], data_args)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=module["tokenizer"],
                                          processor=module.get("processor"), data_args=data_args)
    unbounded = SupervisedDatasetProcessor(template=template, tokenizer=module["tokenizer"],
                                          processor=module.get("processor"),
                                          data_args=replace(data_args, cutoff_len=1_000_000))
    rows, split_lengths, tasks = [], {}, Counter()
    for split in ("train", "val"):
        split_lengths[split] = []
        data = load(HERE / "data" / f"{split}.json")
        assert len(data) == snapshot["counts"][split]
        for index, row in enumerate(data):
            converted = messages(row)
            kwargs = dict(prompt=converted[:-1], response=converted[-1:],
                          system=row.get("system", ""), tools=row.get("tools", ""),
                          images=[], videos=[], audios=[])
            full_ids, full_labels = unbounded._encode_data_example(**kwargs)
            ids, labels = processor._encode_data_example(**kwargs)
            assert ids == full_ids and labels == full_labels, f"truncation:{split}:{index}"
            assert ids and len(ids) == len(labels) <= config["cutoff_len"]
            supervised = sum(value != -100 for value in labels)
            assert supervised > 0, f"empty_target:{split}:{index}"
            json.loads(row["conversations"][-1]["value"])
            tasks[(split, row.get("task_type", "unspecified"))] += 1
            rows.append({"split": split, "index": index, "tokens": len(ids), "target_tokens": supervised,
                         "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            split_lengths[split].append(len(ids))
        print(json.dumps({"split": split, "checked": len(data), "max_tokens": max(split_lengths[split])}),
              flush=True)
    import llamafactory, transformers, torch
    import liger_kernel
    code_roots = [Path(llamafactory.__file__).parent, Path(transformers.__file__).parent,
                  Path(liger_kernel.__file__).parent]
    code = {str(p): sha(p) for root in code_roots for p in sorted(root.rglob("*.py"))}
    save(HERE / "encoded_rows.json", rows)
    report = {
        "created_at_utc": now(), "snapshot_sha256": sha(HERE / "snapshot.json"),
        "counts": snapshot["counts"], "cutoff_len": config["cutoff_len"], "truncated_rows": 0,
        "max_tokens": {split: max(lengths) for split, lengths in split_lengths.items()},
        "task_counts": {f"{split}:{task}": n for (split, task), n in tasks.items()},
        "encoded_rows_sha256": sha(HERE / "encoded_rows.json"),
        "code_sha256": code,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "peft", "datasets", "trl", "llamafactory")},
        "actual_module_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "python": sys.executable, "human_semantic_audit": False,
        "fused_linear_cross_entropy_only": True,
        "previous_attempt": "asa_historical_sft_20260911_v55: OOM after 5 optimizer steps",
        "disk_free_bytes": shutil.disk_usage(HERE).free,
    }
    verify_snapshot()
    save(HERE / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "code_sha256"}), flush=True)


if __name__ == "__main__":
    {"freeze241": freeze241, "preflight248": preflight248}[sys.argv[1]]()
