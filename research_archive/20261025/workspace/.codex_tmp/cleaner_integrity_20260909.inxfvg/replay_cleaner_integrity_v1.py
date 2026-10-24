"""Read existing datasets and replay both implementations in memory only."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
sys.path.insert(0, str(HERE))
from soda_dataset_integrity import question_split_key


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def overlap(train, val):
    return len({question_split_key(r) for r in train} & {question_split_key(r) for r in val})


def main():
    old = load_module("cleaner_old", HERE / "cleaner_original.py")
    new = load_module("cleaner_candidate", HERE / "clean_soda_api_verifier_dataset.py")
    source_dirs = [ROOT / old.DEFAULT_INPUT_DIR, ROOT / old.DEFAULT_OUTPUT_DIR]
    outputs = []
    for directory in source_dirs:
        paths = [directory / name for name in ("train.json", "val.json")]
        assert all(p.is_file() for p in paths), str(directory)
        hashes = {str(p): sha(p) for p in paths}
        train, val = [json.loads(p.read_text()) for p in paths]
        records = train + val
        before = deepcopy(records)
        old_records, old_stats = old.clean_records(records)
        old_train, old_val = old.split_records(old_records, seed=20260601, val_ratio=0.08)
        retained, quarantine, new_stats = new.partition_records(records)
        assert records == before
        by_id = {r["id"]: r for r in records}
        assert all(r == by_id[r["id"]] for r in retained)
        assert all(q["record"] == by_id[q["record"]["id"]] for q in quarantine)
        changed = [
            {
                "id": a["id"], "prompt_key": (a.get("meta") or {}).get("prompt_key"),
                "answer_changed": a["conversations"][-1] != b["conversations"][-1],
                "verifier_changed": (a.get("meta") or {}).get("api_verifier") != (b.get("meta") or {}).get("api_verifier"),
            }
            for a, b in zip(records, old_records) if a != b
        ]
        try:
            new_train, new_val = new.split_records(retained, seed=20260601, val_ratio=0.08)
            split = {"status": "ok", "train": len(new_train), "val": len(new_val),
                     "question_overlap": overlap(new_train, new_val)}
        except ValueError as exc:
            split = {"status": "rejected_integrity_error", "error": str(exc)}
        assert all(sha(Path(p)) == digest for p, digest in hashes.items())
        outputs.append({
            "input_dir": str(directory), "source_sha256": hashes,
            "input_records": len(records), "stored_question_overlap": overlap(train, val),
            "old_replay": {
                "records": len(old_records), "stats": dict(old_stats),
                "changed_records": changed, "question_overlap": overlap(old_train, old_val),
            },
            "candidate_replay": {
                "stats": dict(new_stats), "split": split,
                "quarantined_ids": [q["record"]["id"] for q in quarantine],
                "retained_records_changed": 0, "quarantined_records_changed": 0,
            },
        })
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_ready": False, "datasets_written": False, "source_files_modified": False,
        "script_sha256": {name: sha(HERE / name) for name in (
            "cleaner_original.py", "clean_soda_api_verifier_dataset.py", "soda_dataset_integrity.py",
            "replay_cleaner_integrity_v1.py",
        )},
        "datasets": outputs,
        "limitations": [
            "In-memory replay only; historical datasets and judgments remain untouched.",
            "Quarantine is not a semantic adjudication or a negative training label.",
            "Retained examples are not approved for training; original protocol/label defects may remain.",
            "No generation, retrieval, production answer accuracy, model weight or service changed.",
        ],
    }
    with (HERE / "stored_cleaner_replay_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
