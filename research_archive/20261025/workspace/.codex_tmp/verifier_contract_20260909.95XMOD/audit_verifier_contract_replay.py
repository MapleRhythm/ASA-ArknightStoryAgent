"""In-memory replay of stored judgments. Does not build or write training data."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old = load_module("soda_contract_before", HERE / "original.py")
new = load_module("soda_contract_after", HERE / "build_soda_api_verifier_dataset.py")
stored = ROOT / "data/processed/llama_factory"
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_of_truth": str(ROOT), "evaluation_only": True, "training_ready": False,
    "before_sha256": sha(HERE / "original.py"),
    "candidate_sha256": sha(HERE / "build_soda_api_verifier_dataset.py"),
    "shared_integrity_sha256": sha(HERE / "soda_dataset_integrity.py"),
    "audit_sha256": sha(Path(__file__)),
    "source_files": [], "shards": [], "replay": {},
    "limitations": [
        "Stored judgments are reused for a mechanical audit, not independently re-approved.",
        "No API/model/GPU/retrieval run; no training dataset or adapter is overwritten.",
        "Quote membership and schema validity are not claim entailment or answer correctness.",
        "Question split is global within this replay; per-shard splits must not be concatenated blindly.",
        "Current-source replay is not proof of which historical script bytes produced stored targets.",
    ],
}
groups = {}
verdicts = {}
question_shards = defaultdict(set)
for index in range(3):
    source_dir = stored / f"student_rollout_quote80_sft_v1_200_teacher_replay_v1_verifier_shards/shard{index}"
    verifier_path = stored / f"student_rollout_quote80_sft_v1_200_api_verifier_v1_shard{index}/api_verifier_records.jsonl"
    current_groups = old.load_audit_groups(source_dir)
    current_verdicts = old.load_existing_verifier(verifier_path)
    for path in (source_dir / "audit_records.jsonl", verifier_path):
        report["source_files"].append({"path": str(path), "sha256": sha(path)})
    assert not set(groups) & set(current_groups), "Duplicate prompt keys across input shards"
    groups.update(current_groups)
    verdicts.update(current_verdicts)
    for group in current_groups.values():
        for record in group:
            question_shards[new.question_split_key(record)].add(index)
    report["shards"].append({
        "index": index, "groups": len(current_groups), "verdicts": len(current_verdicts),
    })
report["questions_in_multiple_input_shards"] = sum(len(s) > 1 for s in question_shards.values())

for version, module in (("before", old), ("candidate", new)):
    stats, outputs = Counter(), []
    positive_issues = Counter()
    bad_positive_examples = []
    for key, group in groups.items():
        teacher = module.record_for_role(group, kto_tag=True)
        if teacher is None:
            continue
        if module.infer_task_type(teacher) != "conclusion_generation":
            outputs.extend(group)
            stats["keep:non_conclusion"] += len(group)
            continue
        vr = verdicts.get(key)
        if not vr or vr.get("error"):
            stats["drop:unverified"] += 1
            continue
        outputs.extend(module.relabel_group(key, group, vr, stats))
    summary = {
        "records": len(outputs), "stats": dict(sorted(stats.items())),
        "by_task_tag_action": {}, "label_conflict": None,
    }
    counts = Counter()
    for row in outputs:
        payload = new.strict_json_object(module.response_text(row))
        counts[(row.get("task_type"), row.get("kto_tag"), (payload or {}).get("next_action"))] += 1
        if row.get("kto_tag") is not True or row.get("task_type") != "conclusion_generation":
            continue
        issues = (
            ["invalid_json_object"] if payload is None
            else new.direct_candidate_issues(payload, module.prompt_text(row))
            if payload.get("next_action") == "answer_directly" else []
        )
        positive_issues.update(issues)
        if issues:
            bad_positive_examples.append({"id": row["id"], "issues": issues})
    summary["by_task_tag_action"] = [
        {"task_type": key[0], "kto_tag": key[1], "action": key[2], "count": count}
        for key, count in counts.items()
    ]
    summary["positive_direct_contract_issues"] = dict(positive_issues)
    summary["positive_contract_examples"] = bad_positive_examples
    try:
        new.assert_no_label_conflicts(outputs)
    except ValueError as exc:
        summary["label_conflict"] = str(exc)
    try:
        train, val = module.split_records(outputs, seed=20260531, val_ratio=0.08)
    except ValueError as exc:
        summary["split_error"] = str(exc)
    else:
        summary["global_split"] = {
            "train_records": len(train), "val_records": len(val),
            "overlapping_question_keys": sorted(
                {new.question_split_key(r) for r in train} & {new.question_split_key(r) for r in val}
            ),
        }
    report["replay"][version] = summary

previous = json.loads((HERE / "stored_judgment_replay_v1.json").read_text())
for version in report["replay"]:
    assert report["replay"][version] == previous["replay"][version], "Shared-helper extraction changed replay"
with (HERE / "stored_judgment_replay_v2.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({
    "questions_in_multiple_input_shards": report["questions_in_multiple_input_shards"],
    "replay": {name: {key: value for key, value in summary.items()
                     if key not in {"positive_contract_examples"}}
               for name, summary in report["replay"].items()},
}, ensure_ascii=False, indent=2))
