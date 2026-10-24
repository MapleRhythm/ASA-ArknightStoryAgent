"""In-memory replay only: preserve data bytes and distinguish proposals from labels."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
DATA = ROOT / "data/processed/llama_factory"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    destination = HERE / "stored_followup_replay_v2.json"
    assert not destination.exists()
    old = module("followup_before", HERE / "builder_original.py")
    new = module("followup_after", HERE / "build_soda_api_verifier_dataset_v2.py")
    sources, groups, verdicts = {}, {}, {}
    for index in range(3):
        source = DATA / f"student_rollout_quote80_sft_v1_200_teacher_replay_v1_verifier_shards/shard{index}"
        verifier = DATA / f"student_rollout_quote80_sft_v1_200_api_verifier_v1_shard{index}/api_verifier_records.jsonl"
        for path in (source / "audit_records.jsonl", verifier):
            sources[str(path)] = sha(path)
        current = old.load_audit_groups(source)
        assert groups.keys().isdisjoint(current)
        groups.update(current)
        current_verdicts = old.load_existing_verifier(verifier)
        assert verdicts.keys().isdisjoint(current_verdicts)
        verdicts.update(current_verdicts)
    all_outputs, summaries = {}, {}
    for name, builder in (("before", old), ("candidate", new)):
        outputs, stats = [], Counter()
        for key, group in groups.items():
            teacher = builder.record_for_role(group, kto_tag=True)
            if teacher is None:
                continue
            if builder.infer_task_type(teacher) != "conclusion_generation":
                outputs.extend(deepcopy(group))
                continue
            review = verdicts.get(key)
            if not review or review.get("error"):
                continue
            untouched = deepcopy(group)
            outputs.extend(builder.relabel_group(key, group, review, stats))
            assert group == untouched, "Replay changed source objects"
        counts = Counter()
        for row in outputs:
            payload = new.strict_json_object(builder.response_text(row)) or {}
            counts[(row.get("task_type"), row.get("kto_tag"), payload.get("next_action"))] += 1
        summary = {
            "records": len(outputs), "stats": dict(stats),
            "by_task_tag_action": [
                {"task_type": k[0], "kto_tag": k[1], "action": k[2], "count": v}
                for k, v in counts.items()
            ],
        }
        try:
            train, val = new.split_records(outputs, seed=20260531, val_ratio=0.08)
        except ValueError as exc:
            summary["split"] = {"status": "rejected", "error": str(exc)}
        else:
            overlap = {new.question_split_key(r) for r in train} & {new.question_split_key(r) for r in val}
            summary["split"] = {"status": "mechanically_valid_not_training_approved", "overlap": len(overlap)}
        all_outputs[name] = {r["id"]: r for r in outputs}
        assert len(all_outputs[name]) == len(outputs)
        summaries[name] = summary
    before, after = all_outputs["before"], all_outputs["candidate"]
    assert after.keys() <= before.keys(), "This patch must not create a new positive record ID"
    dropped, changed, proposals = [], [], []
    for rid, row in before.items():
        payload = new.strict_json_object(old.response_text(row)) or {}
        if rid in after and after[rid] == row:
            continue
        assert row["kto_tag"] is True and payload.get("next_action") == "retrieve_more"
        key = row["meta"]["prompt_key"]
        proposed = new.normalize_grounded_action_payload(
            payload, question=new.extract_question(new.prompt_text(row)), prompt=new.prompt_text(row),
        )
        item = {
            "id": rid, "prompt_key": key,
            "question": new.extract_question(new.prompt_text(row)),
            "old_follow_up": payload.get("follow_up_hypothesis"),
            "normalized_candidate_follow_up": proposed.get("follow_up_hypothesis"),
            "normalization_changed_old_target": proposed != payload,
            "retained_response_unchanged": (
                new.strict_json_object(new.response_text(after[rid])) == payload if rid in after else None
            ),
            "new_projection_provenance": (
                after[rid].get("meta", {}).get("verifier_target_projection") if rid in after else None
            ),
        }
        (dropped if rid not in after else changed).append(item)
        if rid not in after:
            proposals.append({
                **item, "source_group_ids": [r.get("id") for r in groups[key]],
                "unreviewed_proposal": proposed, "training_ready": False,
                "requires_new_review": True, "inherits_kto_tag": False,
            })
        else:
            assert item["retained_response_unchanged"] and item["new_projection_provenance"]
    raw_changes = []
    for key, group in groups.items():
        for row in group:
            payload = new.strict_json_object(new.response_text(row)) or {}
            follow = payload.get("follow_up_hypothesis")
            if payload.get("next_action") != "retrieve_more" or not isinstance(follow, dict):
                continue
            question, prompt_text = new.extract_question(new.prompt_text(row)), new.prompt_text(row)
            a = old.normalize_follow_up_payload(follow, question=question, prompt=prompt_text)
            b = new.normalize_follow_up_payload(follow, question=question, prompt=prompt_text)
            if a != b:
                raw_changes.append({
                    "id": row.get("id"), "prompt_key": key, "before": a, "candidate": b,
                })
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "datasets_written": False,
        "source_sha256": sources, "source_objects_and_bytes_unchanged": True,
        "original_sha256": sha(HERE / "builder_original.py"),
        "candidate_sha256": sha(HERE / "build_soda_api_verifier_dataset_v2.py"),
        "replay_script_sha256": sha(Path(__file__)),
        "groups": len(groups), "verdicts": len(verdicts), "replay": summaries,
        "dropped_retrieve_positive_count": len(dropped), "dropped": dropped,
        "changed_retrieve_positive_count": len(changed), "changed": changed,
        "raw_followup_cleanup_count": len(raw_changes), "raw_followup_cleanup": raw_changes,
        "all_other_outputs_unchanged": True,
        "limitations": [
            "Only three stored shards; not a complete historical data audit or a fresh retrieval evaluation.",
            "Old model verdicts are reused only for mechanical replay and are not re-approved.",
            "Dropped targets are neither relabeled negative nor silently cleaned into approved positives.",
            "Proposal file is an audit artifact, not a training dataset or human-gold source.",
            "Other action branches and the semantics of retained targets still require independent review.",
        ],
    }
    save(destination, report)
    save(HERE / "followup_reaudit_proposals_v2.json", {
        "created_at_utc": report["created_at_utc"], "training_ready": False,
        "replay_sha256": sha(destination), "proposals": proposals,
    })
    print(json.dumps({
        "groups": len(groups), "replay": summaries,
        "dropped_retrieve_positive_count": len(dropped),
        "changed_retrieve_positive_count": len(changed),
        "raw_followup_cleanup_count": len(raw_changes),
        "training_ready": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
