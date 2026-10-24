from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline"
if not BASE.is_dir():
    BASE = HERE.parent / "e2e_v23/final_v3"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    visible_path = BASE / "visible_answer_analysis.json"
    auto_path = BASE / "semantic58_content_v3_glm53_retry2.json"
    notes_path = HERE / "source_review_v26.json"
    auto, visible, notes = load(auto_path), load(visible_path), load(notes_path)
    assert sha(visible_path) == notes["baseline_visible_analysis_sha256"]
    assert auto["source_analysis_sha256"] == sha(visible_path)
    sources = {r["id"]: r for r in visible["results"]}
    assert len(auto["results"]) == len(sources) == 58
    for note in notes["reviews"]:
        row = sources[note["id"].split("::")[0]]
        chunks = re.split(r"(?:^|\n)\[(E\d+)\]\s*", row["evidence_brief"])
        blocks = dict(zip(chunks[1::2], chunks[2::2]))
        for ref in note["source_quotes"]:
            assert ref["quote"] in blocks[ref["evidence_id"]], ref
    projection = []
    for row in auto["results"]:
        if row["status"] == "ok":
            label = row["judgement"]
            kind = "model_judge"
        else:
            assert row["id"] == "fixed58-031"
            label = notes["reviews"][0]["judgement"]
            kind = "assistant_source_review_not_human_gold"
        projection.append({
            "id": row["id"], "source": kind, "original_auto_status": row["status"],
            "verdict": label["verdict"],
            "context_sufficiency": label["context_sufficiency"],
            "action_appropriateness": label["action_appropriateness"],
        })
    counts = dict(Counter(row["verdict"] for row in projection))
    assert counts == {
        "supported_partial": 15, "unsupported": 14,
        "supported_complete": 13, "no_answer": 16,
    }
    assert Counter(r["source"] for r in projection) == {
        "model_judge": 57, "assistant_source_review_not_human_gold": 1,
    }
    tests = load(HERE / "cpu_verification_v2.json")
    replay = load(HERE / "replay_after_withdrawal_v1.json")
    assert tests["all_passed"]
    assert replay["statuses"] == {"ok": 56, "not_replayed": 2}
    assert replay["changed_actions"] == 0
    report = {
        "version": "canonical_no_minirag_fixed58_checkpoint_v26",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "model_judgements_are_provisional": True, "independent_human_gold": False,
        "canonical_root": "241:${REMOTE_HOME}/ASA-ArknightStoryAgent",
        "basis": {
            "visible_analysis_sha256": sha(visible_path),
            "automatic_audit_sha256": sha(auto_path),
            "source_review_sha256": sha(notes_path),
            "cpu_verification_sha256": sha(HERE / "cpu_verification_v2.json"),
            "restored_replay_sha256": sha(HERE / "replay_after_withdrawal_v1.json"),
        },
        "automatic_original": {
            "statuses": auto["statuses"],
            "verdicts": dict(Counter(r["judgement"]["verdict"] for r in auto["results"] if r["status"] == "ok")),
        },
        "annotated_baseline_projection": {
            "counts": counts,
            "complete_fraction": "13/58 (22.4%), provisional mixed-source diagnostic",
            "complete_or_partial_fraction": "28/58 (48.3%), not full-answer correctness",
            "sufficient_context": dict(Counter(r["verdict"] for r in projection if r["context_sufficiency"] == "sufficient")),
            "rows": projection,
        },
        "withdrawn_candidate": {
            "changed_replay_ids": ["fixed58-011"],
            "newly_certified_correct_answers": 0,
            "reason": "Unsupported Sami/Sargon subject attribution despite exact copied text and a whole-answer judge positive.",
            "backup": "outputs/rollback_backups/20260909_quote_rebind_withdrawn_v26",
        },
        "retained_changes": [
            "release batch errors produce process exit 1 and retain every result row",
            "root glm_exx_semantic_reward.py restored from 241 HEAD, undoing exact historical v2 reversion",
            "MiniRAG remains disabled; no model/config/service switch",
        ],
        "limitations": [
            "58 distinct development questions; the historical 79 are repeated evidence states, not 79 independent E2E questions.",
            "The content audit's sufficient-context count is not an independently measured retrieval recall rate.",
            "Baseline visible text was reconstructed from saved final evidence and packing lengths; original exact prompt token IDs were not saved by that E2E run.",
            "Only two items were fully source-reviewed in this checkpoint. Other automatic complete/partial/unsupported labels are not certified.",
            "Counterfactual replay covers quote grounding of 56 strictly parseable final outputs, not fresh generation or all earlier rounds.",
            "The source-reversion repair restores test compatibility, not proof of improved training or deployed answer accuracy.",
        ],
    }
    save(HERE / "checkpoint_v26.json", report)
    print(json.dumps({
        "rows": 58, "counts": counts,
        "sufficient": sum(r["context_sufficiency"] == "sufficient" for r in projection),
        "test_counts": {name: value["counts"] for name, value in tests["tests"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
