"""Append the completed audit and independently inspected chronology; no relabel."""
from collections import Counter
from datetime import datetime, timezone
import json

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot


def main():
    report = ROOT / "docs/CANONICAL_COMPLETED_FACT_AUDIT_20260909_leU0LZ_v18.md"
    output = HERE / "checkpoint_v18_state.json"
    review = HERE / "partial_source_review_v2.json"
    assert report.is_file() and not output.exists() and not review.exists()
    audited = load(HERE / "blind_audits_v1.json")
    post = load(HERE / "blind_audits_postflight_v1.json")
    assert post["protocol_verified"] and post["unique_statuses"] == {"ok": 115, "error": 11}
    assert post["audit_report_sha256"] == sha(HERE / "blind_audits_v1.json")
    errors = [r for r in audited["results"] if r["status"] != "ok"]
    limited = [r for r in errors if r.get("finish_reason") == "length"]
    assert len(errors) == 11 and len(limited) == 3
    assert all(r["response_model"] == "glm-5.3" and not r.get("raw_response")
               and r["usage"]["completion_tokens"] == 16384 for r in limited)
    assert sum(r["reproduced_validation_failure"] for r in post["failures"]) == 8
    matrix = load(HERE / "fact_coverage_pairs116_v1.matrix.json")
    pair = matrix["rows"][58:60]
    assert len(pair) == 2 and pair[0]["question"] == "林地部族决定迁徙的原因是什么？"
    evidence = pair[0]["evidence_brief"]
    assert all(r["evidence_brief"] == evidence and sha_text(evidence) == r["evidence_sha256"]
               for r in pair)
    markers = [
        "来自北方的暴风雪不断吞没林地",
        "最终，这个部族分为两支",
        "大部分人最终都跟随雪祀离开了老树",
        "她想过一切可能性，却没有料到乌萨斯人会突然到来",
    ]
    assert all(marker in evidence for marker in markers)
    e6 = evidence[evidence.index("[E6]"):]
    assert [e6.index(marker) for marker in markers] == sorted(e6.index(marker) for marker in markers)
    reviewed = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent_assistant_not_human_gold",
        "evaluation_only": True, "training_ready": False,
        "supplements_without_overwrite": str(HERE / "partial_source_review_v1.json"),
        "previous_review_sha256": sha(HERE / "partial_source_review_v1.json"),
        "matrix_sha256": sha(HERE / "fact_coverage_pairs116_v1.matrix.json"),
        "reviewed_pairs": [{
            "pair_index": 29, "question": pair[0]["question"],
            "parent_record_id": pair[0]["parent_record_id"],
            "evidence_sha256": pair[0]["evidence_sha256"],
            "source_review_scope": "All visible E1-E6 read, including full E6 archive text.",
            "findings": [
                "E6 orders snow danger, tribe split/migration, people staying to guard the tree, then Ursus arrival and deaths.",
                "Candidate explicitly calls the later Ursus invasion/deaths another cause of the earlier split/migration; its own quoted source contradicts that chronology.",
                "E3 and E4 independently preserve the same earlier migration/later killings order within the visible material.",
                "E2 also contains a woodland shaman's migration and divinatory signs; do not assume this question unambiguously names the Wanghan tribe or certify a complete gold answer.",
                "A verbatim E6 quote about deaths does not entail the candidate's added migration causality.",
                "Original variant was rejected to retrieve_more for a missing quote; the wrong candidate claim is a newly displayed semantic failure, not just raw generation.",
            ],
        }],
        "old_labels_overwritten": False, "sent_to_generation_or_auditors": False,
        "complete58_source_review": False, "certified_accuracy": None,
    }
    save(review, reviewed)
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": False, "training_ready": False, "report": str(report),
        "report_sha256": sha(report), "previous_checkpoint_preserved": True,
        "automatic_summary": post["summary"], "unique_audits": 126,
        "unique_statuses": dict(Counter(r["status"] for r in audited["results"])),
        "failure_classification": {"validated_protocol_failures": 8, "completion_length_empty_body": 3},
        "length_failure_details": [{
            "id": r["id"], "response_model": r["response_model"],
            "finish_reason": r["finish_reason"], "usage": r["usage"],
        } for r in limited],
        "runtime_source_sha256": snapshot(SOURCE),
        "artifacts_sha256": {str(p): sha(p) for p in [
            HERE / "checkpoint_v17_state.json", HERE / "blind_audits_v1.json",
            HERE / "blind_audits_postflight_v1.json", review,
        ]},
        "prompt_promoted": False, "no_certified_answer_accuracy_or_retrieval_recall": True,
    })
    print(json.dumps({"report": str(report), "audit_requests": 126,
                      "length_empty_body": 3, "protocol_errors": 8, "goal_complete": False}))


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
