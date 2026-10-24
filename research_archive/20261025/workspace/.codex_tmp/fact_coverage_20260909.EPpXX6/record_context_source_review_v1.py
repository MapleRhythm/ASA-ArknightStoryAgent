"""Selected independently reviewed gains/failures; no labels or prompts changed."""
from datetime import datetime, timezone
import json

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save

NOTES = {
    4: (
        "Same hash as previously full-source-reviewed planetary-fragment evidence.",
        [
            "Shown text exactly equals original4B: still effect/track/gravity, without charging/deployment-direction mechanism.",
            "Removing generator reasoning metadata did not fix this known mechanism omission.",
        ],
    ),
    15: (
        "All visible E1-E12 read in this continuation, and both original/candidate shown answers.",
        [
            "E1/E8 support the banshee searching for Nasti, with the mother/daughter relationship; E4/E5/E6/E8 identify Fualgra.",
            "Original answer incorrectly says the sought person is Fualgra, the searcher herself.",
            "Candidate corrects the subject/object distinction: Fualgra is seeking Nasti. This is a supported per-example delivered-answer improvement.",
            "Mother/daughter here must not be silently expanded to a biological-parent claim.",
        ],
    ),
    17: (
        "All visible E1-E6 read together with full raw/parsed/grounded trace.",
        [
            "Both supported_facts in raw output merely repeat the whole user question; parser deduplication keeps one.",
            "The final-answer draft contains a candy-shop connection, but final-answer grounding warnings trigger fact reconstruction into the unanswered question.",
            "The defect is not output truncation: finish=stop and generated243 tokens.",
            "Do not restore the unvalidated draft and certify its interpretation; reject the non-factual question echo or rebuild only from valid facts.",
            "The applied guard changes this one displayed question to retrieve_more in the saved-state replay; this is removed false acceptance, not an extra correct answer.",
        ],
    ),
    33: (
        "All visible E1-E12 read during native8B review; exact same source hash verified.",
        [
            "Candidate explicitly names Oripathy, whereas original shown answer only says another familiar crisis.",
            "E3/E6/E9 support the suspected Oripathy link; E9 is Yato's own cave/disease line, preserving possible rather than certain causality.",
            "Candidate's second fact retains possibility for a shared Originium influence supported by E11's question.",
            "A real selected-example coverage gain; no claim that this58-wide prompt is safe or improves population accuracy.",
        ],
    ),
    39: (
        "All visible E1-E6 read during native8B review; exact same source hash verified.",
        [
            "Candidate's self-criticism about stubbornness, money/life overextension and failing family is supported in E1/E2.",
            "It removes original4B's unsupported insertion of 'ill-gotten money'; source says 'money', not that it was ill-gotten.",
            "Unlike8B's first fact, candidate does not convert the bank manager's accusation into Leon's own view.",
            "This is another supported selected-example improvement, not a certified overall rate.",
        ],
    ),
    56: (
        "Both shown answers and candidate's attached E10 quotes read; not full-source-certified for this question.",
        [
            "Original abstained; candidate answers about Mikkel's comrades, not Susie, and adds telling Mikkel the truth.",
            "The target mismatch is visible in the question-versus-delivered-answer comparison itself.",
            "Do not claim the rescue/telling events are globally false without further full-source review.",
            "Even if another rescue were supported, it does not answer how Susie was rescued.",
        ],
    ),
}


def main():
    destination = HERE / "without_reasoning_hints58_partial_source_review_v1.json"
    assert not destination.exists()
    matrixpath = HERE / "without_reasoning_hints58_v1.matrix.json"
    analysispath = HERE / "without_reasoning_hints58_validation_v1.json"
    rows, results = load(matrixpath)["rows"], load(analysispath)["results"]
    native = load(HERE / "native8b_all9_shown_source_review_v1.json")
    native_by_index = {r["index"]: r for r in native["reviewed"]}
    old = {r["id"]: r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]}
    reviewed = []
    for index, (scope, findings) in NOTES.items():
        row, result = rows[index], results[index]
        assert row["id"] == result["id"] and row["evidence_brief"] == result["evidence_brief"]
        assert digest(row["evidence_brief"]) == row["evidence_sha256"]
        if index in (33, 39):
            assert row["evidence_sha256"] == native_by_index[index]["evidence_sha256"]
        original = old[row["parent_record_id"] + "::original_prompt_v1"]
        if index == 4:
            assert original["grounded"]["answer"] == result["grounded"]["answer"]
        reviewed.append({
            "index": index, "question": row["question"], "id": row["id"],
            "source_review_scope": scope, "evidence_sha256": row["evidence_sha256"],
            "original_shown_sha256": digest(original["grounded"]["answer"]),
            "candidate_shown_sha256": digest(result["grounded"]["answer"]), "findings": findings,
        })
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "reviewer": "independent_assistant_not_human_gold",
        "matrix_sha256": sha(matrixpath), "analysis_sha256": sha(analysispath), "reviewed": reviewed,
        "all58_shown_pairs_read": True, "all58_sources_read": False,
        "old_labels_overwritten": False, "sent_to_generation_or_auditors": False,
        "certified_accuracy": None,
    })
    print(json.dumps({"selected_reviews": len(reviewed), "all58_shown_pairs_read": True,
                      "all58_sources_read": False}), flush=True)


if __name__ == "__main__":
    main()
