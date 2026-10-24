"""Independent reviews of all9 delivered8B answers, plus nonexclusive rejection counts."""
from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save

NOTES = {
    1: [
        "Same complete E1-E9 source hash as the prior disease review; both this new shown answer and its refs read.",
        "Again converts 'is the disease related to monsters?' into an established relationship, supplies no disease name and asserts an unestablished disease/history/evil-contamination connection.",
        "This is not corrected by8B, and the candidate is actually displayed.",
    ],
    2: [
        "All visible E1-E6 read. The order from the Duke of Caster is a supported sufficient direct reason for the officers' prohibition.",
        "The answer preserves that core, but appends/repeats morale and internal-divisions interpretations.",
        "E1 shows the officer does not want to become the next front-line target; it does not explicitly establish the claimed command motive of morale preservation.",
        "These plausible interpretations must be distinguished from the quoted direct order; not a certified wholly supported causal answer.",
    ],
    26: [
        "All visible E1-E10 read. E10's 'indeed' and 'questions' refer to holes in the detective story, particularly the culprit disposing of bloodied clothing.",
        "The visible source does not contain the question's asserted discussion of a detective's authority to convict.",
        "Candidate invents Istina's belief that detectives sometimes have conviction-like powers, instead of acknowledging missing support for the premise.",
    ],
    33: [
        "All visible E1-E12 read. The shown answer now actually includes the disease name Oripathy in the cave-link conditional; the old shown answer used only a generic crisis.",
        "This is useful coverage gain, but it leaves the central answer indirect and bloats it with repeated/unasked ecological statements.",
        "E2's safe-distance/ecological argument is the Palico's position; E3 explicitly gives Yato's contrasting nearby threat argument.",
        "The unqualified added safe-distance claim therefore must not be represented as Yato's settled attitude.",
        "The raw 'ecological crisis' conclusion was not shown; do not count removal of an unsupported extra as harm.",
    ],
    36: [
        "All visible E1-E10 read. The delivered answer is only 'Steward', directly supported by E1/E2 for this particular mission's arrangement.",
        "E5 also says Liskarm/Franka and sometimes Jessica routinely help; if the question is about every recurring caretaker rather than this scene, a single name is not exhaustive.",
        "The undisplayed fact 'Steward said Big Black was gentle' misattributes Vanilla's line; do not silently count that fact as part of the one-word delivered answer.",
        "The one-word answer is supported for the scene, with scope ambiguity retained instead of inventing a uniquely exhaustive gold.",
    ],
    38: [
        "All visible E1-E6 read. E1 directly supports Jaime's view about Oripathy, death's danger, stigma, exclusion and lack of a place to survive.",
        "The shown answer answers the requested viewpoint without importing the other speakers in E2-E6.",
        "However the per-fact refs on fact2 do not explicitly cover its added danger statement; that support appears elsewhere in E1 and fact1's refs.",
        "Content support across the visible source is not the same as complete local per-fact citation support.",
    ],
    39: [
        "All visible E1-E6 read. E1/E2 support Leon's self-criticism for stubbornness, overextending resources and failing his sons.",
        "The first shown fact instead assigns the bank manager's E4 accusation ('greedily keeping things not belonging to him') to Leon's own view.",
        "E4 even disambiguates Helena's 'she is right' as the safe password, not agreement with that accusation.",
        "Other supported self-criticism does not cure this wrong-speaker fact.",
    ],
    43: [
        "All visible E1-E6 read. E1 directly supports the elder-brother identity and the sword/disposition/Huai Tianpei context.",
        "The core identity answer and added discussion are supported, though the latter is unnecessary for the narrow question.",
    ],
    53: [
        "All visible E1-E6 read. E2 supplies the stated pursuit/independent-action/losses reason; E5 supplies Alberto's authority-boundary explanation.",
        "Candidate's first two facts retain these supported reasons.",
        "Its last fact recasts the old expulsion behavior as being for family interests, supported only by a quote about returning for lunch/news in a later scene.",
        "E1's theatrical claim about current loyalty does not establish the motive for the earlier conduct that caused expulsion; this is an unsupported appended causal/time claim.",
    ],
}


def main():
    destination = HERE / "native8b_all9_shown_source_review_v1.json"
    taxonomy_path = HERE / "native8b_rejection_taxonomy_v1.json"
    assert not destination.exists() and not taxonomy_path.exists()
    matrix_path = HERE / "qwen3_8b_native58_v1.matrix.json"
    analysis_path = HERE / "qwen3_8b_validation_v1.json"
    rows, results = load(matrix_path)["rows"], load(analysis_path)["results"]
    direct_indices = [i for i, r in enumerate(results) if r["grounded"]["next_action"] == "answer_directly"]
    assert direct_indices == list(NOTES)
    prior = load(HERE / "partial_source_review_v1.json")
    disease = next(r for r in prior["reviewed_pairs"] if r["pair_index"] == 1)
    assert disease["evidence_sha256"] == rows[1]["evidence_sha256"]
    reviewed = []
    for index, notes in NOTES.items():
        row, result = rows[index], results[index]
        assert row["id"] == result["id"] and row["evidence_brief"] == result["evidence_brief"]
        assert digest(row["evidence_brief"]) == row["evidence_sha256"]
        reviewed.append({
            "index": index, "id": row["id"], "question": row["question"],
            "evidence_sha256": row["evidence_sha256"],
            "shown_answer_sha256": digest(result["grounded"]["answer"]),
            "scope": "same previously fully read source" if index == 1 else "full visible source read in this continuation",
            "findings": notes,
        })
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "reviewer": "independent_assistant_not_human_gold",
        "matrix_sha256": sha(matrix_path), "analysis_sha256": sha(analysis_path),
        "all9_delivered_answers_reviewed": True, "reviewed": reviewed,
        "not_reviewed_full_sources_for_all49_nondirect": True,
        "no_gold_or_old_labels_overwritten": True, "sent_to_generators_or_judges": False,
        "certified_accuracy": None,
    })
    patterns = Counter()
    failures = []
    for i, result in enumerate(results):
        if result["grounded"]["next_action"] == "answer_directly":
            continue
        issues = result["grounded"].get("generation_diagnostics", {}).get("grounding_issues", [])
        names = sorted({re.sub(r"\d+", "N", value.split(":")[0]) for value in issues})
        patterns.update(names)
        failures.append({"index": i, "id": result["id"], "action": result["grounded"]["next_action"],
                         "issue_categories": names})
    save(taxonomy_path, {
        "analysis_sha256": sha(analysis_path), "raw_direct": 58, "delivered_direct": 9,
        "nondirect": 49, "nonexclusive_row_issue_counts": dict(patterns),
        "nonexclusive_counts_do_not_sum_to49": True, "rows": failures,
        "no_counterfactual_relaxation_or_deployment_change": True,
    })
    print(json.dumps({"all9_shown_source_reviewed": True, "nondirect": 49,
                      "nonexclusive_issues": dict(patterns)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
