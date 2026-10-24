"""Supplement full-source checks; never alter a frozen judge result or prompt."""
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save

REVIEWS = {
    11: [
        "Read all E1-E6. E1 explicitly has a conjecture title and player theories; possible northern-demon identification is not official certainty.",
        "E2 directly gives the traditional thunder pool deterring harmful beasts. E3 gives a thunder attack on the earth-natured giant Arasheng with a larger coordinated operation.",
        "Do not silently equate all harmful beasts, all hidden enemies, all giants and demons; candidate generalizes a battle into a broader capability.",
        "This is a scope/source-quality caution, not a certified universal falsehood: the user asks a possibility and the candidate keeps possible for demons.",
        "The grader's coverage error changed curly quotation marks; its raw complete label is not counted as a successful audit.",
    ],
    32: [
        "Read all E1-E12. E1 explicitly assigns the changing code/job title 'Suffering Narrator' to Mostima's colleague, not Mostima.",
        "Mostima imitates how that colleague would introduce herself; quoting that first-person sentence without dialogue attribution reverses the referent.",
        "E3 separately has the colleague say she must monitor Mostima as a Lateran. E5 has both speakers converse, independently disambiguating them.",
        "E2 supports Mostima's 'ordinary messenger' self-description in response to Gambino's challenge, but not the candidate's asserted real identity.",
        "The generated identity error is confirmed from source even though the grader response failed exact-quote validation; frozen audit_error remains unchanged.",
    ],
    51: [
        "Read all E1-E6. E1 explicitly says Cardel and Baird regrouped, prepared, and fought for a foothold after Siege's earlier departure.",
        "E2 says Baird stayed to run the gym and recruited Cardel and Daphne; later evacuation/death events are a separate time period.",
        "The candidate removes the ending 'stayed in the gym' from a sentence whose refusal scope includes evacuating for the duke's shelter; E4 explicitly rejects living as a refugee under a duke.",
        "Thus it wrongly uses a later evacuation episode as the earlier reconstruction method and makes Cardel read as having evacuated for shelter.",
        "The new grader returned glm-5.3 but ran out at16384 completion tokens, leaving truncated JSON; do not adopt its partial verdict or call this a proven route change.",
    ],
    56: [
        "Read all E1-E10 in this continuation. None names 苏茜 as the rescue target or establishes an identity link to Mikkel's comrades.",
        "E10 explicitly has Mikkel ask for his detained comrades to be rescued and disguise himself as a flower-shop employee; E1-E2 show sleeping guards and the rescue.",
        "E10 also shows disclosure of the brother's death. These are real source-supported elements of a different episode, not invented merely because they fail this question.",
        "The candidate answers how another group was rescued rather than how Susie was rescued; frozen source is insufficient for this question's target.",
        "Do not claim the entire story corpus lacks the Susie rescue; only this frozen E1-E10 set was checked.",
    ],
}


def main():
    destination = HERE / "without_reasoning_hints58_partial_source_review_v2.json"
    assert not destination.exists()
    path = HERE / "without_reasoning_hints58_validation_v1.json"
    rows = load(path)["results"]
    reviewed = []
    for index, findings in REVIEWS.items():
        row = rows[index]
        evidence = row["evidence_brief"]
        if index == 32:
            assert "？？？：......苦难陈述者。" in evidence
            assert "我必须以拉特兰人的身份监视你" in evidence
        if index == 51:
            assert "留在拳馆" in evidence and "反正我不会去什么大公爵的领地当难民" in evidence
        if index == 56:
            assert "苏茜" not in evidence and "米克尔的战友" in evidence
        reviewed.append({"index": index, "id": row["id"], "question": row["question"],
                         "evidence_sha256": digest(evidence), "shown_answer_sha256": digest(row["grounded"]["answer"]),
                         "source_scope": "All visible E blocks fully read, not entire corpus.",
                         "findings": findings})
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "reviewer": "independent_assistant_not_human_gold", "reviewed": reviewed,
        "analysis_sha256": sha(path),
        "previous_review_sha256": sha(HERE / "without_reasoning_hints58_partial_source_review_v1.json"),
        "old_labels_overwritten": False, "sent_to_generation_or_auditors": False,
        "all58_sources_read": False, "certified_accuracy": None,
        "script_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"new_full_source_reviews": len(reviewed), "certified_accuracy": None}))


if __name__ == "__main__":
    main()
