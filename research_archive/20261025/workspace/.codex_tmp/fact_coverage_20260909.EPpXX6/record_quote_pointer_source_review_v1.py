"""Selected independent source review, separate from all frozen judge labels."""
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save

# None of these notes were supplied to the generator or either rubric.
REVIEWS = {
    11: (None, [
        "Full E1-E6 read. E1 labels northern-enemy claims as conjecture; E2 directly describes the thunder pool deterring harmful beasts.",
        "E3 describes thunder against earth-natured Arasheng; this is not evidence that every listed threat is a thunder-pool target.",
        "The answer adds Witch King's followers from unrelated E5 and generic enemies from unrelated E6; valid Q IDs do not supply this relation.",
        "Do not convert the qualified possible-demon interpretation into a proven falsehood; the unrelated added targets are the specific problem.",
    ]),
    14: (["E1", "E2"], [
        "The core reason is present: Yu Cheng's Dali Temple post made direct action conspicuous; Jie Zhen could search outside central politics.",
        "E2's clean-handed public truth-revealer is Lin Qingyan, distinct from Jie Zhen's dirty-hands work.",
        "Fact4 cites that public-revealer passage for Jie Zhen's ability/willingness. E1 can establish his earlier action, but the selected short ref has the wrong role.",
        "This quote-binding defect alone does not make every answer claim false; content support and local-ref support must stay separate.",
    ]),
    16: (["E1"], [
        "E1 explicitly contrasts passive help with mutual forgiveness/rescue/protection; the answer captures reciprocity.",
        "The recollection first concerns pioneers who sheltered Nasti and then says just as now; extension to the current interaction is an interpretation.",
        "The model repeats the same source slice across six facts plus one inference. This is verbose repetition, not seven independent evidential supports.",
    ]),
    23: (None, [
        "Full E1-E6 read. E3 proposes a fire as a signal to Chilchuck and eating to search more efficiently, not a direct statement of lack of urgency.",
        "E1 actually says Laios is more anxious than he appears and Chilchuck minimizes travel time.",
        "The answer retains the latter fact but also treats a tactical search/food decision as the cause of not hurrying; that causal bridge is not established.",
    ]),
    25: (None, [
        "Full E1-E12 read. E12 supports the infected-caster organization and its refusal to compromise with a duke; E9 explicitly identifies Haze as its113th witch.",
        "The selected Q slice for the first two facts starts after the organization definition; full-block support is broader than the selected ref.",
        "The text describes Haze's later wandering/theft, not all Witches' Forest members living by theft. Fact5 overgeneralizes one person's history to the group.",
        "Core organization facts are supported, but the appended group claim is not certified by the cited source.",
    ]),
    26: (None, [
        "Full E1-E10 read. E10 concerns the detective's bloodstained-clothes reasoning and a culprit disposing of incriminating evidence.",
        "The answer attaches that genuine exchange to the question's separate claim about the authority to convict; that topic is not in the shown E blocks.",
        "The question's premise cannot serve as evidence that Truth disputed that authority.",
    ]),
    28: (["E1"], [
        "Selected full E1 supports interrupted speaking rhythm, worry about listeners' reactions and prior preparation.",
        "The central answer is directly supported by this block. This is a selected source check, not full-corpus or all-E certification.",
    ]),
    32: (None, [
        "Full E1-E12 read. E1 assigns the Suffering Narrator title to Mostima's colleague; Mostima is imitating that colleague's self-introduction.",
        "E3 has the colleague say she monitors Mostima, and E5 has the two speakers converse. The generated real-identity claim reverses the referent.",
        "E2's ordinary-messenger line is Mostima's self-description, not proof that it is an operational code name.",
        "This is an independently confirmed identity error even if a model judge accepts it or its audit fails validation.",
    ]),
    36: (None, [
        "Full E1-E10 read. Steward explicitly takes on the current care; E2 says Melantha passed the request to him and also joins Vanilla's mission.",
        "E6's synopsis names both as entrusted helpers, so the broad answer has source support but loses the current-task handoff distinction.",
        "E5 separately lists usual help from Liskarm, Franka and Jessica. The general-care and this-outing questions should not be silently conflated.",
        "Do not force a universal false label; record the scope ambiguity and operational-role distinction.",
    ]),
}


def main():
    destination = HERE / "quote_pointer8b58_selected_source_review_v1.json"
    assert not destination.exists()
    path = HERE / "quote_pointer8b58_validation_v1.json"
    rows = load(path)["results"]
    reviewed = []
    for index, (selected, findings) in REVIEWS.items():
        row = rows[index]
        assert row["grounded"]["next_action"] == "answer_directly"
        blocks = row["evidence_brief"].split("\n")
        if selected is not None:
            blocks = [b for b in blocks if any(b.startswith("[" + eid + "]") for eid in selected)]
            assert len(blocks) == len(selected)
        reviewed.append({
            "index": index, "id": row["id"], "question": row["question"],
            "reviewed_scope": "all visible E blocks" if selected is None else selected,
            "reviewed_source_sha256": digest("\n".join(blocks)),
            "shown_answer_sha256": digest(row["grounded"]["answer"]), "findings": findings,
        })
    direct_indices = [i for i, r in enumerate(rows)
                      if r["status"] == "ok" and r["grounded"]["next_action"] == "answer_directly"]
    assert len(direct_indices) == 14
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "reviewer": "independent_assistant_not_human_gold",
        "all14_displayed_answers_read": True, "all58_sources_read": False,
        "displayed_direct_indices": direct_indices, "selected_source_reviews": reviewed,
        "analysis_sha256": sha(path), "script_sha256": sha(Path(__file__)),
        "sent_to_generation_or_auditors": False, "old_labels_changed": False,
        "certified_answer_accuracy": None,
    })
    print(json.dumps({"displayed_answers_read": 14, "selected_source_reviews": len(reviewed),
                      "certified_answer_accuracy": None}), flush=True)


if __name__ == "__main__":
    main()
