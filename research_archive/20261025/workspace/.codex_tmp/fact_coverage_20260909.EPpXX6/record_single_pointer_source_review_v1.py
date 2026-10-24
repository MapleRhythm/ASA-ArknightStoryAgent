"""Independent source/fact-scope review, not supplied as labels to any model."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save

FINDINGS = [
    [
        "The source pointers mechanically resolve correctly, but the raw answer mixes Quercus/Susie with Mikkel's detained comrades and ventilation-pipe rescue.",
        "Q80 is the vine assault; Q93/Q94 are the Mikkel episode. Q99 shows arrival behind Susie, not a ventilation-pipe escape.",
        "Grounding rejects on a lexical warning about 制造, not a proven semantic event-mismatch detector; do not overstate the protection.",
    ],
    [
        "The raw answer again targets Mikkel's comrades instead of Susie.",
        "Q143/Q144 only show the ventilation-pipe/flower/sleeping-guard rescue dialogue; they do not support the added vine assault or disguise clause in the bound fact.",
        "The final answer is rejected by the unchanged token gate; no correct answer is delivered.",
    ],
    [
        "The actual raw JSON chooses retrieve_more, not abstain.",
        "The canonical parser turns that into abstain because round2/2 is exhausted. The legacy summary's raw_actions is therefore actually a parsed-action count for this row.",
        "This is an observed avoidance of the wrong-event direct answer on this state, not a newly correct displayed answer.",
    ],
    [
        "The raw/displayed answer now identifies the Susie event, including vines, stopping the attack and comforting her; this is not another Mikkel rescue.",
        "Fact1 binds only Q84/Q85 in E2. They describe germinating seeds and vines knocking down attackers, but not Susie's later attack on the councillor or Quercus stopping her.",
        "The explicit restraint of downed mercenaries is in Q86, which was not selected. The stopping/embracing event is in E3, not the selected fact1 references.",
        "Fact2 binds Q142/Q143 in E3. They support arriving through wreckage, embracing and calming Susie, but omit the departure clause.",
        "Q144 adds '我们回去吧'; canonical chunk0026 continues reassurance and proposes returning. This surrounding source is not selected evidence for fact2 and does not turn every action into a directly quoted completed event.",
        "Thus mechanical binding succeeds but complete per-fact citation support is not established. Do not certify this as a fully correct grounded answer or auto-create a positive training label.",
    ],
]


def main():
    output = HERE / "single_pointer_handoff4_source_review_v1.json"
    assert not output.exists()
    validationpath = HERE / "single_pointer_handoff4_validation_v1.json"
    matrixpath = HERE / "single_pointer_handoff4_v1.matrix.json"
    storypath = HERE / "susie_rescue_canonical_story_v1.json"
    rows, matrix = load(validationpath)["results"], load(matrixpath)["rows"]
    post = load(HERE / "single_pointer_handoff4_postflight_v1.json")
    assert post["protocol_verified"] and post["states"] == 4
    assert all(sha(Path(p)) == h for p, h in post["artifacts_sha256"].items())
    assert len(rows) == len(matrix) == 4
    reviewed = []
    for index, (row, source) in enumerate(zip(rows, matrix)):
        assert row["id"] == source["id"] and row["status"] == "ok"
        raw = row["pointer_payload"]
        assert raw == json.loads(row["strict_json_with_known_prefill"])
        reviewed.append({
            "index": index, "id": row["id"], "question": row["question"],
            "raw_action": raw["next_action"], "parsed_action": row["parsed"]["next_action"],
            "grounded_action": row["grounded"]["next_action"],
            "raw_output_sha256": digest(row["raw_output"]),
            "shown_answer_sha256": digest(row["grounded"]["answer"]),
            "bound_facts": row["expanded_payload"].get("supported_facts", []),
            "evidence_sha256": digest(row["evidence_brief"]), "findings": FINDINGS[index],
        })
    assert reviewed[2]["raw_action"] == "retrieve_more"
    assert reviewed[2]["parsed_action"] == reviewed[2]["grounded_action"] == "abstain"
    assert reviewed[3]["grounded_action"] == "answer_directly"
    assert all(r["grounded_action"] == "abstain" for r in reviewed[:3])
    catalog = {r["quote_id"]: r for r in matrix[3]["quote_catalog"]}
    assert catalog["Q84"]["evidence_id"] == catalog["Q85"]["evidence_id"] == "E2"
    assert "苏茜" not in catalog["Q84"]["quote"] + catalog["Q85"]["quote"]
    assert "被植物束缚" in catalog["Q86"]["quote"]
    assert "她依然没有放手" in catalog["Q143"]["quote"]
    assert "我们回去吧" in catalog["Q144"]["quote"]
    story = {d["id"]: d for d in load(storypath)["documents"]}
    prefix = "activities/act10mini/level_act10mini_st05#chunk-"
    reviewed_story = [prefix + f"{i:04d}" for i in range(19, 27)]
    assert all(i in story for i in reviewed_story)
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "reviewer": "independent_assistant_not_human_gold", "reviewed": reviewed,
        "actual_action_counts": {
            k: dict(Counter(r[k] for r in reviewed)) for k in ("raw_action", "parsed_action", "grounded_action")},
        "legacy_summary_raw_actions_are_parsed_actions": True,
        "reviewed_story_chunk_ids": reviewed_story,
        "reviewed_story_text_sha256": {i: digest(story[i]["clean_text"]) for i in reviewed_story},
        "detailed_catalog_context": [catalog[f"Q{i}"] for i in [83, 84, 85, 86, 87, 138, 139, 140, 141, 142, 143, 144]],
        "artifacts_sha256": {str(p): sha(p) for p in (
            validationpath, matrixpath, storypath, HERE / "single_pointer_handoff4_postflight_v1.json")},
        "source_scope": "All four raw/bound/grounded outputs; selected referenced/context passages and canonical story0019-0026, not full58 corpus certification.",
        "old_outputs_or_labels_changed": False, "sent_to_generation_or_auditors": False,
        "new_certified_correct_answers": 0, "new_certified_answer_accuracy": None,
        "script_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"reviewed_outputs": 4, "new_certified_correct_answers": 0,
                      "raw_action_counts": dict(Counter(r["raw_action"] for r in reviewed))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
