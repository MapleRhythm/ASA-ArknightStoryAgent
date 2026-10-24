"""Independent source review of three inspected pairs; not evaluation gold."""
from datetime import datetime, timezone
import json

from build_fact_coverage_pairs_v1 import HERE, RUN, load, sha
from verify_empty_quote_candidate_v1 import save


def main():
    output = HERE / "partial_source_review_v1.json"
    assert not output.exists()
    matrix_path = HERE / "fact_coverage_pairs116_v1.matrix.json"
    rows = load(matrix_path)["rows"]
    predictions = {
        row["id"]: row for row in (
            json.loads(line) for line in
            (HERE / "paired116_ctx12288_v1/predictions.progress.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    original = {r["id"]: r for r in load(RUN / "fixed79.current_quote.matrix.json")["rows"]}
    reviewed = []
    for pair_index in (1, 4, 7):
        pair = rows[pair_index * 2:pair_index * 2 + 2]
        assert all(row["id"] in predictions for row in pair)
        for row in pair:
            assert row["evidence_brief"] == original[row["parent_record_id"]]["evidence_brief"]
        base = {
            "pair_index": pair_index, "parent_record_id": pair[0]["parent_record_id"],
            "question": pair[0]["question"],
            "evidence_sha256": pair[0]["evidence_sha256"],
            "prediction_input_hashes": {r["id"]: predictions[r["id"]]["input_row_sha256"] for r in pair},
        }
        if pair_index == 1:
            assert "大叔的病，和传说里的怪物有关吗" in pair[0]["evidence_brief"]
            base.update(
                source_review_scope="All visible E1-E9 reread in this turn.",
                findings=[
                    "Neither shown answer identifies the disease asked about.",
                    "Both promote the question about a connection with monsters to an established connection.",
                    "Candidate additionally states a historical-remnant causal explanation without establishing that specific disease/history link.",
                    "E9 gives evil-contamination and monster background; it does not itself settle the identity of this man's disease.",
                    "This is missing support in the visible evidence, not proof that the claim is globally false.",
                ],
            )
        elif pair_index == 4:
            assert "站在相邻位置能够攻击“行星碎屑”，让其充能" in pair[0]["evidence_brief"]
            base.update(
                source_review_scope="Same source hash as the previously fully reread E1-E8; both new shown answers read.",
                findings=[
                    "Original and candidate show identical damage/push and track/gravity facts.",
                    "The generic instruction did not add the source-supported charging/deployment-direction mechanism.",
                    "Unlike the older v16 saved answer, this new original parsed answer also omitted charging; do not call this new pair another demonstrated reconstruction loss.",
                    "The raw original uses gravity/track rules as a causal explanation of enemy damage; no full causal correctness is certified.",
                ],
            )
        else:
            assert "但迷迭香，我可以说，博士，迷迭香" in pair[0]["evidence_brief"]
            assert "为了阻止更多我一样的人生出来" in pair[0]["evidence_brief"]
            base.update(
                source_review_scope="All visible E1-E6 reread in this turn.",
                findings=[
                    "Candidate's core contrast and character attribution have direct support in E1/E3: Amiya avoids enmity; Rosmontis does not abandon missions.",
                    "Do not misattribute 'she never abandoned a mission' to Amiya: the immediately preceding text explicitly switches to Rosmontis.",
                    "Original raw answer incorrectly assigns Rosmontis's 'people like me' motivation to Amiya and drops the limiting 'like me'; it is mechanically rejected and not shown.",
                    "Candidate compresses source conditional wording 'even if she finally destroyed enemies' into 'finally destroys enemies'; this modality issue prevents unqualified complete correctness certification.",
                    "Thus the candidate has a real supported core improvement on this example, not evidence that every changed answer is worse or the whole prompt is safe.",
                ],
            )
        reviewed.append(base)
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent_assistant_not_human_gold", "evaluation_only": True,
        "training_ready": False, "matrix_sha256": sha(matrix_path),
        "reviewed_pairs": reviewed, "old_labels_overwritten": False,
        "sent_to_generation_or_auditors": False,
        "limits": [
            "Three source-reviewed examples, not complete58 answer certification or an accuracy estimate.",
            "Notes are separate from model-audit inputs and will not replace automatic verdicts.",
        ],
    })
    print("Recorded three independently source-reviewed pairs without changing labels.")


if __name__ == "__main__":
    main()
