"""Four further complete-source reviews, independent of generator and judge."""
from datetime import datetime, timezone
import json

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save


REVIEWS = {
    12: {
        "scope": "All visible E1-E3 and both shown answers read.",
        "required_markers": [
            "只有这样的目标能让我们马上搞到一大笔钱",
            "你从博物馆高处跳下来......是因为你想名扬萨尔贡",
            "娜仁图亚：嗯？算是吧",
            "娜仁图亚：欠了一笔债，挺头疼的",
        ],
        "findings": [
            "Both answers' core financial goal is directly supported by the kidnapping discussion in E1/E2.",
            "Debt is also mentioned, but in another exchange; a debt-causes-this-kidnapping link is contextual inference, not an explicit causal statement in these sources.",
            "The per-fact quote '嗯？算是吧' responds to jumping/fame, not an acknowledgment that debt caused kidnapping.",
            "Thus the claimed debt bridge should not be treated as independently established just because those short quotes match.",
            "Both variants share the same causal bridge; the judge's complete-versus-unsupported difference is not reliable evidence of a candidate-specific causal regression.",
            "Candidate drops '计划' in the first shown fact; source describes failed plans, so do not infer a completed successful kidnapping from that phrasing.",
        ],
    },
    13: {
        "scope": "All visible E1-E6 and both shown answers read.",
        "required_markers": [
            "直到生命终结，你们都无法抹除身上沾染的其他狼的味道",
            "它受到了“头狼的赐福”",
            "狼之主不死不灭，狼之主永恒存在",
            "这份气味会始终提醒你她的存在",
        ],
        "findings": [
            "E1 explicitly gives the wolves' lifelong scent rule and wolf lords' immortality; E2/E6 give the cut/blood interaction and E1 the dagger blessing.",
            "Both shown answers omit the actual scent rule/immortality explanation and turn the persistent reminder of Red into the cause of scent permanence.",
            "Remembering Red is presented in E3 as a consequence of the persistent scent, not the mechanism making scent irremovable.",
            "Original additionally asserts Lappland exchanged their scents through a dream; the visible event is Red's blade cutting Caesar and his blood covering Red, with Lappland orchestrating the encounter.",
            "Original judge's supported_partial label does not make its reversed causal explanation correct.",
        ],
    },
    19: {
        "scope": "All visible E1-E12 and both shown answers read.",
        "required_markers": [
            "哦，其实我一点也不喜欢这里",
            "我可以走神，只需要在结尾的时候，握着他们的手表示支持",
            "这有助于保持您的政治形象，对接下来的竞选有帮助",
            "我今天像是一个第一次上科学课的孩子",
        ],
        "findings": [
            "E9 explicitly contrasts public praise of Trimounts with Jackson's private 'I do not like it at all' and his missing frontier spirit.",
            "E7 describes scientists as personally simple, his limited understanding/wandering attention, easy ceremonial support and political-image benefit.",
            "Candidate converts scientists being simple into their research being pure, and presents public praise/interest as a sufficient description of his real attitude.",
            "E12 also includes public praise and concrete help for a Rhine base application; do not invent the opposite extreme that every supportive action is false.",
            "Nevertheless the shown answer fails the question's public-versus-private distinction and drops the explicit private dislike.",
            "Original was rejected to abstain; the candidate's mechanical acceptance newly exposes this incomplete/misleading attitude answer.",
        ],
    },
    27: {
        "scope": "All visible E1-E6 read; E1-E2 reread separately after a combined terminal view was truncated. Both shown answers read.",
        "required_markers": [
            "我倒要看看这群人想要干什么",
            "天火：放开手，我自己会走",
            "明明都要杀人灭口了",
            "为什么伪装成感染者",
        ],
        "findings": [
            "E5 describes Skyfire deliberately accompanying the captors to investigate, including a twenty-minute journey to the factory.",
            "E1/E2/E3 put her fighting and bomb destruction at the factory against captors threatening to kill the hostages; she also demands answers.",
            "E3 explicitly reveals the apparent infected captors' blackened glass/fake crystals; the shown candidate repeats their apparent identity as established.",
            "Candidate says she intervened to keep people from being captured and taken to the factory, contradicting the already-completed capture/transport.",
            "Its own quote 'I will walk myself' does not support preventing capture or transport.",
            "Original raw output mentions hostage killing but is quote-rejected to retrieve_more; do not count unshown raw content as a correct delivered answer.",
        ],
    },
}


def main():
    destination = HERE / "partial_source_review_v3.json"
    assert not destination.exists()
    matrix_path = HERE / "fact_coverage_pairs116_v1.matrix.json"
    matrix = load(matrix_path)
    analysis = {r["id"]: r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]}
    reviewed = []
    for index, review in REVIEWS.items():
        pair = matrix["rows"][2 * index:2 * index + 2]
        evidence = pair[0]["evidence_brief"]
        assert all(r["evidence_brief"] == evidence and r["evidence_sha256"] == digest(evidence) for r in pair)
        assert all(s in evidence for s in review["required_markers"])
        reviewed.append({
            "pair_index": index, "question": pair[0]["question"],
            "parent_record_id": pair[0]["parent_record_id"], "evidence_sha256": digest(evidence),
            "source_review_scope": review["scope"], "findings": review["findings"],
            "shown_answer_sha256": {r["renderer"]: digest(analysis[r["id"]].get("grounded", {}).get("answer", ""))
                                   for r in pair},
        })
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "reviewer": "independent_assistant_not_human_gold",
        "matrix_sha256": sha(matrix_path), "reviewed_pairs": reviewed,
        "old_labels_overwritten": False, "sent_to_generation_or_auditors": False,
        "earlier_notes_preserved_sha256": {
            name: sha(HERE / name) for name in ("partial_source_review_v1.json", "partial_source_review_v2.json")
        },
        "all58_source_review_complete": False, "certified_accuracy": None,
    })
    print(json.dumps({"new_complete_source_reviews": len(reviewed), "old_labels_overwritten": False}))


if __name__ == "__main__":
    main()
