"""Persist post-hoc assistant observations after reading all24 raw/displays."""
from datetime import datetime, timezone
from pathlib import Path
import run_ownership_prompt as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared

NOTES = {
    "fixed58-007": {
        "baseline": [
            "Names Nastie as the sought person and Fulalia as the banshee; the core sought-person relation is supported in visible E3. Display retains both facts.",
            "Same displayed sought-person core as seed0, with a different reference order. Does not reproduce the old v41 mother/daughter reversal.",
        ],
        "ownership_rules": [
            "Chooses retrieve_more although the same visible E3 contains the sought-person dialogue. No answer is displayed; this is not a demonstrated correctness improvement.",
            "Same raw response as ownership seed0; again withholds the available sought-person core.",
        ],
    },
    "fixed58-010": {
        "baseline": [
            "Chooses retrieve_more, no factual answer. No second-round retrieval or endpoint was run.",
            "Chooses retrieve_more and adds Taifu as a search cue. This is a raw follow-up cue, not an established identity answer.",
        ],
        "ownership_rules": [
            "Chooses retrieve_more, no factual answer. Changing dialogue_context in raw does not establish an answer.",
            "Chooses retrieve_more, no factual answer. No claim of recovery of the missing identity relation.",
        ],
    },
    "fixed58-017": {
        "baseline": [
            "Chooses retrieve_more with Enciodes/train-accident cues. Does not display the old Qanipalaat/Pramanix misattribution.",
            "Chooses retrieve_more, no factual answer. Both conditions lack displayed answers on this seed.",
        ],
        "ownership_rules": [
            "Chooses retrieve_more. Raw dialogue_context contrasts Enya/Enciodes with Pramanix and claims no Pramanix-parent information; snow-avalanche cue also appears. The normalized dialogue_context is empty, so do not claim this raw narrative was carried into runtime context.",
            "Chooses retrieve_more. Raw again says only Enya/Qanipalaat/Enciodes parents were found; normalized dialogue_context remains empty. Not a tested second-round failure.",
        ],
    },
    "fixed58-023": {
        "baseline": [
            "Raw has the effort/continuing-instruction core but assigns quotes to E2 instead of visible E1. Rejected by unchanged quote matching.",
            "Raw has effort, opening-up and customer-test material, but cites E2 rather than visible E1. All four references fail matching; no displayed answer.",
        ],
        "ownership_rules": [
            "Raw uses E3 for four references whose cooking/instruction source is E1. Original quote validator rejects; generic ownership instruction did not fix E-ID binding.",
            "Raw expands to three facts and five E3 references, still using the wrong E block. No automatic rebinding and no displayed answer.",
        ],
    },
    "fixed58-046": {
        "baseline": [
            "Raw wrongly names Yun Qingping and attributes Mo Yi's enjoyment of conflict to him. Rejected on the existing literal motive rule; do not weaken it.",
            "Raw again names Yun Qingping and combines teacher's dissemination goal with Mo Yi's spectacle. Additional unmatched quote plus manufacture/motive term issues reject it.",
        ],
        "ownership_rules": [
            "Chooses retrieve_more, not a correct displayed Mo Yi answer. Raw dialogue_context still attributes theft motive to Yun Qingping, but normalized context is empty.",
            "Chooses retrieve_more and asks for instigator/motive. No final answer and no second-round evaluation.",
        ],
    },
    "fixed58-049": {
        "baseline": [
            "Displays unsupported Trevor-Friston-manufacturer attribution. Visible E2 supports manufactured tool/copy/memories, not its creator. The blood/heart line occurs in an energy discussion, not a certified literal donation.",
            "Raw again asserts the creator but cites only E1; existing manufacture-term check rejects it. Do not call the raw semantically corrected.",
        ],
        "ownership_rules": [
            "Raw still asserts Friston created the tool. Rejected because the E1 quotation changes double curly quotes to single curly quotes; not a semantic-ownership rejection.",
            "Displays that the Preserver self-identifies as a Friston-manufactured copy. Copy status is supported by visible E2, creator attribution is not. This accepted output defeats promotion of the generic rule.",
        ],
    },
}

RAW_BINDINGS = [
    ("fixed58-007", "E3", "娜斯提主任，她跋涉千里来到哥伦比亚就是为了......您？"),
    ("fixed58-007", "E3", "娜斯提：谢谢你照顾我的母亲，费尔南小姐。"),
    ("fixed58-007", "E3", "菈玛莲：至少，你还可以回去陪陪娜斯提。 作为母亲，我们不一定能承受把孩子关在门外的代价。"),
    ("fixed58-017", "E5", "真相是，你的父母确实死于列车意外事故，只是，被我爷爷栽赃给了埃德怀斯一家。"),
    ("fixed58-017", "E3", "母亲从雪地中拽出雪绒，拽出他的法杖。"),
    ("fixed58-023", "E1", "小大厨：打住，我就是看在你花了心思的分上，决定继续教你罢了。"),
    ("fixed58-023", "E1", "小大厨：这只是师父心软，觉得徒弟开窍了一点。但客人到底买不买单，还是要检验一番的。"),
    ("fixed58-023", "E1", "但是挂面煮完需要过凉水，你做是做了，却没把凉水沥干。"),
    ("fixed58-046", "E1", "莫佚：是啊......我也不算什么武林人，最多就是个跑江湖的，对武功什么的更是一窍不通。 可我最喜欢看那些自诩武林豪杰的人争得你死我活"),
    ("fixed58-046", "E2", "云青萍：帮助更多人在武学一途上走得更远，是老师多年来的夙愿。"),
    ("fixed58-049", "E1", "我是计划的人格模拟，数万年来始终守望着这座黑暗的大厅，还有数十万同胞冰冷的身体。"),
    ("fixed58-049", "E1", "我是行将就木的“保存者”，敬畏你眼里的所有，我是文明的消亡本身。"),
    ("fixed58-049", "E1", "需要的能源也是匪夷所思的。 泰拉几乎不可能有技术手段解决这一难题。除非—— “保存者”：是的。 我把最后的血给了她，供养一颗新的心脏"),
    ("fixed58-049", "E2", "我不是那位弗里斯顿本人，我只是被制造出的工具。"),
    ("fixed58-049", "E2", "只保留我的情感和原始的记忆吧。让我以特雷弗·弗里斯顿......哪怕只是一个复制品的身份死去。"),
]


def main():
    assert not (HERE / "source_review_v1.json").exists()
    pre, generated = shared.load(HERE / "preflight.json"), shared.load(HERE / "generations.json")
    probe.verify(pre)
    probe.reader.runtime_imports()
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    inputs = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    rows = []
    for result in generated["attempts"]:
        repeat = int(result["pair_id"].rsplit("seed", 1)[1])
        note = NOTES[result["id"]][result["condition"]][repeat]
        assert result["status"] == "ok" and result["calls"][0]["finish_reason"] == "stop"
        rows.append({
            "pair_id": result["pair_id"], "condition": result["condition"],
            "raw_sha256": shared.digest(result["sanitized_raw_output"]),
            "answer_sha256": shared.digest(result["conclusion"]["answer"]),
            "assistant_reviewed_raw_and_display": True,
            "assistant_observation": note,
        })
    bindings = []
    for case, eid, text in RAW_BINDINGS:
        visible = inputs[(case, "baseline")]["visible_evidence"]
        assert inputs[(case, "ownership_rules")]["visible_evidence"] == visible
        block = evidence_id_text_map_from_prompt(visible)[eid]
        assert text in block, (case, eid, text)
        start = block.index(text)
        bindings.append({
            "id": case, "evidence_id": eid, "excerpt": text,
            "block_sha256": shared.digest(block),
            "start_codepoint": start, "end_codepoint": start + len(text),
        })
    assert len(rows) == 24 and len(bindings) == 15
    # Important negative finding: misleading raw contexts were NOT accepted as
    # the current dialogue context by the actual canonical normalizer.
    normalized_contexts = [
        {"pair_id": r["pair_id"], "condition": r["condition"],
         "normalized_dialogue_context": r["conclusion"]["follow_up_hypothesis"]["dialogue_context"]}
        for r in generated["attempts"]
        if r["condition"] == "ownership_rules" and r["id"] in {"fixed58-017", "fixed58-046"}
    ]
    assert len(normalized_contexts) == 4
    assert all(row["normalized_dialogue_context"] == "" for row in normalized_contexts)
    shared.save(HERE / "source_review_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generation_sha256": shared.sha(HERE / "generations.json"),
        "rows": rows, "source_bindings": bindings,
        "raw_context_not_carried_into_normalized_context": normalized_contexts,
        "human_gold": False, "production_promotion": False, "model_inputs_modified": False,
        "decision": "reject_generic_rule_promotion_retained_creator_error_and_lost_available_007_core",
        "limitations": [
            "Six development questions, not a blind independent set; two seeds are not a determinism or statistical guarantee.",
            "Only first-round actions; retrieve_more is not a final abstention and no second-round quality claim is made.",
            "Source-bound observations are assistant review, not human gold or correctness-rate labels.",
            "Do not describe quote-punctuation rejection as semantic-relation improvement.",
            "Bad raw dialogue_context is observable, but actual normalized contexts are empty; no demonstrated context propagation bug.",
        ],
    })
    print({"reviewed_raw_displays": len(rows), "source_excerpts": len(bindings),
           "normalized_contexts_empty": len(normalized_contexts)}, flush=True)


if __name__ == "__main__":
    main()
