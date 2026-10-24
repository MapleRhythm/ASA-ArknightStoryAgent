"""Additional scoped source review, kept separate from all automatic labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt

source = HERE / "canonical_fixed79_validation_v1.json"
rows = {r["id"]: r for r in json.loads(source.read_text())["results"]}
previous = json.loads((HERE / "source_review_notes_v2.json").read_text())
notes = list(previous["notes"])
specs = [
    {
        "prefix": "2711ab940eb8a3b6__",
        "review_verdict": "uncertain", "review_confidence": "medium",
        "disputed_span": "诺希斯认为谢拉格弱小，是维多利亚施舍的结果，不愿继续现状。",
        "source_quotes": [{"evidence_id": "E5", "quote": "只是因为现在的谢拉格太弱小了，维多利亚可怜你们，才施舍你们一些东西"}],
        "reason": (
            "The source makes weakness the reason for charity. As written, the displayed first clause "
            "can instead make weakness the result of charity. Certifying it requires repairing an omitted subject "
            "(development) or reversing the causal reading. Mark ambiguous rather than silently fixing the answer. "
            "The second promise claim is independently supported by E6."
        ),
        "failure_type": "causal_direction_ambiguous_after_compression",
    },
    {
        "prefix": "relabel-f4ec283e29fcc994d667b6b1::",
        "review_verdict": "unsupported", "review_confidence": "high",
        "disputed_span": "卡铎尔则拒绝服从推进之王并随之撤离接受温德米尔公爵庇护",
        "source_quotes": [{"evidence_id": "E2", "quote": "卡铎尔拒绝服从推进之王并随之撤离接受温德米尔公爵庇护，留在拳馆。"}],
        "reason": (
            "The complete source clause ends with staying at the gym. The answer drops that resolution, "
            "and presents departure/accepting protection as an event after refusing obedience. "
            "The later evacuation event is also not the requested earlier process of regaining a foothold."
        ),
        "failure_type": "scope_of_refusal_and_event_time_lost",
    },
    {
        "prefix": "relabel-ffc93a494808c61623543ea9::",
        "review_verdict": "unsupported", "review_confidence": "high",
        "disputed_span": "威尔因萨卡兹军营扣人而愤怒，指责芙蕾达向萨卡兹低头求情导致威尔被扣留。",
        "source_quotes": [
            {"evidence_id": "E1", "quote": "但是汤姆那个混球沾酒就胡说，他骂你就是仗着魔族佬气焰的叛徒！"},
            {"evidence_id": "E1", "quote": "大家现在还能勉强过日子，难道不是你低声下气从魔族佬那求来的？"},
            {"evidence_id": "E1", "quote": "我和老汉克气不过就和他赌酒"},
        ],
        "reason": (
            "Will objects to Tom insulting Freida and defends her negotiations as the reason people can still live. "
            "The answer reverses his stance and invents the cause of his detention. "
            "Freida's refusal to risk the whole town is separately supported by E6."
        ),
        "failure_type": "reported_insult_misattributed_and_stance_reversed",
    },
    {
        "prefix": "relabel-6e628226188dc73db52ba8ad::",
        "review_verdict": "unsupported", "review_confidence": "high",
        "disputed_span": "试图挽回家族关系",
        "source_quotes": [
            {"evidence_id": "E1", "quote": "难道说，你以为我是为了向这个老东西摇尾乞怜，让他同意我回到这个该死，哦，不，这个伟大的家族而回来的吗？"},
            {"evidence_id": "E1", "quote": "为家族带来至关重要的情报"},
            {"evidence_id": "E5", "quote": "因为除此之外，我感受不到一点乐趣，我的父亲。"},
        ],
        "reason": (
            "The source rejects the interpretation of returning to beg for readmission. Her following family-loyalty "
            "speech cannot safely be treated as a literal reconciliation motive, especially with E5's explanation. "
            "The expulsion cause in the first answer sentence has independent support in E2 and is not disputed."
        ),
        "failure_type": "explicit_denial_and_sarcasm_recast_as_reconciliation_motive",
    },
    {
        "prefix": "relabel-23ebdc399dbe63646b934d15::",
        "review_verdict": "supported_complete", "review_confidence": "high",
        "disputed_span": None,
        "source_quotes": [
            {"evidence_id": "E1", "quote": "被原谅，被拯救，受庇护"},
            {"evidence_id": "E1", "quote": "我们总是互相原谅，互相拯救，互相庇护。"},
            {"evidence_id": "E1", "quote": "就像现在一样，就像本来应该的那样。"},
        ],
        "reason": (
            "Interpretive control: the question explicitly asks for an implication. Mutual help versus a passive, "
            "one-way recipient role is a grounded interpretation of the dialogue, not an unrelated plot fact. "
            "Do not reject every paraphrase or interpretation merely because it is not a verbatim source sentence."
        ),
        "failure_type": None,
    },
]
for spec in specs:
    matches = [row for rid, row in rows.items() if rid.startswith(spec["prefix"])]
    assert len(matches) == 1
    row = matches[0]
    if spec["disputed_span"] is not None:
        assert spec["disputed_span"] in row["grounded"]["answer"]
    blocks = evidence_id_text_map_from_prompt(row["evidence_brief"])
    for ref in spec["source_quotes"]:
        assert ref["quote"] in blocks[ref["evidence_id"]], ref
    notes.append({
        **{k: v for k, v in spec.items() if k != "prefix"},
        "id": row["id"], "question": row["question"], "shown_answer": row["grounded"]["answer"],
        "review_scope": "shown_answer_and_relevant_complete_visible_blocks_not_all_visible_blocks",
        "reviewed_block_ids": sorted({r["evidence_id"] for r in spec["source_quotes"]}),
    })
automatic = {
    r["id"]: r for r in json.loads((HERE / "semantic79_content_v2.json").read_text())["results"]
}
for note in notes:
    judgment = automatic[note["id"]]
    note["auto_status_at_review"] = judgment["status"]
    note["auto_verdict_at_review"] = (judgment.get("judgement") or {}).get("verdict")
    note["auto_reason_at_review"] = (judgment.get("judgement") or {}).get("reason")
    note["agrees_with_automatic_verdict"] = note["review_verdict"] == note["auto_verdict_at_review"]
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reviewer_type": "assistant_source_review_not_human_gold",
    "training_ready": False, "auto_judgments_modified": False,
    "analysis_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "reviewed_states": len(notes), "notes": notes,
    "limitations": [
        "Eight selected source checks are not review of all 39 direct answers or all 79 states.",
        "New checks read relevant complete E blocks, not every visible E block for each question.",
        "Medium-confidence causal ambiguity remains uncertain, not forced into a definite error.",
        "Automatic labels and displayed answers remain untouched; no corrected overall accuracy is asserted.",
    ],
}
with (HERE / "source_review_notes_v3.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({
    "reviewed_states": len(notes),
    "verdict_disagreements": [
        {"id": n["id"], "automatic": n["auto_verdict_at_review"], "review": n["review_verdict"]}
        for n in notes if not n["agrees_with_automatic_verdict"]
    ],
}, ensure_ascii=False))
