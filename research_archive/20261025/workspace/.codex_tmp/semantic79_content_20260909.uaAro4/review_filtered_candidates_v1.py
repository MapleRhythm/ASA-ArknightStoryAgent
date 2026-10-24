"""Source-scoped assistant checks of the seven frozen diagnostic candidates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt

candidates_path = HERE / "semantic_filter7_candidate_v1.json"
candidates = json.loads(candidates_path.read_text())["results"]
auto_path = HERE / "semantic_filter7_audit_v1.json"
auto = {r["id"]: r for r in json.loads(auto_path.read_text())["results"]}
specs = {
    "0b3f0002117b06cb__": {
        "verdict": "supported_complete",
        "reason": "Only the independently supported parent occupation/sacrifice claim remains; the wrong ancestor-origin claim is absent.",
        "refs": [("E10", "我的父母是灯塔维修师 他们为伊比利亚之眼而牺牲了")],
    },
    "2711ab940eb8a3b6__": {
        "verdict": "supported_complete",
        "reason": "E6 directly supports the promise and his response about their joint stage. The compressed, ambiguous weakness/charity assertion is absent.",
        "refs": [("E6", "我来把谢拉格变成一个足够大，能够让你施展拳脚的舞台"),
                 ("E6", "既然要做的话，应该是我们一起把谢拉格变成一个足够大，足够让我们施展拳脚的舞台")],
    },
    "relabel-23ebdc399dbe63646b934d15::": {
        "verdict": "supported_complete",
        "reason": "The supported interpretive control is unchanged; reciprocal help addresses the asked implication.",
        "refs": [("E1", "我们总是互相原谅，互相拯救，互相庇护。 就像现在一样")],
    },
    "relabel-3c4ca0be007bfbbb7dfde508::": {
        "verdict": "no_answer",
        "reason": "Abstention caused by one malformed claim-judge quotation is not a correct answer; supported first fact was also withheld by the frozen rule.",
        "refs": [],
    },
    "relabel-92de73ea55deebeb48f5d07c::": {
        "verdict": "no_answer",
        "reason": (
            "Neither shown sentence names the asked crisis. The automatic audit incorrectly says the evidence does not "
            "identify it, despite E3's oripathy factor and E6's possible spread. A true generic description plus an "
            "unrelated supported sentence is still not an answer to 'which crisis'."
        ),
        "refs": [("E3", "不论矿石病的因素"),
                 ("E6", "能够一劳永逸地解决火龙本身，以及其可能会传播矿石病的威胁")],
    },
    "relabel-f4ec283e29fcc994d667b6b1::": {
        "verdict": "no_answer",
        "reason": "The compound fact was removed as a whole, including its supported gym-operation subclaim; the result is abstention, not recovered answer correctness.",
        "refs": [],
    },
    "relabel-ffc93a494808c61623543ea9::": {
        "verdict": "supported_complete",
        "reason": (
            "The retained sentence states the disputed cooperative action and life-protecting motive, addressing the "
            "root of the already-stated opposing evaluations. It no longer reverses Will's stance or invents detention causality."
        ),
        "refs": [("E6", "可我不能赌上全镇人的性命去冒险。"),
                 ("E6", "我......不敢再轻易打碎求之不易的和平。"),
                 ("E1", "大家现在还能勉强过日子，难道不是你低声下气从魔族佬那求来的？")],
    },
}
notes = []
for candidate in candidates:
    selected = [s for prefix, s in specs.items() if candidate["id"].startswith(prefix)]
    assert len(selected) == 1
    spec = selected[0]
    blocks = evidence_id_text_map_from_prompt(candidate["evidence_brief"])
    refs = []
    for eid, quote in spec["refs"]:
        assert quote in blocks[eid], (eid, quote)
        refs.append({"evidence_id": eid, "quote": quote})
    if candidate["id"].startswith("relabel-92de"):
        assert "矿石病" not in candidate["grounded"]["answer"]
    notes.append({
        "id": candidate["id"], "question": candidate["question"],
        "shown_answer": candidate["grounded"]["answer"], "review_verdict": spec["verdict"],
        "reason": spec["reason"], "source_quotes": refs,
        "review_scope": "shown_answer_and_relevant_complete_visible_blocks_not_all_visible_blocks",
        "automatic_verdict": (auto[candidate["id"]].get("judgement") or {}).get("verdict"),
    })
assert len(notes) == 7
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reviewer_type": "assistant_source_review_not_human_gold",
    "candidate_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
    "automatic_audit_sha256": hashlib.sha256(auto_path.read_bytes()).hexdigest(),
    "training_ready": False, "promotion_approved": False, "automatic_labels_modified": False,
    "notes": notes,
    "limitations": [
        "Source-scoped review of deliberately selected development states is not a general accuracy estimate.",
        "A provisional supported verdict is not training approval or independent gold.",
        "No runtime semantic gate, served answer, training target or automatic label is changed.",
    ],
}
with (HERE / "semantic_filter7_source_notes_v1.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({
    "reviewed_states": len(notes), "disagreements": [
        {"id": n["id"], "automatic": n["automatic_verdict"], "review": n["review_verdict"]}
        for n in notes if n["automatic_verdict"] != n["review_verdict"]
    ]
}, ensure_ascii=False))
