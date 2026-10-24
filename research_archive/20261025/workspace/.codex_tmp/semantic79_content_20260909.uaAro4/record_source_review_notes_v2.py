"""Additional source checks; all automatic judgments remain untouched."""
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
previous = json.loads((HERE / "source_review_notes_v1.json").read_text())
notes = list(previous["notes"])
specs = [
    {
        "id": "relabel-3c4ca0be007bfbbb7dfde508::current_quote_fixed79",
        "review_verdict": "unsupported",
        "disputed_span": "莫斯提马的真实身份是拉特兰公证所的“苦难陈述者”。",
        "source_quotes": [
            {"evidence_id": "E1", "quote": "莫斯提马：名称是？ ？？？：......苦难陈述者。"},
            {"evidence_id": "E3", "quote": "苦难陈述者？：......我必须以拉特兰人的身份监视你。"},
        ],
        "reason": (
            "The title is the interlocutor's current assignment, not Mostima's. "
            "Mostima asks for its name and later addresses the bearer as a separate person who monitors her. "
            "No external identity/name is inferred for the masked speaker. "
            "The first claim about calling herself an ordinary messenger is separately supported by E2."
        ),
        "failure_type": "dialogue_questioner_misattributed_as_named_respondent",
    },
    {
        "id": "relabel-92de73ea55deebeb48f5d07c::current_quote_fixed79",
        "review_verdict": "no_answer",
        "disputed_span": "夜刀认为火龙可能与另一个为泰拉所熟知的危机有关联。",
        "source_quotes": [
            {"evidence_id": "E3", "quote": "不论矿石病的因素"},
            {"evidence_id": "E6", "quote": "能够一劳永逸地解决火龙本身，以及其可能会传播矿石病的威胁"},
        ],
        "reason": (
            "The question asks which crisis. The shown answer only says another familiar crisis, "
            "leaving the requested identity unspecified. No actual crisis is named in the answer, "
            "while visible E3/E6 identify the oripathy factor/threat. "
            "An entailed generic description is not a substantive answer to the identity slot."
        ),
        "failure_type": "requested_identity_replaced_by_generic_placeholder",
    },
]
for spec in specs:
    row = rows[spec["id"]]
    assert spec["disputed_span"] in row["grounded"]["answer"]
    blocks = evidence_id_text_map_from_prompt(row["evidence_brief"])
    for ref in spec["source_quotes"]:
        assert ref["quote"] in blocks[ref["evidence_id"]], ref
    notes.append({
        **spec, "question": row["question"], "shown_answer": row["grounded"]["answer"],
        "review_confidence": "high",
    })
for note in notes:
    if "verbatim_correction" in note:
        blocks = evidence_id_text_map_from_prompt(rows[note["id"]]["evidence_brief"])
        assert note["verbatim_correction"] in blocks[note["supporting_correction_evidence_id"]]
judgments = {}
lines = (HERE / "semantic79_content_v2.progress.jsonl").read_text().splitlines()
for index, line in enumerate(lines):
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        assert index == len(lines) - 1
        break
    judgments[result["id"]] = result
for note in notes:
    result = judgments.get(note["id"])
    note["auto_status_at_review"] = result["status"] if result else "pending"
    note["auto_verdict_at_review"] = (result.get("judgement") or {}).get("verdict") if result else None
    if result and result.get("judgement"):
        note["auto_reason_at_review"] = result["judgement"]["reason"]
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reviewer_type": "assistant_source_review_not_human_gold",
    "training_ready": False, "auto_judgments_modified": False,
    "analysis_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "reviewed_states": len(notes), "notes": notes,
    "limitations": [
        "Three source checks do not constitute review of all 79 states.",
        "No automatic label, training label or user-facing generated answer is changed.",
        "Overall model-audit scores remain provisional; no production accuracy claim.",
    ],
}
with (HERE / "source_review_notes_v2.json").open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
print(json.dumps({
    "reviewed_states": len(notes),
    "auto_statuses": [{"id": n["id"], "status": n["auto_status_at_review"],
                      "verdict": n["auto_verdict_at_review"]} for n in notes],
}, ensure_ascii=False))
