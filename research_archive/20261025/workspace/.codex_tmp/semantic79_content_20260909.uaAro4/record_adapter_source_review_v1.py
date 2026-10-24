"""Independent assistant source review, not human gold or training labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE / "adapter_pairs_validation_v1.json"
    output = HERE / "adapter_pairs_source_review_v1.json"
    assert not output.exists()
    analysis = json.loads(source.read_text())
    rows = analysis["results"]
    notes = {
        "89641928246119c3": {
            "anchors": ["我们已经多久没为一件高兴事忙起来了？", "刚刚我发现，食物其实还剩一点儿", "主题就是“乌萨斯风味甜点”"],
            "off": {
                "assessment": "not_shown_quote_rewrite",
                "notes": [
                    "E4 quote was changed from the original rhetorical question into 我已经很久没为一件高兴事忙起来了. Runtime correctly rejected it as quote_not_found.",
                    "The event name and participation core have context support; this does not authorize repairing the quote while inheriting a positive label.",
                ],
            },
            "on": {
                "assessment": "shown_core_supported_activity_summary_coarse",
                "notes": [
                    "The displayed answer names the food festival, preparation and food sharing, supported by E1/E4 and the complete context.",
                    "The more concrete stall/riddle/dessert details in raw final_answer are not in the displayed answer after the supported-fact rewrite.",
                    "Do not silently score the richer raw answer instead of the actual shown answer; no independent human completeness label is assigned.",
                ],
            },
        },
        "b5c3bd1132dc9df8": {
            "anchors": ["重力会使得它始终处于轨道底端", "提供足够推力能直接推动“行星碎屑”", "一定次数后，能够将“行星碎屑”朝干员部署方向推动"],
            "off": {
                "assessment": "shown_supported_mechanics_causal_completeness_unresolved",
                "notes": [
                    "Displayed facts contain the effect, gravity/track constraint and sufficient pushing force, adding actuation information over earlier effect-only outputs.",
                    "The displayed answer still does not explicitly join movement/collision to damage; do not predeclare full why-question success.",
                    "Raw final_answer adds designed-purpose/mass reasoning and combines charging plus pushing; these are not all established by the selected short quotes and are not the shown answer.",
                ],
            },
            "on": {
                "assessment": "not_shown_malformed_quote_json",
                "notes": [
                    "The second quote opens with a Chinese quote character rather than the required JSON string delimiter. Recovery leaves fact2 without evidence_refs.",
                    "It was rejected despite finish_reason=stop and ample context; this is malformed output, not token-limit truncation.",
                ],
            },
        },
        "relabel-23ebdc399dbe63646b934d15": {
            "anchors": ["他们嫌弃我身为女妖却不擅使用咒言，我嫌弃他们制作的捕猎笼效率太低", "我们总是互相原谅，互相拯救，互相庇护", "就像现在一样，就像本来应该的那样", "然后一切都将变成所谓的“命运”"],
            "off": {
                "assessment": "shown_reciprocity_core_supported_interpretive_extension_unresolved",
                "notes": [
                    "Mutual active help/equality rather than one-sided receiving is supported by E1/E2.",
                    "Refusal to passively accept fate is a broader interpretive extension, not what the fact1 quote directly states. E3 contains a fate discussion by Kristen, so do not assert no related context exists.",
                    "Fact2 cites only 就像现在一样...; the complete E1 explains mutuality, but that short quote alone does not establish every clause.",
                ],
            },
            "on": {
                "assessment": "shown_supported_core_with_unsupported_temporal_contrast",
                "notes": [
                    "Mutual active support is supported, but contrasting the ideal relationship against a past experience of being disliked misreads the cited relation.",
                    "E1 recounts mutual disparagement together with reciprocal help as an example of the same pattern, explicitly 就像现在一样, not an opposite past mode.",
                    "A formerly stable positive-control question is not guaranteed correct under a new generation.",
                ],
            },
        },
        "relabel-4835313d2c74e78706d2b4e3": {
            "anchors": ["比起苦闷焦虑，我还是希望我们都能尽量地为未来做一些美好的打算吧", "莱欧斯其实比看起来要更着急", "为了把我们在路上的时间缩到最短"],
            "off": {
                "assessment": "shown_unsupported_causal_link",
                "notes": [
                    "E1 explains Marcille's hobbies and hopeful preparation for meeting Falin; it does not establish that these caused her to think the party was not hurrying.",
                    "E1 instead describes Laios as more anxious than he looks and Chilchuck minimizing travel time. A matching quote does not validate the claimed causal link.",
                    "Complete E1-E6 were read. Other cooking/search scenes do not establish the specific cause asserted in this shown answer.",
                ],
            },
            "on": {
                "assessment": "retrieve_more_no_answer",
                "notes": ["No answer was shown. This avoids this particular unsupported cause but is not a successful final answer or proof that current retrieval is sufficient/insufficient."],
            },
        },
        "relabel-92de73ea55deebeb48f5d07c": {
            "anchors": ["它极有可能和另一个为泰拉所熟知的危机紧密相关", "不论矿石病的因素", "以及其可能会传播矿石病的威胁"],
            "off": {
                "assessment": "shown_missing_crisis_identity",
                "notes": ["It repeats 另一个为泰拉所熟知的危机 without naming the crisis. E3/E6 provide the relevant oripathy context; this is an unfilled asked identity slot despite valid quotes."],
            },
            "on": {
                "assessment": "shown_missing_crisis_identity_with_modal_strengthening",
                "notes": [
                    "It likewise leaves the crisis unnamed. The extra sentence that the creature requires serious handling does not fill the missing identity.",
                    "Raw final_answer says 极有可能; supported_fact1 drops this uncertainty, and the runtime's fact-based rewrite displays the stronger assertion. This is not a truncation fix.",
                ],
            },
        },
        "relabel-c637f5c136cd58d46096f7ce": {
            "anchors": ["因为我太固执，我太自大，我把钱，把生活，把他的选择......全都透支掉了", "莱昂图索以为自己明白他们想说的事情", "一个马上就要完蛋的达维镇"],
            "off": {
                "assessment": "shown_core_supported_by_full_cited_block_not_all_short_quotes",
                "notes": [
                    "The concise stubbornness/arrogance and exhausted choices answer has direct support in E3, without the wrong character or completed-town-collapse claim.",
                    "Its selected short quotes omit the strongest exact stubbornness/choices sentence. Full-block support is not the same as each displayed quote independently entailing the fact.",
                ],
            },
            "on": {
                "assessment": "not_shown_quote_rewrite_and_wrong_character_claim",
                "notes": [
                    "The first quote starts 可我太固执 instead of source 因为我太固执 and is mechanically rejected.",
                    "Independently, fact2 attributes E1's 莱昂图索/embellished-violence reflection to 里昂 without source support.",
                    "The first fact also changes a town about to fail into completed failure. Do not label the entire stubbornness core invented; separate correct core and overstatement.",
                ],
            },
        },
    }
    reviewed = []
    for row in rows:
        matching = [prefix for prefix in notes if row["id"].startswith(prefix)]
        assert len(matching) == 1
        note = notes[matching[0]]
        assert all(anchor in row["evidence_brief"] for anchor in note["anchors"])
        variant = "off" if row["renderer"] == "adapter_off_same_tokenizer_v1" else "on"
        reviewed.append({
            "id": row["id"], "question": row["question"], "variant": variant,
            "grounded_action": row["grounded"]["next_action"], "shown_answer": row["grounded"]["answer"],
            "evidence_sha256": hashlib.sha256(row["evidence_brief"].encode()).hexdigest(),
            "assessment": note[variant]["assessment"], "notes": note[variant]["notes"],
            "verified_source_anchors": note["anchors"],
        })
    assert len(reviewed) == 12
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent_assistant_source_review_not_human_gold",
        "evaluation_only": True, "training_ready": False, "source_sha256": sha(source),
        "reviewed_rows": reviewed, "source_scope": "Complete model-visible evidence for all six questions, including the long food-festival and planet-debris blocks.",
        "limitations": [
            "Selected development questions; no production or holdout accuracy estimate.",
            "Keep nonanswers and interpretive uncertainty, never infer a complete-correct count from release counts.",
            "Original generated answers and any prior automatic labels are preserved, not patched.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"source_review": str(output), "rows": 12, "training_ready": False}))


if __name__ == "__main__":
    main()
