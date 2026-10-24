"""Record source-checked review notes, preserving all original machine labels."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
analysis_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
audit_path = run / "fixed79_configured_lora_ctx12288/rejected34.semantic_raw_retry1.json"
rebind_path = run / "fixed79_configured_lora_ctx12288/exact_rebinding79.counterfactual.json"
original = json.loads(analysis_path.read_text())
audit = json.loads(audit_path.read_text())
counterfactual = json.loads(rebind_path.read_text())
assert audit["source_analysis_sha256"] == sha(run / "fixed79_configured_lora_ctx12288/rejected34.raw_analysis.json")
assert counterfactual["source_analysis_sha256"] == sha(analysis_path)
by_id = {r["id"]: r for r in original["results"]}
judges = {r["id"]: r for r in audit["results"]}
rebound = {r["id"]: r for r in counterfactual["results"]}
specs = [
    {
        "prefix": "relabel-25610af6b165944933207d23",
        "target": "raw_rejected_proposal",
        "answer_excerpt": "虚弱的女妖在骸骨荒原寻找的人是拂哀菈",
        "evidence_excerpts": [
            "娜斯提：谢谢你照顾我的母亲，费尔南小姐。",
            "一路辗转，寻着娜斯提的踪迹来到哥伦比亚，已经超过半年。",
        ],
        "assessment": "raw_judge_false_positive_subject_object_reversal",
        "note": "第一分句把被寻找者说成拂哀菈，第二分句又说她寻找娜斯提。judge将第一句默改读为“女妖是拂哀菈”，才判完整。不能替实际答案改语法来消除矛盾；原答案不应判完整。这条实际被拦，未展示。",
    },
    {
        "prefix": "relabel-f5f3526070623b86994d7fc9",
        "target": "raw_rejected_proposal",
        "answer_excerpt": "她曾对担忧的居民说",
        "evidence_excerpts": [
            "布莉吉：艾玛，你醒了？ 担忧的居民：尼亚——尼亚尔！",
            "拉芙希妮：不，她还睡着。",
            "担忧的居民：尼亚尔...... 你骂我吧，打我也可以，只是别丢下我不管......",
        ],
        "assessment": "raw_judge_missed_speaker_recipient_error",
        "note": "“担忧的居民”就是艾玛的角色标签；相关话语是她睡梦中对尼亚尔的呼喊，不是艾玛对另一名担忧居民所说。judge只判部分、称所答内容均受支持，漏掉了这个实质性错误。实际答案被拦，未展示。",
    },
    {
        "prefix": "relabel-c637f5c136cd58d46096f7ce",
        "target": "raw_rejected_proposal",
        "answer_excerpt": "导致达维镇完蛋",
        "evidence_excerpts": [
            "因为我太固执，我太自大，我把钱，把生活，把他的选择......全都透支掉了。",
            "争到了一个马上就要完蛋的达维镇",
        ],
        "assessment": "supported_core_with_temporal_scope_caution",
        "note": "自认固执自大的核心成立。原文描述小镇即将完蛋，答案省去“马上就要”，有把将来危机写成既成结果的风险；也可读成口语概括，暂记时态精度争议，不强行判全错。",
    },
    {
        "prefix": "relabel-328279d9de236c232c8fd571",
        "target": "counterfactual_rebound_display",
        "answer_excerpt": "芙蕾达为了生存不得不向萨卡兹低头。",
        "evidence_excerpts": [
            "他骂你就是仗着魔族佬气焰的叛徒！",
            "大家现在还能勉强过日子，难道不是你低声下气从魔族佬那求来的？",
            "明明我什么都听他们的了。我完成了他们交代的所有要求，为什么还要留下威尔！",
        ],
        "assessment": "rebound_core_supported_explanation_quality_limited",
        "note": "E3->E6唯一逐字定位成立。重校验后的最终文本已不包含原答案“背叛乌萨斯”和错误的因为句，核心对立可由E3支持。但事实拼接重复，未清楚解释镇民从依附压迫者评价、威尔从保住大家生计评价的分歧。只记潜在修复机会，不记完整正确率提升。",
    },
    {
        "prefix": "relabel-3cb890fce036c72e3174bbc4",
        "target": "counterfactual_rebound_display",
        "answer_excerpt": "巫女林是一个收容感染者术师的组织。",
        "evidence_excerpts": [
            "夏栎：巫女林？是那个收容感染者术师的组织？ 苦根：是她们。",
            "我是巫女林的第一百一十三位女巫，夜烟......",
            "曾经的女巫同伴们，那位慈祥的大女巫格兰特，维多利亚军人的火炮点燃巫女林的那个夜晚。",
        ],
        "assessment": "rebound_definition_affiliation_supported_causal_question_ambiguous",
        "note": "E8->E12唯一逐字定位成立；组织性质和夜烟成员身份有支持。若“为何来自那里”是在问身份依据，已给核心信息；若问加入缘由，回答仍近似同义反复，证据也未直接交代加入过程。不能以放行或引用真实就统一判完整。未部署。",
    },
    {
        "prefix": "f943e2bd0a766456",
        "target": "raw_rejected_proposal",
        "answer_excerpt": "陈家因联姻传闻被苛待",
        "evidence_excerpts": [
            "陛下没有因为这件事为难陈家，但公主家中不睦的流言却遍传朝野。",
            "有了苛待公主的恶名，陈家在京城的遭遇可想而知。",
        ],
        "assessment": "raw_judge_over_generous_rumor_content",
        "note": "原文流言是陈家苛待公主/家中不睦，不是婚姻消息本身。答案混淆了流言内容和遭遇；judge把它善意改述为“受流言牵连”后只判部分。ID反事实只修两处，短引用不修，仍未放行；不为提高放行数进一步放宽。",
    },
]
notes = []
for spec in specs:
    matches = [r for key, r in by_id.items() if key.startswith(spec["prefix"])]
    assert len(matches) == 1
    row = matches[0]
    item = dict(spec)
    item.update(id=row["id"], question=row["question"])
    answer = (rebound[row["id"]]["grounded"]["answer"]
              if spec["target"] == "counterfactual_rebound_display" else row["parsed"]["answer"])
    assert spec["answer_excerpt"] in answer
    assert all(span in row["evidence_brief"] for span in spec["evidence_excerpts"])
    item.update(
        reviewed_answer=answer,
        original_production_action=row["grounded"]["next_action"],
        machine_raw_verdict=judges.get(row["id"], {}).get("judgement", {}).get("verdict"),
        all_recorded_spans_verified=True,
    )
    notes.append(item)
report = {
    "version": "same_evidence_repair_source_review_notes_v2",
    "reviewer_type": "agent_source_review_not_independent_human_gold",
    "evaluation_only": True, "training_ready": False, "original_judgements_overwritten": False,
    "analysis_sha256": sha(analysis_path), "raw_audit_sha256": sha(audit_path),
    "counterfactual_sha256": sha(rebind_path), "script_sha256": sha(Path(__file__)),
    "notes": notes,
    "limitations": [
        "Targeted review documents specific disagreements, not all-record human adjudication.",
        "Raw rejected proposals and counterfactual displays are not actual production answers.",
        "A source-supported answer can still have incomplete explanation or inaccurate attached citations.",
    ],
}
with (run / "repair_source_review_notes_v2.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"verified_review_notes": len(notes), "original_judgements_overwritten": False}))
