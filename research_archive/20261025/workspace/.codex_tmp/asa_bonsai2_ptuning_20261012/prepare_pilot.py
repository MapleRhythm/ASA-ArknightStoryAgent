"""Small, manually specified storyline pilot; full original index documents.

These are offline evidence packs, not live-retrieval evaluation. Labels are
conservative rewrites of inspected original excerpts, not fresh API labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

RECIPES = [
    ("train", "凛冬在学生自治团中扮演什么角色？她是怎样成为领袖的？", [
        ("凛冬曾是乌萨斯学生自治团的领袖；她自述经过一段时间获得了大家的信任与拥戴，但这段证据没有给出完整的推举过程。",
         "obt/memory/story_headbr_2_1#chunk-0012")]),
    ("train", "马里亚姆主任的盒子里装的是什么，为什么它如此危险？", [
        ("盒中是一朵没有颜色的花；标本箱启封后，它会浸染实验室的空间，固定位置并构成自己的领土。",
         "activities/act15mini/level_act15mini_st07#chunk-0015"),
        ("打开盒子会使安德斯科塔尔尼尔觉察。",
         "activities/act15mini/level_act15mini_st07#chunk-0015")]),
    ("train", "小樱桃用来练习写字的炉灰有什么特殊含义？她知道真相吗？", [
        ("小樱桃写字的灰堆里混着萨卡兹的骨灰；谈话者推测她大概不知道，不能把这一推测说成已确认的事实。",
         "activities/act17mini/level_act17mini_st03#chunk-0009")]),
    ("train", "乌尔比安为什么声称可能需要杀死斯卡蒂？", [
        ("乌尔比安认为斯卡蒂面对海嗣神祇时发生的不是正常的杀死或捕食，而是族群的一次喂食。",
         "activities/act17side/level_act17side_st08#chunk-0015"),
        ("他警告如果斯卡蒂发生问题就得杀死她；这是针对潜在风险的条件性警告，不是已经决定无条件杀死她。",
         "activities/act17side/level_act17side_st08#chunk-0016")]),
    ("train", "为什么说大荒城的五豆粥是特色？", [
        ("它的关键特色在于以大荒城独特的稻米为基底：当地稻米香甜、风味浓郁，使五豆粥成为当地饮食文化的象征。",
         "activities/act31side/level_act31side_01_beg#chunk-0011")]),
    ("train", "香草为什么从萨尔贡家乡出来当佣兵？", [
        ("香草自述加入黑钢当佣兵是因为缺钱。",
         "activities/act7d5/level_act7d5_st01#chunk-0021"),
        ("她提到自己的家乡贫瘠、童年时商队会讲述沙漠外的繁华世界；但仅凭这段不能把听故事直接认定为加入佣兵的原因。",
         "activities/act7d5/level_act7d5_st01#chunk-0022")]),
    ("val", "霍尔海雅向弗里斯顿提出了什么羽蛇血脉相关的请求？", [
        ("霍尔海雅希望获得羽蛇血脉中的翅膀、风雷力量和翱翔的荣耀，背负着羽蛇家族四百五十一年的执念。",
         "activities/act25side/level_act25side_st03#chunk-0027")]),
    ("val", "祖拜尔在临终时表达了怎样的心愿？", [
        ("祖拜尔希望以完整人类的身份离开。",
         "activities/act35side/level_act35side_09_end#chunk-0005"),
        ("他认为宝石彻底碎裂后会获得永恒的宁静，并请求进入存放着独属于自己的东西的宝石中央。",
         "activities/act35side/level_act35side_09_end#chunk-0000")]),
]
ANCHORS = {
    "obt/memory/story_headbr_2_1#chunk-0012": "获得了大家的信任与拥戴",
    "activities/act15mini/level_act15mini_st07#chunk-0015": "安德斯科塔尔尼尔",
    "activities/act17mini/level_act17mini_st03#chunk-0009": "大概是不知道的",
    "activities/act17side/level_act17side_st08#chunk-0015": "族群的喂食",
    "activities/act17side/level_act17side_st08#chunk-0016": "发生任何问题，我们就得杀死她",
    "activities/act31side/level_act31side_01_beg#chunk-0011": "大荒城稻米香甜",
    "activities/act7d5/level_act7d5_st01#chunk-0021": "是因为缺钱",
    "activities/act7d5/level_act7d5_st01#chunk-0022": "我的家乡，非常贫瘠",
    "activities/act25side/level_act25side_st03#chunk-0027": "羽蛇家族四百五十一年",
    "activities/act35side/level_act35side_09_end#chunk-0005": "完整的人类的身份离开",
    "activities/act35side/level_act35side_09_end#chunk-0000": "会获得永恒的宁静",
}
RULES = (
    "只使用当前可见证据；next_action只能是answer_directly、retrieve_more或abstain；"
    "answer_directly时只输出next_action和1至8条supported_facts，每个可核验原子事实的"
    "evidence_ids绑定1至2个当前存在的E编号；retrieve_more时输出follow_up_hypothesis；"
    "abstain时输出reason；不要复制引文，不要输出evidence_refs、quote、final_answer、answer"
    "或inferred_facts；只保留完整回答问题所必需的最少事实，通常1至4条，禁止重复或近义改写"
    "同一事实；当前证据足够时不要补检索；证据不足且未到最后一轮时才retrieve_more，"
    "最后一轮则abstain。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", required=True)
    ap.add_argument("--out", default="pilot_data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    prefixes = {doc.split("#")[0] for _, _, facts in RECIPES for _, doc in facts}
    family_prefixes = {"/".join(doc.split("/")[:2]) + "/" for doc in prefixes}
    docs = {}
    digest = hashlib.sha256()
    with open(args.documents, "rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if any(row["id"].startswith(prefix) for prefix in family_prefixes):
                docs[row["id"]] = row
    all_rows, review = [], []
    for idx, (split, question, facts) in enumerate(RECIPES):
        required = list(dict.fromkeys(doc for _, doc in facts))
        assert all(doc in docs for doc in required), required
        assert all(ANCHORS[doc] in docs[doc]["clean_text"] for doc in required)
        assert all("act10d5" not in doc and "act10mini" not in doc for doc in required)
        candidate_prefixes = {"/".join(x.split("/")[:2]) + "/" for x in required}
        candidates = [doc for doc in docs if any(doc.startswith(x) for x in candidate_prefixes)
                      and doc not in required]
        # Neighbors selected offline around the supporting passage, not new runtime rules.
        def distance(doc):
            number = int(doc.split("#chunk-")[-1])
            same_story = doc.split("#")[0] in {x.split("#")[0] for x in required}
            return (not same_story,
                    min(abs(number - int(x.split("#chunk-")[-1])) for x in required), doc)
        candidates.sort(key=distance)
        selected = required + candidates[:20 - len(required)]
        assert len(selected) == 20, (question, len(selected))
        random.Random(20261012 + idx).shuffle(selected)
        evidence = []
        for eid, doc in enumerate(selected, 1):
            raw = docs[doc]["clean_text"]
            # Keep the entire indexed passage byte-for-byte.
            assert raw.strip()
            evidence.append({"evidence_id": f"E{eid}", "document_id": doc,
                             "text": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()})
        id_map = {x["document_id"]: x["evidence_id"] for x in evidence}
        target = {"next_action": "answer_directly", "supported_facts": [
            {"fact": fact, "evidence_ids": [id_map[doc]]} for fact, doc in facts]}
        body = (
            f"task: grounded_action_generation\nquestion: {question}\n"
            'hypothesis: {"query_type":"reasoning","expected_answer_type":"剧情解释"}\n'
            "round: 2/2\nevidence:\n" +
            "\n".join(f"[{x['evidence_id']}]\n{x['text']}" for x in evidence) +
            "\noutput_schema: grounded_action_exx_v1\nrules: " + RULES)
        # Standard native nonthinking assistant prefill; no textual CoT targets.
        prompt = (
            "<|im_start|>system\n你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
            "<|im_end|>\n<|im_start|>user\n" + body +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        row = {"id": f"pilot-{idx}", "split": split, "question": question,
               "prompt": prompt, "target": json.dumps(target, ensure_ascii=False),
               "documents": evidence, "evidence_count": 20, "truncated": False,
               "source_family": required[0].split("/")[1]}
        assert all(x["text"] in prompt for x in evidence)
        all_rows.append(row)
        review.append({"id": row["id"], "split": split, "question": question, "target": target,
                       "supporting_original_documents": [docs[x] for x in required]})
    families = {split: {x["source_family"] for x in all_rows if x["split"] == split}
                for split in ("train", "val")}
    assert not families["train"] & families["val"]
    for split in ("train", "val"):
        (out / f"{split}.jsonl").write_text("".join(
            json.dumps(x, ensure_ascii=False) + "\n" for x in all_rows if x["split"] == split))
    (out / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "source_index_sha256": digest.hexdigest(), "train": 6, "val": 2,
        "evidence_count_per_example": 20, "truncated_documents": 0,
        "train_val_source_family_overlap": [], "new_api_labels": 0,
        "kind": "tiny_offline_evidence_reader_pilot_not_full_chain_eval",
        "historical_training_sources_not_unseen_benchmark": True,
        "manual_model_assistant_review_pending": False,
        "review": "assistant read original passages and corrected stale citation IDs; not independent human gold",
        "stale_source_citation_ids_corrected": 4,
    }, indent=2))
    print(json.dumps({"train": 6, "val": 2, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
