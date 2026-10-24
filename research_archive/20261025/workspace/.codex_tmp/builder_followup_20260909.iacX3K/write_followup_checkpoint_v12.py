"""Publish the tested data-contract repair and failed anchor-ablation checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ANCHOR = RUN / "anchor_ablation_v1.VHleHq"
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
DEST = ROOT / "docs/CANONICAL_FOLLOWUP_INTEGRITY_AND_ANCHOR_ABLATION_20260908_irPugk_v12.md"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not DEST.exists()
    applied = load(HERE / "applied_verified.json")
    replay = load(HERE / "stored_followup_replay_v2.json")
    post = load(ANCHOR / "anchor_ablation_postflight_v1.json")
    analysis = load(ANCHOR / "anchor_ablation_validation_v1.json")
    assert applied["canonical_tests"] == {"tests": 276, "failures": 0, "errors": 0, "skipped": 0}
    assert all(sha(ROOT / p) == digest for p, digest in applied["applied_sha256"].items())
    assert all(sha(Path(p)) == digest for p, digest in replay["source_sha256"].items())
    assert replay["dropped_retrieve_positive_count"] == 31
    assert replay["changed_retrieve_positive_count"] == 29
    assert post["base_and_adapter_assets_unchanged"] and post["runtime_source_unchanged"]
    notes = []
    for row in analysis["results"]:
        rid = row["id"]
        no_previous = row["renderer"] == "no_previous_answer_v1"
        if rid.startswith("896419"):
            verdict = "no_answer"
            reason = (
                "两组都被grounding转为retrieve_more，展示答案为空。无旧答案组quote擅加说话人标记导致不匹配；"
                "有旧答案组quote超过80字符。未展示的raw答案不计已答对。"
            )
        elif rid.startswith("b5c3"):
            verdict = "not_complete_mechanism_unresolved"
            reason = (
                "两组展示答案完全相同，仍只陈述伤害/推动效果和轨道/重力规则，没有补足所问机制。"
                "无旧答案组raw final_answer用因为连接两条规则，并不证明因果成立；"
                "实际展示已由grounding按facts重建，不能把raw当展示答案评分。"
            )
        elif rid.startswith("relabel-23eb"):
            verdict = "interpretation_overreach_requires_review" if no_previous else "supported_complete_visible_context"
            reason = (
                "无旧答案组把被动语态解释成被他人定义或忽视，偏离原先受助/互助的解释；"
                "该题完整E1至E6已读取。E1支持互相拯救，E2包含被斯凯忽视的妹妹，"
                "但不能直接把该叙述等同于娜斯提对被动语态的定义。此处保留解释扩展待复核，"
                "不把非逐字解读一律判为确定错误。"
                if no_previous else
                "展示答案维持平等互助而非单向受助的解释，与完整E1对話支持一致。"
            )
        else:
            assert rid.startswith("relabel-92de")
            verdict = "pipeline_error" if no_previous else "no_answer_identity_unfilled"
            reason = (
                "有旧答案组仍只有另一个熟知危机，无名称；无旧答案组使用action而非next_action导致管线失败，"
                "其raw正文也仍未给名称。原可见E3/E6/E9含矿石病关联。两组都未修复目标问题。"
            )
        notes.append({
            "id": rid, "question": row["question"], "variant": row["renderer"],
            "status": row["status"], "shown_answer": row.get("grounded", {}).get("answer"),
            "source_review_verdict": verdict, "reason": reason,
        })
    now = datetime.now(timezone.utc)
    notes_path = ANCHOR / "anchor_ablation_source_review_v1.json"
    save(notes_path, {
        "created_at_utc": now.isoformat(), "reviewer_type": "assistant_source_review_not_human_gold",
        "evaluation_only": True, "training_ready": False, "promotion": "not_promoted",
        "analysis_sha256": sha(ANCHOR / "anchor_ablation_validation_v1.json"), "notes": notes,
    })
    text = f"""# ASA：检索目标审核链修复与旧答案对照（v12）

UTC：{now.isoformat()}；北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()}。
文件名延续UTC 20260908系列。241为规范源，248为训练镜像。

## 结论

本次落地修复了数据构建器的诊断词传播与检索目标审核继承问题，276项规范回归通过。
没有训练、更新默认LoRA、重启服务、提交/推送或重写旧数据，也没有证明生产回答正确率提升。
另一组8次真实生成表明：单纯去掉历史答案未解决夜刀/行星碎屑的核心问题，此改动不晋级。

## 1. 构建器修复

规范改动仅：

- scripts/build_soda_api_verifier_dataset.py
- tests/test_soda_followup_integrity.py（新增42项）

原build_fallback_follow_up会将内部诊断先分词，例如长supported_fact错误码会被切成难以识别的碎片；
normalize_follow_up_payload及旧conclusion路径也没有与运行时统一清洗。
现在复用运行时clean_hypothesis_terms：先识别整条内部诊断，再分词；先清理，再分配词数预算。
保留正常名字、英文标识及用户明确询问的诊断词；quote/支撑等歧义词不作全局禁用。

旧relabel_group还可能：

1. 没有获认可的retrieve_more原候选时，从问题/缺口新拼一个请求，却沿用旧verifier正例身份；
2. 以宽松JSON修复后的输出当正例；
3. 清洗/裁剪/增补后续检索payload后，仍称它是原审核对象；
4. 把缺失的action_error或不明确的prior标志当作认可。

修复只接受明确认可的严格JSON候选。查询内容变化或无法确定无损的变换须重新审核；
未获新审核的目标不自动获得正例身份，也不改成负例。原始审核和负例输出保留。

允许一个可证明无损的旧格式投影：删除空answer、空clarification_question、空missing_slots，以及
与实际用户问题完全相同的顶层question。follow_up_hypothesis必须原样保持。
这种转换保留源记录ID、源response_text哈希和字段清单，不额外调用语义审核。
非空字段、未知字段、意图删除或关键词改变不借此放行。

首版候选采用更严格的对象全等，真实回放暴露其会挡住单纯空壳字段转换；
最终版补上上述窄范围无损投影，并增加12项测试。首版实验结果保留，没有改写成最终成绩。

## 2. 三个真实保存分片的只读回放

共552组，存盘数据字节和内存源对象前后不变。未调用API、检索或训练。

| 计数 | 修复前当前代码回放 | 最终修复回放 |
|---|---:|---:|
| 全部输出记录 | 866 | 835 |
| retrieve_more正例 | 185 | 154 |
| answer_directly正例 | 68 | 68 |
| 所有其他原有输出内容 | 原样 | 原样 |

最终31条检索正例待复核。其中16条涉及仍会改变的候选内容/结构，其余没有可直接保留的获认可严格候选。
首版曾暂扣60条，最终恢复其中29条无损旧格式投影；这29条的输出response和首版基线相同，只新增来源说明。
**待复核不等于31条都答错，恢复也不等于新获语义认证。**

45个获认可旧候选会删除顶层旧字段，12个还涉及follow_up_hypothesis.intent删除，
4个涉及既有关键词规范化；这些计数有交叉，详见followup_drop_explanations_v1.json。
本次三分片的原有/新清洗结果中，没有观测到实际诊断词变化（raw_followup_cleanup_count=0）。
所以诊断清理的缺陷由回归测试确认，不能冒称真实这批数据已有召回改善。

内存切分的共享问题重叠为0，仅说明本次机械切分，不是新训练集已合格。
68个直接正例、154个检索正例、未改动的其他任务/弃答样本仍需语义和协议审核。
没有运行build_dataset生成新训练目录，也没有修改此前存在KTO冲突的旧clean_v1数据。
另存followup_reaudit_proposals_v2.json为待审核方案，不带继承的kto标签，不是训练集。

## 3. 测试和安全

首轮新增30项：旧代码18失败/12通过；首版候选加原50项通过80项。
最终42项加原50项通过92项。落地后完整组合276/276通过（此前234+新增42）。
首次组合选漏test_evidence_id_grounding的23项和test_hybrid_chain_rendering的1项，
252项虽通过，但完整性guard拒绝认证；保留v1 XML，补齐后的v2才是276项结果。
备份复制曾因实验盘不支持保留权限而返回错误；后经cmp和SHA256确认原始字节完整，再继续。

目标代码哈希、其余规范源码、运行配置、git HEAD、6个来源数据文件均在应用前后核对。
248未同步本轮补丁，未占用其带进程的A100；未来在248构建/训练前必须先完成受控同步和数据审定。

## 4. 是否携带旧答案：8次真实生成

沿用4个开发问题，两组同题同显式seed、同完整原证据、同代码/LoRA和原JSON规则。
共同提示相同，唯一差别为previous_answer空字符串或旧展示答案；无gold、槽反馈和源文纠正。
2对无旧答案先跑、2对有旧答案先跑；GPU启动前连续60秒无计算进程。

| 机械结果 | 不带旧答案4条 | 带旧答案4条 |
|---|---:|---:|
| 直接回答 | 2 | 3 |
| 继续检索 | 1 | 1 |
| 管线失败 | 1 | 0 |
| 输出因length结束 | 0 | 0 |

机械计数不是正确率。独立源文检查：

- 美食节两组都未展示答案：一组quote不匹配，一组quote超过80字符。
- 行星碎屑展示答案相同，仍未补足所问机制。无旧答案raw回答把重力规则用“因为”接到伤害效果上，
  不能据一个连接词就确认因果正确；评分应看实际展示的grounded答案。
- 夜刀带旧答案仍是泛称；无旧答案使用错误action字段而失败，raw也没有危机名称。目标缺陷未修复。
- 娜斯提带旧答案保留有据的平等互助解释；无旧答案改为“被他人定义或忽视”。
  完整E1至E6已核对，E2有妹妹被忽视的叙述，但不能直接视作娜斯提对被动语态的定义。
  此处保留解释扩展待复核，不把一切非逐字解读判为确定错误。

没有为这8条再发一次已知会漏检的整体审核；源文意见明确不是人类gold。
不晋级，不报经过挑选的开发题正确率，不反写训练标签。

资产/输入检查：真实token数与预检一致，证据未截断；base与adapter文件前后SHA256一致，运行源码未变。
matrix.selection复制了上一实验的选择来源及旧seed字段；实际新seed在row.sampling_seed和
sampling_seed_override.json中，已按预注册的新公式逐请求核验。原冻结文件不改写，旁路说明歧义。
slot_feedback_preflight_v1.json只是复用启动器留下的文件名，在本独立目录内已核验对应新matrix哈希。

## 5. 后续

继续审查长引用长度规则造成的未回答：只做离线分组，先区分长度阻断和语义错误，不直接放宽线上校验。
不要把“去掉旧答案”或“加槽提示”当成已经解决生成质量的方案。
目标仍进行中。

构建器实验：`{HERE}`

生成实验：`{ANCHOR}`
"""
    with DEST.open("x", encoding="utf-8") as stream:
        stream.write(text)
    files = [
        HERE / "stored_followup_replay_v1.json", HERE / "stored_followup_replay_v2.json",
        HERE / "followup_drop_explanations_v1.json", HERE / "preapply_manifest.json",
        HERE / "applied_verified.json", HERE / "canonical_combined_tests_v2.xml",
        ANCHOR / "anchor_ablation_postflight_v1.json", notes_path,
    ]
    save(HERE / "checkpoint_v12_manifest.json", {
        "created_at_utc": now.isoformat(), "report_path": str(DEST), "report_sha256": sha(DEST),
        "training_ready": False, "goal_complete": False,
        "artifacts_sha256": {str(p): sha(p) for p in files},
    })
    print(json.dumps({"report": str(DEST), "sha256": sha(DEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
