"""Record failed development feedback experiment and independent source review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
DEST = ROOT / "docs/CANONICAL_ANSWER_SLOT_AND_PAIRED_GENERATION_20260908_irPugk_v11.md"


def load(name):
    return json.loads((HERE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not DEST.exists()
    coverage = load("answer_slot_coverage_direct39_v1.json")
    paired = load("slot_feedback_comparison_v1.json")
    post = load("slot_feedback_postflight_v1.json")
    generated = load("slot_feedback_validation_v1.json")
    assert coverage["states"] == 39 and paired["states"] == 8
    assert post["base_and_adapter_assets_unchanged"] and post["source_unchanged"]
    review = []
    for row in generated["results"]:
        rid = row["id"]
        feedback = row["renderer"] == "slot_retry_feedback_v1"
        if rid.startswith("896419"):
            verdict = "supported_complete_visible_context" if feedback else "no_answer"
            reason = (
                "展示答案命名美食节，说明真理准备、开心与凛冬解谜失败后的惩罚；E2/E4/E6及相关完整E块支持。"
                "此结论按全可见证据，不等于每条短quote都覆盖其fact全部内容。"
                if feedback else
                "引用超过80字符导致规范grounding改为retrieve_more，展示答案为空；不是本次已答对。"
            )
            scope = "相关完整E块；并非逐块阅读所有可见源文"
        elif rid.startswith("b5c3"):
            verdict = "not_complete_mechanism_unresolved"
            reason = (
                "两组都复述伤害/推动效果。反馈增加足够推力可推动碎屑；控制增加轨道/重力规则。"
                "没有把移动与对敌伤害之间的解释关系表达出来，不算完整解决所问机制。"
                "整体审核均判partial，但其理由要求额外充能/我方伤害等背景可能过度扩展问题，"
                "其中碰撞因果叙述也不得不经复核就当原文事实。"
            )
            scope = "已完整检查关键E2/E3/E4/E6/E7；未声称读完长E1/E5"
        elif rid.startswith("relabel-23eb"):
            verdict = "supported_complete_visible_context"
            reason = (
                "两组展示答案完全相同。E1斯凯自称被帮助者，娜斯提强调互相原谅/拯救/庇护，"
                "支持平等互助的解释；保持解释型正向对照，不因非逐字复述而判错。"
            )
            scope = "完整E1及已记录相关上下文"
        else:
            assert rid.startswith("relabel-92de")
            verdict = "no_answer_identity_unfilled" if feedback else "pipeline_error"
            reason = (
                "反馈仍只有另一种熟知危机，未命名。自动审核再次假阳性，并称原文没有具体名称；"
                "完整E3含不论矿石病的因素，E6含可能会传播矿石病的威胁，E9也有矿石病与火龙相关假设。"
                "控制误包answer_directly顶层字段，管线失败，不能从未展示的raw正文补出一个已答对结果。"
            )
            scope = "本次已重新完整读取E1至E12"
        review.append({
            "id": rid, "question": row["question"], "review_verdict": verdict,
            "reason": reason, "source_review_scope": scope,
        })
    now = datetime.now(timezone.utc)
    notes = {
        "created_at_utc": now.isoformat(), "evaluation_only": True, "training_ready": False,
        "reviewer_type": "assistant_source_review_not_human_gold",
        "automatic_labels_overwritten": False, "notes": review,
        "source_analysis_sha256": sha(HERE / "slot_feedback_validation_v1.json"),
    }
    notes_path = HERE / "slot_feedback_source_review_v1.json"
    write_json(notes_path, notes)
    text = f"""# ASA：回答槽覆盖与真实配对生成（v11）

生成时间 UTC：{now.isoformat()}；北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()}。
文件名延续 UTC 20260908 系列；规范源为241，248为训练镜像。

## 结论

**本次反馈方案不晋级，没有证明回答正确率提高。**
8次新生成（4个开发问题×2组）已完成，模型资产、源码、配对seed和无截断检查均通过。
反馈组更容易通过机械验证，但核心夜刀题仍未回答危机名称；整体GLM再次误判为完整。
这些问题有选择偏差，不是独立holdout；不能把4/4直接回答或3/4自动完整当成正确率。

## 1. 回答槽检查

此检查只输入问题和展示答案，没有证据、gold或源文纠正，目的是防止审核器替答案补全所问信息。
逐事实有依据与实际答中问题是两件事；覆盖通过也不代表内容真实。

先检查7个离线过滤候选：6成功/1格式失败；4 complete、2 none、1 error。
夜刀泛称被识别为none；卡铎尔弃答审核漏字段，保留为error，未擅自补字段。

随后使用同一rubric检查全部39个历史直接回答状态（仍不是新生成）：

- 38个去重question+answer输入，其中1个完全相同成功请求复用，37个新API请求。
- 32成功、7审核格式失败；30 complete、2 none、7 error，全部保留分母。
- 两个有效none是夜刀（缺危机名称）和行星碎屑（复述效果和移动规则，未给机制）。
- 7个error涉及漏answer_spans、非逐字引号或多字段，不等于这7个答案事实错误。
- 军官禁令题的失败原始审核还显露误杀风险：答案“公爵下令不许讨论”可构成直接原因，
  审核却要求禁令背后的深层动机。没有把这种扩展要求安装成生产gate。

## 2. 8次真正新生成

选入上面的2个有效none和美食节、娜斯提2个正向控制。
两组均看到同样旧展示答案与完整原证据；反馈组额外看到只由问题/旧答案推导出的槽检查。
没有把矿石病等源文纠正、标准答案注入反馈。夜刀反馈内容有专门的防泄漏断言。

使用当前规范代码、原默认LoRA、同题同显式seed，temperature=0.1、top_p=0.8、max_new_tokens=2048、
max_context=12288。四对恰巧都是反馈先运行，不声称顺序严格平衡。
旧harness冻结配置的上下文为10000；实验只明确覆盖ctx_size/max_model_len为当前12288，其余generator配置逐字段一致。
第一次预检因配置哈希差异拦截，未启动GPU；未改生产配置或旧harness以隐去差异。

| 指标 | 普通重试4条 | 槽反馈4条 |
|---|---:|---:|
| 机械管线成功 | 3 | 4 |
| 直接回答 | 2 | 4 |
| 继续检索 | 1 | 0 |
| 管线失败 | 1 | 0 |
| 自动完整支持 | 1 | 3 |
| 自动部分支持 | 1 | 1 |
| 自动未回答 | 1 | 0 |
| 因管线失败无法审核 | 1 | 0 |

反馈组行星碎屑的原始输出不符合严格JSON，但被现有生产解析器接受；未改解析器来提高通过率。
所有输入真实token计数与预检一致，没有证据截断，输出无length结束。
审核7个有效输出，另1个pipeline error保留分母，没有重复发无效请求。

## 3. 独立源文复核

- 美食节：反馈展示答案得到完整可见上下文支持；控制因quote超过80字符而转继续检索，不能算答对。
  全上下文支持不等于每个fact自己的短quote都充分覆盖其所有断言。
- 娜斯提：两组展示答案相同，平等互助解释有E1支持，未见此正向控制退化。
- 行星碎屑：两组均未完整解释所问机制。反馈增加推力规则，控制增加轨道/重力规则；
  不能把并列的有据规则当成完整因果回答。只审了相关完整E块，未声称读完长E1/E5。
  GLM两组均判partial，但其额外背景要求与推导出的碰撞描述不是经人类认可的gold。
- 夜刀：反馈仍只写“另一个为泰拉所熟知的危机”，危机身份槽未填。
  本次重新读完该题E1至E12：E3、E6及E9含矿石病关联。
  自动审核称“未给更具体名称”并判完整，与可见源文和问题要求不符。
  控制发生顶层answer_directly字段错误，不能把raw中的未展示内容当有效答案。

源文意见记录为assistant_source_review_not_human_gold；自动输出不覆盖，不反写训练标签。
不拿这4题构造经过事后挑选的“真实正确率”。

## 4. 资产与安全

GPU为241上经连续60秒无计算进程检查后的单张4090。实验进程已正常退出，未停止其他任务。
完整base/adapter文件清单在实验前后逐项SHA256一致；规范源码前后未变。
metadata中的seed旧描述由sampling_seed_override.json补充，实际每条请求的seed和prompt哈希已对账。
248的A100有进程，即便利用率为0也未占用。
没有训练、切换默认模型、提交/推送、重启服务或重写旧数据。

## 5. 后续

反馈失败结果保留，不继续把这4题调到高分就声称泛化。
可以预注册不携带旧答案的配对生成，以分离旧输出锚定和读题能力；
同时继续修复构建器诊断词传播、审核标签继承问题。相关新工作不属于此已完成实验。
目标仍进行中。

实验目录：`{HERE}`。
关键文件：answer_slot_coverage_direct39_v1.json、slot_feedback_validation_v1.json、
slot_feedback_content_audit_v1.json、slot_feedback_comparison_v1.json、
slot_feedback_source_review_v1.json、slot_feedback_postflight_v1.json。
"""
    with DEST.open("x", encoding="utf-8") as stream:
        stream.write(text)
    files = [
        HERE / name for name in (
            "answer_slot_coverage7_v1.json", "answer_slot_coverage_direct39_v1.json",
            "slot_feedback_comparison_v1.json", "slot_feedback_source_review_v1.json",
            "slot_feedback_postflight_v1.json", "slot_feedback_content_audit_v1.json",
        )
    ]
    write_json(HERE / "checkpoint_v11_manifest.json", {
        "created_at_utc": now.isoformat(), "report_path": str(DEST), "report_sha256": sha(DEST),
        "training_ready": False, "goal_complete": False,
        "artifacts_sha256": {str(p): sha(p) for p in files},
    })
    print(json.dumps({"report": str(DEST), "sha256": sha(DEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
