"""Publish a canonical checkpoint only after the paired diagnostic completes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PATCH = RUN / "cleaner_integrity_patch_v1.LTD6na"
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
DESTINATION = ROOT / "docs/CANONICAL_SEMANTIC_AUDIT_AND_CLEANER_REPAIR_20260908_irPugk_v10.md"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def main():
    combined = load(HERE / "semantic79_combined_v1.json")
    notes = load(HERE / "source_review_notes_v3.json")
    claim = load(HERE / "cited_claim_probe13_v2.json")
    paired = load(HERE / "semantic_filter7_comparison_v1.json")
    filtered_notes = load(HERE / "semantic_filter7_source_notes_v1.json")
    replay = load(PATCH / "stored_cleaner_replay_v1.json")
    applied = load(PATCH / "applied_verified.json")
    assert combined["states"] == 79 and paired["states"] == 7
    assert notes["reviewed_states"] == 8
    assert len(filtered_notes["notes"]) == 7
    assert applied["canonical_tests"]["tests"] == 234
    assert all(sha(ROOT / path) == digest for path, digest in applied["applied_sha256"].items())
    now = datetime.now(timezone.utc)
    files = [
        HERE / "preflight_manifest.json", HERE / "semantic79_content_v2.json",
        HERE / "semantic79_retry1.json", HERE / "semantic79_combined_v1.json",
        HERE / "source_review_notes_v3.json", HERE / "cited_claim_probe13_v2.json",
        HERE / "semantic_filter_plan_v1.json", HERE / "semantic_filter7_comparison_v1.json",
        HERE / "semantic_filter7_source_notes_v1.json",
        PATCH / "stored_cleaner_replay_v1.json", PATCH / "preapply_manifest.json",
        PATCH / "applied_verified.json",
    ]
    paired_table = "\n".join(
        f"| {p['question']} | {p['baseline_auto_verdict']} | {p['candidate_auto_verdict']} | {p['kept_fact_count']} |"
        for p in paired["pairs"]
    )
    review_table = "\n".join(
        f"| {n['question']} | {n['auto_verdict_at_review']} | {n['review_verdict']} |"
        for n in notes["notes"]
    )
    provenance = "\n".join(f"- `{path.relative_to(RUN)}`: `{sha(path)}`" for path in files)
    text = f"""# ASA：完整79状态内容审核、逐事实试验与旧清洗路径修复（v10）

生成时间（UTC）：{now.isoformat()}

北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()}。文件名日期按 UTC 延续既有系列。

规范源：241 `${REMOTE_HOME}/ASA-ArknightStoryAgent`；248 是训练镜像，本轮未修改。实验根目录：`{RUN}`。

## 结论

这轮确认并修复的是训练数据清洗路径的证据链与切分问题，**没有证明生产回答正确率已提升**。
完整79状态自动审核已完成一次，随后仅重试3个审核格式失败项；合并后仍有1项失败，保留在分母。
“原文有引用”“审核格式通过”和“真正回答正确”是三个不同条件。新审核口径仍有明确假阳性，不能把其完整回答数当成真实正确数。

逐事实审核对乔迪的代际混淆、威尔的立场颠倒有帮助，但也有引用格式失败、错误拒绝理由和无法判断答案是否答中问题的边界。
逐事实过滤只作为7个已知开发状态上的离线诊断，未上线、未写训练标签、未更换模型。

## 1. 完整79状态的范围和结果

79是58个去重问题的历史可见证据状态，不是79道独立的新端到端题。当前241源码重放结果为39直接回答、31继续检索、9弃答。
本轮未重新检索、未重新生成这79条答案。原始预测、证据、版本哈希均保留。

审核模型请求/返回名为 `glm-5.3`，协议 `asa_visible_answer_semantics_v2`。首轮76成功/3格式失败；只重试失败3条后2成功/1失败，总计82次请求。
失败不是被移出分母，也不是被擅自改成模型认为的答案标签。

| 自动审核标签 | 全部79状态 | 39个直接回答状态 |
|---|---:|---:|
| 完整支持 | {combined["all79_verdict_counts"].get("supported_complete", 0)} | {combined["direct39_verdict_counts"].get("supported_complete", 0)} |
| 部分支持 | {combined["all79_verdict_counts"].get("supported_partial", 0)} | {combined["direct39_verdict_counts"].get("supported_partial", 0)} |
| 不支持 | {combined["all79_verdict_counts"].get("unsupported", 0)} | {combined["direct39_verdict_counts"].get("unsupported", 0)} |
| 未回答 | {combined["all79_verdict_counts"].get("no_answer", 0)} | {combined["direct39_verdict_counts"].get("no_answer", 0)} |
| 审核失败 | {combined["all79_verdict_counts"].get("audit_error", 0)} | {combined["direct39_verdict_counts"].get("audit_error", 0)} |

这些只是模型评判数。不能宣称“19/79是真实正确率”，也不能机械扣除发现的几例就声称得到真实分数。
证据充分性模型标记为 `{pretty(combined["context_sufficiency_counts"])}`。该指标不是RAG召回率，没有检索全集的gold分母。

仍失败的梅希亚条目：审核同时给出 `supported_partial` 和非空 `unsupported_spans`，违反同一协议；原始输出和失败原因保留，没有通过删字段来“修好”标签。

## 2. 源文复核与自动审核的分歧

下面8个状态均有单独复核记录。前三例延续此前的详细源文检查；新增例只读取相关完整E块和展示答案，不声称逐块审完该题的所有可见证据。
这是助手的源文复核，不是人类gold。其余自动完整样本仍需继续逐源检查。

| 问题 | 自动首轮结论 | 源文复核 |
|---|---|---|
{review_table}

关键事实：

- 乔迪：E8明确更正“父母，严格来说，是祖父母”，答案却把出身赋给父母；维修师/为灯塔牺牲这一条有E9/E10独立支持，不能一并判错。
- 莫斯提马：“苦难陈述者”是对话另一方的职务。无需借剧情记忆为匿名人物补姓名即可判定主语错误。
- 夜刀：只说“另一个熟知的危机”，没有回答“什么危机”；E3/E6含矿石病因素。原句有出处也不等于填上被问的身份。
- 诺希斯：压缩后的句子可能把“因弱小而获施舍”读成“因施舍而弱小”；保留 `uncertain`，不把中等确信的语义歧义强行当确定错误。
- 卡铎尔：源句末尾“留在拳馆”限制“拒绝……撤离”的范围，答案丢掉这部分并混用不同时期的事件；不能用词面重合认证。
- 威尔：他因汤姆辱骂芙蕾达而气愤，并为芙蕾达辩护；答案改成他指责芙蕾达导致自己被扣留。
- 拉普兰德：第一句被除名原因有支持；新增“试图挽回家族关系”无支持，且E1反问否定乞求重新加入。不能把嘲讽的忠诚表态直接当动机。
- 娜斯提：问题要求解释暗示，“互相帮助、不是单向受助”的解读有完整对话支持，保留为正向解释型对照；不是所有转述都应被否决。

自动标签、旧报告和原展示答案均未覆盖。

## 3. 13条事实及7状态离线过滤试验

从7个开发状态取出全部13条事实，包括同题正确事实和娜斯提解释型对照；只发送问题、原样claim、其引用的完整可见E块。
没有发送旧judge、gold、复核意见或隐藏剧情。结果：

```json
{pretty({"statuses": claim["statuses"], "relations": claim["relations"]})}
```

这不是验证器准确率。乔迪和威尔的错误被识别；诺希斯被拒绝的理由主要是其引用块的归属/缺证，而不是充分解决全部因果歧义。
卡铎尔虽然被拒绝，但审核理由仍误读部分“拒绝/留下”关系，因此“挡住了”不能等同于“正确理解了”。
莫斯提马坏事实的原始审核语义上指出了人物混淆，但引文漏空格，未满足严格逐字协议，故仍计审核失败；未擅自修引文后冒充原输出合格。

过滤规则在查看完成的逐事实结果前冻结：只保留 `ok+entailed` 的原事实和原引用，使用现有规范渲染函数；某条审核失败或无事实可保留时，该诊断状态弃答。
不新编事实，不自动补答案，不把旧答案作为无事实时的回退。候选对象保留原始生成trace作为来源，但顶层明确 `new_generation=false`，不可当成一次新模型生成。
过滤后的展示答案再次使用同口径、同完整证据审核，检查是否仍答中问题。

| 问题 | 原答案自动结论 | 过滤后自动结论 | 保留事实数 |
|---|---|---|---:|
{paired_table}

原答案模型标签统计：`{pretty(paired["baseline_auto_counts"])}`

过滤后模型标签统计：`{pretty(paired["candidate_auto_counts"])}`

动作统计：`{pretty(paired["candidate_actions"])}`

候选源文复核记录另外保存，没有改自动标签：乔迪现在只保留有据的维修/牺牲关系；诺希斯只保留有据的舞台承诺；威尔题不再颠倒其立场，保留引发分歧的合作行为及保护动机；娜斯提解释型对照原样保留。
夜刀仍是未答中“什么危机”，而新一轮整体审核仍判完整，甚至错误声称证据没有给出危机名称。自动“5/7完整”因此仍不可当真实正确率。
两个弃答不算正确答案，也不能靠只减少答题量宣称问题已经解决。

**不晋级。** 基线含已知假阳性，小样本又是选择过的开发案例，自动分数变化不能表示生产正确率变化。
夜刀的两条泛称事实均通过事实审核，却仍未命名危机；必须另查“实际回答了哪个子问”。
此外，整条fact包含多个断言时，删掉坏fact也会丢掉其中正确部分；审核格式失败也会导致保守弃答。
下一步应先做无标签提示的“所问信息槽/答案具体填充值”核对，并用独立问题验证误杀率，不应直接部署过滤器或在这些案例上继续调到高分后宣称泛化。

## 4. 已落地：旧清洗路径修复

改动仅为：

- `scripts/clean_soda_api_verifier_dataset.py`
- `tests/test_soda_cleaner_integrity.py`

旧脚本的目标是修正弱接受样本，但实现会按4个固定prompt ID改写答案，并同步覆写verifier的支持答案、先验知识和错误字段；没有产生新的审核，因此不能继承“已认可”身份。
这不等于已证明4段替换文案全错，也不等于能把历史缺陷全部归因于2026-09-06的某个模型。

修复：

1. 移除硬编码替换文案和审核字段覆写；将历史争议/`clean_v1`标记作为待复核线索。
2. 按共享问题分组，把关联状态和正负对一起隔离；原记录完整保留在隔离输出，隔离不是新负标签。
3. 复用共享问题级切分和同输入同输出的正负冲突校验；不再按各prompt分开切分。
4. 缺输入、无ID、冲突重复ID、非法比例、覆盖已有输出、输出指向输入等明确失败。
5. 输出保留原始审核文件字节和哈希，明确 `training_ready=false`。保留集未获语义或训练协议的重新审定。

新增20测试：旧实现19失败/1通过，修复20通过。规范合并测试234/234通过。
首次合并测试命令漏设release/src的PYTHONPATH，产生10个收集错误；该XML保留为v1，修正调用路径后的v2才是234通过结果，没有把调用错误当作已修代码回归。

真实数据仅内存回放，两目录各208记录：

| 来源 | 存盘train/val重叠问题 | 旧清洗回放 | 新清洗回放 |
|---|---:|---|---|
| merged | 7 | 改写4条答案及verifier；切分重叠10题 | 隔离4题21条，保留187条，但仍存在KTO正负冲突，拒绝生成训练集 |
| clean_v1 | 10 | 4条历史改写已在盘上；重跑切分重叠8题 | 同样隔离21条，保留187条，但同样因KTO冲突拒绝 |

不能把这个真实回放宣称为“新训练集已无泄漏并可训练”：它根本没有通过剩余标签冲突检查。
源文件哈希在前后核对一致；本轮未重写这两套数据，未启动训练，未改权重，未提交/推送，未重启服务。运行时代码未因本次清洗补丁变化。

## 5. 下一阶段仍需做

- 审完其余自动完整答案，并保留“审核模型意见”和“源文复核”的独立来源，不给虚假真正确率。
- 明确区分逐fact有据与逐子问答全。夜刀说明两者不能合并成一个分数。
- 历史保留集仍有同输入同输出正负冲突；查源审核再决定隔离/再标注，不能随意选一个标签。
- 构建器独立的 `build_fallback_follow_up` / `normalize_follow_up_payload` 仍会传播旧诊断词，需要按运行时同口径处理，但不能改目标后沿用旧审核训练。
- 未评测的KTO候选、未审定正例不晋级。提升目标仍在进行，本报告不是完成声明。

## 证据与哈希

以下路径相对实验根目录：

{provenance}
"""
    with DESTINATION.open("x", encoding="utf-8") as stream:
        stream.write(text)
    manifest = {
        "created_at_utc": now.isoformat(), "report_path": str(DESTINATION),
        "report_sha256": sha(DESTINATION), "training_ready": False, "goal_complete": False,
        "artifacts_sha256": {str(p): sha(p) for p in files},
    }
    with (HERE / "checkpoint_v10_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(pretty({"report_path": str(DESTINATION), "report_sha256": manifest["report_sha256"]}))


if __name__ == "__main__":
    main()
