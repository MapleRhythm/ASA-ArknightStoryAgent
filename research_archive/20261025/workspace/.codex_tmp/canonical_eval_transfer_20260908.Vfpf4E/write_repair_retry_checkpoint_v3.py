"""Persist completed repairs and negative experiments without changing gold."""
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
applied = read(run / "canonical_quote_integrity_patch_v1/applied_verified.json")
pair_folder = run / "same_evidence_retry_paired68_ctx12288_v1"
pair = read(pair_folder / "semantic_summary68_retry1.json")
base = pair["summary"]["retry_control_v1"]
feedback = pair["summary"]["retry_quote_feedback_v1"]
assert base["records"] == feedback["records"] == 34
assert applied["test_count"] == 106
for change in applied["cumulative_changed_files_this_phase"]:
    assert sha(repo / change["path"]) == change["after_sha256"]
old_rows = {r["id"]: r for r in read(pair_folder / "validation_after_v2.json")["results"]}
new_rows = read(pair_folder / "validation_quote_integrity_v1.json")["results"]
assert len(old_rows) == len(new_rows) == 68
assert all(old_rows[r["id"]]["grounded"]["answer"] == r["grounded"]["answer"] for r in new_rows)
native = read(run / "strict9_native_exx_ctx12288_v1/validation_native_exx_v1.json")
native_summary = native["summary"]["native_exx_same_evidence_v1"]
assert native_summary["rows"] == 9 and native_summary["parse_or_validation_errors"] == 8
synthetic = read(run / "inferred_fact_boundary_synthetic_v1.json")
assert [r["unsupported_inferred_assertion_exposed"] for r in synthetic["results"]] == [False, True, True]
time = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows = []
for title, summary in (("普通同证据重试", base), ("引用复制反馈重答", feedback)):
    labels = summary["labels"]
    rows.append(
        f'| {title} | {summary["records"]} | {summary["grounded_actions"].get("answer_directly", 0)} | '
        f'{labels.get("supported_complete", 0)} | {labels.get("supported_partial", 0)} | '
        f'{labels.get("unsupported", 0)} | {labels.get("no_answer", 0)} | {labels.get("error", 0)} |'
    )
report = f"""# ASA 修复与同证据重答检查点 v3

时间（UTC）：`{time}`。241为规范源；本报告接续v1、v2，不覆盖它们，不表示长期目标完成。

## 本阶段已经落地

工作区中累计5个文件变动，HEAD仍为`{applied["head"]}`，未提交、未push、未重启服务、未训练：

1. VllmRunner精确检查完整输入与请求输出预算，记录实际token及finish reason。
2. GPU quote配置的ctx_size/max_model_len从10000增加至经真实4090验证的12288。
3. 8项token预算测试。
4. `payload/utils.py`不再将模型输出的quote静默切成80字；完整保留给校验器执行既有长度限制。
5. 8项quote原文完整性测试。

正式release **106/106测试通过**。208份冻结源/配置中仅3份发生上述预期变化，其余205份未变；另2份为新增测试。
完整备份、patch和实际应用后哈希核对分别位于`canonical_context_guard_patch_v1/`与`canonical_quote_integrity_patch_v1/`。

## 取消quote静默裁切的实际影响

同一批79条模型原始输出重新解析/校验，不重新生成：

- 原有41条放行变为40条；只有玛露西尔题由回答变为继续检索。
- 该题原文表示莱欧斯更着急、齐尔查克想缩短路程，答案却推成“因此不急着赶路”，经源文复核属于错误。
- 模型该段quote实际为101字，旧解析器切成80字再校验，绕过了超长检查。
- 香草题另有128字quote，原先已被拒绝，现在额外记录超长问题，答案文本不变。
- 其他78条的答案文本不变。strict9的9条grounded结果全部不变。
- 这证明挡住了一条已复核错答，**不等于多答对一题，也不是新的独立正确率评测**。
- 合法但过长的quote同样会被拒绝，后续仍需重生成符合约定的引文，不能把格式错误都当证据不足。

## 配对68条重答已完整结束：不采用为默认

原79状态中的全部34条“拟回答但引用校验拒绝”作为开发队列，27个规范化问题、有重复状态。
每个状态各做普通vLLM重试、附简短引用反馈重答；每对显式使用同一sampling seed，顺序交替。
两组证据、问题、hypothesis、轮次、模型和生成预算相同，不发送历史答案、gold或judge意见。

全部68条生成正常stop，输入token逐一匹配；语义审核66条初次成功、2条失败，仅重试失败后67成功、1仍schema失败。
所有记录保留，错误没有从分母剔除。以下是自动暂判，仍非独立人工gold：

| 配置 | 状态数 | 引用放行 | 完整 | 部分 | 不受支持 | 未答 | 审核失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

自动完整从5增至7，但不受支持回答从1增至5。不能以新增10条引用放行宣称成功：

- 女妖题反馈组明确说“寻找的人是拂哀菈”，实际被寻找者是娜斯提，仍是主体/客体混淆。
- 道尔顿题反馈组把凶案归于格蕾塔，证据指向卢西恩；只是引用复制更规范，并未纠正凶手身份。

因此不把该反馈加入正式重答链。保留所有失败实验与审核原文。
用已经落地的quote原文保留逻辑再校验这68条，全部最终答案文本与实验旧解析器一致，仅4条已拒答的诊断详情变化；
上述新增错答风险并没有被单纯取消裁切自动解决。

## 已有Exx编号协议的兼容性试验也未通过

同一strict9、相同完整可见证据，调用代码已有的Exx模板，以事实+evidence_ids代替quote。
使用原有E-ID的512输出预算与空assistant prefix，未修改默认quote协议。

9条中8条解析失败，另1条解析后被E-ID校验拒绝；没有可展示回答。
多数原始输出把supported_facts写成字符串数组（有的只在字符串末尾写E编号），而规范解析器要求事实对象。
其中6条本身是合法JSON，也不能因此冒称协议正确；1条输出达到512token上限。
现有Exx提示只文字描述字段，没有给实际fact/evidence_ids对象示例。
**不把任意字符串列表猜测成规范引用，也不据此切换默认协议。**

后续隔离对照只追加明确JSON对象结构，保持其余条件相同，用来区分提示格式不清与模型事实能力问题。
本报告尚不包含该后续对照结论。

## 新复现的未解决漏洞：推断引用丢失并被当作确定事实

合成探针只提供“角色甲读书、角色乙散步”：
给inferred_facts塞入“角色乙已经投降”，无论不写引用还是写不存在的E99，
现有解析器都会丢弃推断的引用，校验器不逐条校验推断，随后能输出：
“根据当前证据可确认：角色甲在窗边读书。；角色乙已经投降。。”

这是实际源码复现，不是模型judge意见；见`inferred_fact_boundary_synthetic_v1.json`。
正在隔离候选中保留推断引用、复用引用校验，并明确区分事实与推断；**截至本报告尚未部署**。
引用真实仍不证明推断逻辑成立，候选也不能替代语义审核。

## 原79正确率口径仍须谨慎

原始自动19/79完整存在至少两项明确过宽判分，不等于真实正确率。
79记录是58个问题的历史证据状态，不能冒充79道独立最终E2E问题；“证据充分”的judge判数也不是RAG实际召回率。
本阶段实证支持的是两项完整性修复、一个已复核错答被拦，以及两个尚不能晋级的生成方案。

规范实验目录：

```text
{run}
```
"""
paths = [
    run / "REPAIR_RETRY_CHECKPOINT_v3.md",
    repo / "docs/CANONICAL_REPAIR_RETRY_CHECKPOINT_20260908_irPugk_v3.md",
]
for path in paths:
    with path.open("x") as output:
        output.write(report)
assert sha(paths[0]) == sha(paths[1])
with (run / "repair_retry_checkpoint_v3_artifacts.json").open("x") as output:
    output.write(json.dumps({
        "report_sha256": sha(paths[0]), "pair_summary_sha256": sha(pair_folder / "semantic_summary68_retry1.json"),
        "applied_state_sha256": sha(run / "canonical_quote_integrity_patch_v1/applied_verified.json"),
        "native_analysis_sha256": sha(run / "strict9_native_exx_ctx12288_v1/validation_native_exx_v1.json"),
        "inferred_probe_sha256": sha(run / "inferred_fact_boundary_synthetic_v1.json"),
        "training_ready": False,
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"canonical_report": str(paths[1]), "report_sha256": sha(paths[0]), "goal_complete": False}))
