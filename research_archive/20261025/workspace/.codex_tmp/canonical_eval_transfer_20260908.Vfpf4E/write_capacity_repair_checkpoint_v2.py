"""Write a new immutable checkpoint, keeping v1 and original labels."""
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
applied = read(run / "canonical_context_guard_patch_v1/applied_verified.json")
notes = read(run / "repair_source_review_notes_v2.json")
raw = read(run / "fixed79_configured_lora_ctx12288/rejected34.semantic_summary_retry1.json")["summary"]["after"]
context = read(run / "strict9_vllm_context_guard_ctx12288_stable_idle/semantic_summary9_v1.json")["summary"]["after"]
rebind = read(run / "fixed79_configured_lora_ctx12288/exact_rebinding79.counterfactual.json")
assert applied["applied"] and applied["tests_passed"] == 98
assert applied["expected_modified_frozen_files"] == 2
assert len(notes["notes"]) == 6 and raw["records"] == 34 and context["records"] == 9
assert len(rebind["newly_answered_record_ids"]) == 2
for change in applied["changes"]:
    assert sha(repo / change["path"]) == change["after_sha256"]
created = datetime.datetime.now(datetime.timezone.utc).isoformat()
report = f"""# ASA 规范检查点补充：容量保护与同证据修复

创建时间（UTC）：`{created}`。规范实验：`{run.name}`。
本文接续`CANONICAL_ANSWER_QUALITY_CHECKPOINT_20260908_irPugk.md`，不覆盖旧报告，也不表示目标完成。

## 已落地的范围

241仍是唯一代码与结果基准，248仍仅作训练镜像；本阶段未训练、未向248写入，未提交、未push、未重启服务。

本阶段正式工作区新增三文件改动：

1. `release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/model_runtime/vllm_runner.py`：
   生成前按实际tokenizer计数，要求完整输入与请求输出预算同时容纳；超额明确报错，不裁输入或缩减输出预算。
   记录请求预算、实际生成token、finish reason和stop reason。诊断目前保存在runner对象，并非所有流水线trace都已透传。
2. `release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json`：
   仅`generator.ctx_size`和`generator.vllm.max_model_len`由10000调整为12288；其他配置逐字不变。
3. `release/ASA-ArknightStoryAgent/tests/test_vllm_context_budget.py`：
   新增8项预算/边界/诊断测试；正式release全套从90/90变为98/98通过。

核验208份原冻结源/配置，仅上述runner与配置两份变化，其余206份保持原字节；新增测试另行核对。
HEAD仍为`{applied["head"]}`。没有覆盖既有dirty内容。
备份、精确patch及应用后核验在`canonical_context_guard_patch_v1/`，不需要也不应使用宽泛git reset回退。

## 真正验证了什么

空闲GPU持续60秒无计算进程后启动，实际4090、当前LoRA、vLLM 0.26.0、12288容量完成9条开发题：

- 每条实际输入token ID与完整提示逐个一致；预留2048输出token，9条均正常stop，无length结果。
- 最大输入8005 token。另79状态回放输入最大8077 token，即使加配置的3000输出预算也均能容纳12288。
- 这不证明任意未来提示都会容纳；超过容量仍明确失败。也不解决模型重复生成直至输出上限的问题。
- 此次生成经真实quote校验放行6/9；自动语义暂判完整4、部分1、不受支持1、未答3。
  旧10000容量vLLM开发题自动完整也是4/9。本次是容量完整性修复，**没有宣称语义正确率提高**。
- `VLLM_USE_FLASHINFER_SAMPLER=0`仍仅是实验进程设置，系统CUDA、包和线上服务未变。

## 34条被拦拟回答的全量诊断

34条均为原模型想回答、但quote校验没有放行的记录；不是用户实际收到的答案。
审核初始31条成功、3条失败；仅重试失败项后32条成功、2条仍因输出length失败。未丢分母、未自动视作错误或正确。

自动暂判：完整{raw["labels"].get("supported_complete", 0)}、部分{raw["labels"].get("supported_partial", 0)}、
不受支持{raw["labels"].get("unsupported", 0)}、审核失败{raw["labels"].get("error", 0)}。
34记录涉及27个规范化问题，仍有重复状态。

**13条“完整”也不是13条已确认正确。** 源文复核新增发现：

- 女妖题原答“寻找的人是拂哀菈，她正在寻找娜斯提”，第一分句混淆寻找者与被寻找者。
  judge把它默读成“女妖是拂哀菈”，才判完整；不能替实际答案改语法消除矛盾。
- 艾玛题称“她曾对担忧的居民说”，但“担忧的居民”就是艾玛自身的角色标签，
  原话是她睡梦中对尼亚尔的呼喊；judge只判部分，漏掉了说话对象错误。
- 炎景公主题把“陈家苛待公主”的恶名混成“联姻传闻”，judge又善意改读后判部分。
- 里昂题核心自我归因成立，但“马上就要完蛋”缩为“完蛋”有时态精度风险；记录争议，不强行判全错。

源文与答案片段均已逐字核对，见`repair_source_review_notes_v2.json`。
这些是代理源文复核，不冒充独立人工gold，也没有覆写原GLM结果。

## 唯一逐字quote重绑定：仅实验，不部署

保留完整79分母，只尝试修复被拦拟回答中长度20–80字、在恰好一个可见E块出现的quote编号。
不改fact、不改quote、不查隐藏正文；歧义或过短引用不修。

结果：3条有编号修正，2条新增放行：

- 芙蕾达题：E3改E6，核心对立有证据；重校验后不再展示原答中“背叛乌萨斯”和错误因为句，
  但最终文本仍是重复的事实拼接，对分歧原因解释不清楚。
- 巫女林题：E8改E12，组织性质和成员身份有证据；“为何来自那里”若要求加入缘由仍未充分解释。
  不能凭放行或quote真实统一判完整。

炎景公主题仍有未修短quote，继续被拦；没有为凑通过率放宽规则。
**这不是79正确率提高了2条，不加入正式链路。**

## 正在验证的下一项方案

`same_evidence_retry_paired68_v1.matrix.json`冻结全部34个机械拒答状态的两种重答：
普通同证据再生成 vs 简短quote复制反馈。每对使用相同显式采样seed，交替排列两种顺序。
选择依据仅是机械引用失败，不使用语义评分；提示不注入历史答案、gold、judge评论或额外证据。
对照组是vLLM额外生成，不能拿原Transformers答案代替，避免把多一次采样的收益冒充反馈收益。

完整68条及其审核尚未在本报告里给出结论。后续必须同时比较解释完整性、实质错误和未答比例，
尤其逐读新增放行项；反馈若只提高quote匹配却增加事实错误，不晋级。

## 当前正确率口径仍不变

此前79历史可见证据状态自动暂判19完整、12部分、9不受支持、37未答、2审核失败；
19中已有至少两项明确过宽判分。**不作为确认正确率，也不悄悄改成17/79当真实值。**
79记录是58个问题的多种历史状态，不是79道独立最终E2E问题；
“证据充分”judge数量不是RAG实际召回率。

规范结果目录：

```text
{run}
```

关键凭据：`canonical_context_guard_patch_v1/applied_verified.json`、
`canonical_release_tests_context_guard_v1.xml`、`repair_source_review_notes_v2.json`、
`fixed79_configured_lora_ctx12288/rejected34.semantic_summary_retry1.json`、
`strict9_vllm_context_guard_ctx12288_stable_idle/semantic_summary9_v1.json`。
"""
paths = [
    run / "CAPACITY_REPAIR_CHECKPOINT_v2.md",
    repo / "docs/CANONICAL_CAPACITY_REPAIR_CHECKPOINT_20260908_irPugk_v2.md",
]
for path in paths:
    with path.open("x") as output:
        output.write(report)
assert sha(paths[0]) == sha(paths[1])
print(json.dumps({"canonical_doc": str(paths[1]), "report_sha256": sha(paths[0]), "goal_complete": False}))
