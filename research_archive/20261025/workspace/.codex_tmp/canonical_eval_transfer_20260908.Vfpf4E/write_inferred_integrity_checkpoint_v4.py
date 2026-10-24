"""Save the applied inference fix and completed negative Exx experiment."""
import datetime
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
applied_path = run / "canonical_inferred_integrity_patch_v2/applied_verified.json"
applied = load(applied_path)
assert applied["test_count"] == 115 and applied["applied"]
for item in applied["effective_changes"]:
    assert sha(repo / item["path"]) == item["after_sha256"]
comparison_path = run / "inferred_integrity_v2_comparison.json"
comparison = load(comparison_path)
assert comparison["v1_cross_pool_regression_reproduced"]
exx_folder = run / "strict9_native_exx_explicit_schema_ctx12288_v1"
analysis_path = exx_folder / "validation_explicit_schema_v1.json"
analysis = load(analysis_path)
predictions = load(exx_folder / "predictions.json")
semantic = load(exx_folder / "semantic9_v1.json")
assert len(analysis["results"]) == len(predictions) == len(semantic["results"]) == 9
assert all(r["grounded"]["next_action"] == "retrieve_more" for r in analysis["results"])
assert all(r["status"] == "ok" and r["judgement"]["verdict"] == "no_answer" for r in semantic["results"])
assert sum(r["normalized_strict_json"] for r in analysis["results"]) == 0
stopped = [r for r in predictions if r["finish_reason"] == "stop"]
assert len(stopped) == 8 and all(r["raw_output"] == r["raw_output_with_special_tokens"] for r in predictions)
assert all(r["raw_output"].endswith("}]") for r in stopped)
report = f"""# ASA 推断引用完整性与生成诊断检查点 v4

记录时间（UTC）：`{datetime.datetime.now(datetime.timezone.utc).isoformat()}`。
241为代码、数据、评测规范源；248仅训练评测镜像。本报告接续v3，不覆盖历史结果，不表示长期正确率目标完成。

## 本阶段已落地：推断引用不能绕过校验，也不能冒充直接事实

241 release已更新4个源码文件，另新增9项推断引用测试：

- `payload/utils.py`：推断也保留规范化后的引用，不再无条件丢弃。
- `grounding/quote_validation.py`：推断引用按相同可见E块、quote匹配和长度规则检查，总引文字数计入推断。
- `grounding/quote_checks.py`：引用不合法时续检索或最后轮弃答；通过时仍区分事实与推断。
- `grounding/grounded_fact_answers.py`：推断明确标为“推断（非直接事实）”，不拼入“可确认”的直接事实列表。

正式release **115/115测试通过**。本阶段累计3组修复共9个不同文件；208份冻结源/配置中只有6份预期变化，另3份为新增测试。
逐阶段备份、patch及同一路径最后一次修改的哈希链均已核验；HEAD仍为`{applied["head"]}`。
未提交、未push、未重启服务、未开始训练；运行中的服务是否加载新文件不能由工作区测试代替确认。

### 在上线前抓到候选v1自身的负面改动

v1曾合并直接事实与推断的quote池。合成反例中，直接事实“甲是乙的母亲”只引用“甲与乙碰面”，
却借用了推断引用“丙是丁的母亲”中的关系词而获得放行。
新增回归测试对v1真实失败（错误放行），修正v2分离两类quote池后通过。
**v1没有部署到正式源码；上线的是修正后的v2。**

引用文本真实仍不等于语义蕴含成立；同组不同事实借用关系词、旧compactor限制引用数量等仍是单独待查问题。
不能把此修复宣传为完整事实验证器。

## 三组逐条回放结果

同一批原始输出重新解析校验，没有重新生成，也没有重新检索：

- 原79状态：40回答变为39回答，31续检索、9弃答。唯一答案/动作变化为老天师题。
- 老天师原模型确实写了推断引用；旧parser丢掉了它。保留后发现`inferred_fact_1_ref_2_quote_not_found`，
  不再展示“沉默樵夫和诡异织物都是徒弟或相关存在”的未经支持推断。
- 其他78条答案文本不变；大帝题原本已弃答，只增加推断缺引用诊断。
- strict9：9条grounded结构全部不变（6回答、3续检索）。
- 配对68：68条动作与最终答案全部不变；只有原已弃答的大帝控制组补充诊断。
- v1与v2在这些冻结真实输出上的grounded结果相同；合成回归测试才暴露v1跨池风险。

这是一条历史错误展示被拦下，**不是新增一条正确回答，更不是新的79题E2E正确率**。
79条仍是58个问题的历史证据状态；旧自动19/79完整存在已复核假阳性，不能当确认正确率。

## 显式Exx对象schema实验已收尾：不晋级

对原Exx strict9，仅追加fact/evidence_ids对象示例，保持证据和E-ID映射相同：

- 9/9经修复解析后产生拟回答，但严格JSON合法数为0；全部被E-ID校验转为续检索。
- 8条正常stop的原始完成文本均缺少最外层结束大括号；1条达到512输出上限。
- 完成文本与cleaner输出逐条相同，故不能把此处缺括号归咎于输出清理器裁切。
- 修复解析无异常不等于原始JSON合规。
- 既有生成还出现不存在的E8、将莫斯提马与苦难陈述者混同、将威尔转述的观点当成威尔本人立场等问题。
- 同时，E-ID检查出现“雷吉斯希望”等拼接长词不能在引文逐字匹配的问题；其拒绝数也不是语义错答数。
- 最终可展示答案审核9/9完成，全部no_answer；审核的“证据充分”只是模型判断，不能当RAG召回率。

因此不切换默认协议，不放宽词项门槛来制造提升；缺括号与事实错误的根因仍需分开验证。
仅检查到base/adapter中大括号和EOS相关token编号相同，尚不能由此排除所有训练/序列化问题。

## 下一步（尚无结论）

已经准备新的18条配对开发实验：现有strict9每条各生成普通控制组和“读准问题、区分主体与对象”提示组。
每对采样seed相同、顺序交替；完整可见证据、问题、hypothesis、模型和预算不变。
不发送旧答案、gold或judge意见，不修改正式prompt。
这区别于已失败的机械引用反馈方案；只在持续60秒无GPU进程后运行，不抢占已有任务。
它仍是多次检查过的开发集，不可宣称独立测试正确率。

规范凭据目录：

```text
{run}
```

关键文件：`canonical_inferred_integrity_patch_v2/`、`inferred_integrity_v2_comparison.json`、
`inferred_integrity_v1_cross_pool_regression.xml`、`canonical_release_tests_inferred_integrity_v2.xml`、
`strict9_native_exx_explicit_schema_ctx12288_v1/`。
"""
destinations = [
    run / "INFERRED_INTEGRITY_CHECKPOINT_v4.md",
    repo / "docs/CANONICAL_INFERRED_INTEGRITY_CHECKPOINT_20260908_irPugk_v4.md",
]
for path in destinations:
    with path.open("x") as output:
        output.write(report)
assert sha(destinations[0]) == sha(destinations[1])
with (run / "inferred_integrity_checkpoint_v4_artifacts.json").open("x") as output:
    output.write(json.dumps({
        "report_sha256": sha(destinations[0]), "applied_state_sha256": sha(applied_path),
        "comparison_sha256": sha(comparison_path), "explicit_exx_analysis_sha256": sha(analysis_path),
        "explicit_exx_semantic_sha256": sha(exx_folder / "semantic9_v1.json"),
        "training_ready": False, "goal_complete": False,
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": str(destinations[1]), "sha256": sha(destinations[0]), "goal_complete": False}))
