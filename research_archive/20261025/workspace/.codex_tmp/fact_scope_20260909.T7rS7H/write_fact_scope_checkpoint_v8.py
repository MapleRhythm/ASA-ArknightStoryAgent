from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
replay = json.loads((HERE / "fact_scope_replay_comparison_v1.json").read_text())
applied = json.loads((HERE / "applied_verified.json").read_text())
probe = json.loads((HERE / "hypothesis_diagnostic_probe_v1.json").read_text())
assert applied["canonical_tests"]["tests"] == 186
assert applied["canonical_tests"]["failures"] == applied["canonical_tests"]["errors"] == 0
assert sum(c["rows"] for c in replay["comparisons"]) == 183
assert all(not c["action_changed_ids"] and not c["answer_changed_ids"] for c in replay["comparisons"])
assert probe["results"][0]["cleaned_missing_slots"] == ["离开的具体原因"]
assert "answer_directly" in probe["results"][2]["merged"]["entities"]
rows = "\n".join(
    f"| {c['name']} | {c['rows']} | {len(c['action_changed_ids'])} | {len(c['answer_changed_ids'])} | {len(c['grounded_changes'])} |"
    for c in replay["comparisons"]
)
files = "\n".join(
    f"- `{name}`：`{replay['candidate_source_hashes'][name]}`"
    for name in replay["changed_source_files"]
)
text = f"""# ASA 单条事实引用边界与诊断传播检查点 v8

- UTC 时间：{datetime.now(timezone.utc).isoformat()}。
- 规范源：241 的 `${REMOTE_HOME}/ASA-ArknightStoryAgent`；HEAD `{applied['head']}`，未提交/push。
- 本阶段证据目录：`{HERE}`。

## 已落地

修复 supported_facts 内部、inferred_facts 内部的“跨事实借用引用词项”问题。校验单条 fact 时只检查该 fact 自己的 evidence_refs；重建展示答案的相同词项检查也按单条 fact 执行。没有扩大或缩减引用，没有改提示词、模型、预算、检索配置或服务。

原实现虽然已隔离 supported/inferred 两组，但同组仍可借用兄弟 fact 的引用。例如“甲是乙的母亲”只引用“甲与乙在门口碰面”，却因另一 fact 引用了“丙是丁的母亲”通过“母亲”词项检查。这个例子只证明**引用归属不正确**，不能凭词项匹配判断全局剧情真伪。

新增 13 个测试：旧实现 10 失败、3 通过；候选 release 128/128 通过（另有一个测试缓存路径权限警告，不是运行代码错误）；正式规范仓库的 release 加 v7 构建/合并测试共 **186/186 通过**。其他已有 dirty 文件哈希未变；源文件校验覆盖 {applied['source_files_checked']} 个 Python 文件。

## 冻结输出前后回放

使用相同 v4 分析器分别加载修改前规范源码与隔离候选源码；逐条核验原始生成输出、解析结果、矩阵/预测文件哈希相同，没有重新生成答案或检索。

| 开发状态集 | 条数 | 动作变化 | 展示答案变化 | 结构/诊断变化 |
| --- | ---: | ---: | ---: | ---: |
{rows}

两条诊断变化都属于“塔露拉为什么会引发切尔诺伯格事件？”：原 79 状态及其普通重试状态。二者之前和之后均为 retrieve_more，原诊断“目的”新增“设计”；没有新增拒答，也没有新增正确答案。

79 状态仍为 39 直接回答、31 继续检索、9 弃答。**39/79 是机械放行动作比例，不是回答正确率**。这些 183 条包含重复开发问题/状态，不是 183 道独立留出测试。本阶段不更新语义正确率结论；仅修复一个已复现的校验漏洞。

## 诊断词污染：确认传播，尚未修复

只读探针分别测试了干净 hypothesis、合成已污染 hypothesis、真实历史样本的 hypothesis：

- 干净 hypothesis + 技术 missing_slots：当前清理已生效，保留剧情缺口“离开的具体原因”，不把新诊断加进实体/关键词。
- 已污染 hypothesis + 空 missing_slots：`answer_directly`、`missing_supported_facts`、完整诊断短语仍被保留。
- 历史斯第奇样本回放到当前代码：entities/keywords 中的 `answer_directly`、`缺少可校验`、`quote`、`支撑`、`missing_supported_facts` 经 heuristic follow-up 和 merge 后仍存在。

这证明当前函数会**传播已有污染**，不证明它会从干净输入首次制造相同污染。没有实际运行检索来衡量召回损失，也没有在本阶段修改、清洗历史数据。

下一步应在 hypothesis 的规范化、已有状态合并与检索查询入口做一致且保守的清理测试，保留正常英文人物名、别名，以及用户问题本身要求的术语。不能把“删除所有英文/下划线词”当作解决方案。v7 中提到的旧清洗脚本和训练 fallback 路径也仍需处理。

## 仍然存在的边界

按单条引用做关键词检查仍不是语义蕴含验证；主客体颠倒、同词不同人、因果错配、答非所问都可能通过。不能因此晋级模型、批准自动训练标签或声称已经提高端到端正确率。默认 LoRA 未换，248 没有新训练，服务没有重启；源码修复不等于已运行服务自动热加载。

## 凭据

- `baseline_fact_scope_tests_v1.xml`：旧实现失败。
- `candidate_fact_scope_tests_v1.xml`、`canonical_fact_scope_tests_v1.xml`：候选与规范回归。
- `fact_scope_replay_comparison_v1.json`：183 状态配对结果、完整变更实例、源码/矩阵/预测哈希。
- `hypothesis_diagnostic_probe_v1.json`：三组原始输入与传播输出。
- `original/`、`candidate/src/`、`preapply_manifest.json`、`applied_verified.json`：备份、隔离候选与落地校验。

修改后的两个运行源码：

{files}

新增测试：`release/ASA-ArknightStoryAgent/tests/test_fact_scoped_quotes.py`。

状态：持续改进中的检查点，未完成整个正确率提升目标。
"""
path = ROOT / "docs/CANONICAL_FACT_SCOPE_AND_DIAGNOSTIC_CHECKPOINT_20260908_irPugk_v8.md"
with path.open("x", encoding="utf-8") as output:
    output.write(text)
print(json.dumps({"report": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))
