"""Persist training-contract, split-integrity and corrected-rubric findings."""
from collections import Counter
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
applied = load(run / "canonical_inferred_integrity_patch_v2/applied_verified.json")
assert applied["test_count"] == 115
for item in applied["effective_changes"]:
    assert sha(repo / item["path"]) == item["after_sha256"], item["path"]
lineage_path = run / "adapter_lineage_and_label_conflicts_v2.json"
lineage = load(lineage_path)
assert len(lineage["all_recorded_train_question_overlap_with_fixed79"]) == 31
assert lineage["fixed79_distinct_questions"] == 58
latest, parent, sft = lineage["stages"]
assert [len(s["train_val_question_overlap"]) for s in (sft, parent, latest)] == [0, 121, 7]
assert len(latest["splits"]["val"]["exact_label_conflicts"]) == 1
folder = run / "question_focus_paired18_ctx12288_v1"
summary_path = folder / "semantic_summary18_content_v2.json"
summary = load(summary_path)
control = summary["summary"]["focus_control_v1"]
focus = summary["summary"]["question_focus_v1"]
assert control["records"] == focus["records"] == 9
assert control["labels"] == {"supported_complete": 3, "no_answer": 4, "supported_partial": 1, "unsupported": 1}
assert focus["labels"] == {"supported_complete": 2, "no_answer": 6, "unsupported": 1}
empty_question_id = "9af639a51c9a8d7bf0cfb860::after::runtime_raw::question_focus_v1"
assert focus["label_changed_ids"] == [empty_question_id]
claims_path = run / "cited_claim_entailment_probe7_retry1.json"
claims = load(claims_path)
assert len(claims["results"]) == 7 and all(r["status"] == "ok" for r in claims["results"])
claim_counts = dict(Counter(r["judgement"]["relation"] for r in claims["results"]))
assert claim_counts == {"entailed": 4, "not_established": 2, "contradicted": 1}
metadata = load(folder / "metadata.json")
assert metadata["completed_rows"] == 18 and not metadata["input_truncated"]
stage_lines = []
for label, stage in (("最初SFT", sft), ("上一阶段KTO", parent), ("当前LoRA最后KTO", latest)):
    train, val = stage["splits"]["train"], stage["splits"]["val"]
    stage_lines.append(
        f'| {label} | {train["rows"]} / {train["distinct_questions"]} | '
        f'{val["rows"]} / {val["distinct_questions"]} | {len(stage["train_val_question_overlap"])} |'
    )
text = f"""# ASA 训练契约、评测污染与实质回答审核检查点 v5

记录时间（UTC）：`{datetime.datetime.now(datetime.timezone.utc).isoformat()}`。
241为规范源，248仅训练评测镜像。本报告接续v4，所有历史结果保留，不表示长期正确率目标完成。

## 结论

1. 当前LoRA保存的三阶段训练链全部使用旧式`answer`输出，与当前事实逐条绑定引用的运行契约不同。
   这证明存在训练—运行格式差异，但不能单凭差异证明所有错答都由此引起。
2. 后续KTO的训练/验证问题大量重叠；不能把其验证损失或原79状态当作独立泛化正确率。
3. 新“读题/主客体”提示实验失败，不加入默认prompt。
4. 旧审核口径存在代码设计缺陷：禁止将`answer_directly`下的空洞问句归为未答。
   新版已用8项测试及18条真实输出重评纠正，未改写旧结果。
5. 逐条引用块语义检查在两题7条事实中识别出一项明确主体错误，值得继续验证；尚未接入正式链路。

正式release仍是v4核验的**115/115测试通过**版本；本阶段没有再修改运行源码、配置或模型指针。
未提交、未push、未重启服务、未训练；工作区修复不能冒充运行中服务已热加载。

## 三阶段训练链：格式差异和验证问题重叠

通过当前LoRA真实路径、模型卡、配置中的adapter输入/输出路径和dataset_info追溯到：

`基础4B → SFT → 550-question KTO → 当前最后阶段KTO`。

当前保存的正例conclusion输出：最初SFT训练492条、上一KTO训练1092条、最后KTO训练63条，
均为旧式`answer`字段，未包含当前的`supported_facts`结构。假设生成和续检索任务单独统计，没有误当回答任务。
旧的负例包含格式失败是KTO的正常用途，不能把负例格式错误都说成正例标签污染。

| 阶段 | 训练记录 / 问题数 | 验证记录 / 问题数 | 训练与验证重叠问题数 |
|---|---:|---:|---:|
{chr(10).join(stage_lines)}

最初SFT的问题分组隔离是正确的；后两阶段KTO没有保持该隔离。
三阶段训练集问题并集与原79状态的58个问题有**31个规范化文本完全相同的问题**。
这里的“见过问题”包括假设或回答任务、正负例，不等于证明记住了完整正确答案；
也没有做近义问题或剧情家族级去重，因此不是完整污染上界。

最后KTO验证集还有同一输入和完全相同输出被赋予相反标签的一组记录：

- `7cb1582e206cb408-teacher-verifier-rejected`：false。
- `7cb1582e206cb408-verifier-chosen`：true。

它位于验证集，不应说成在训练集发现同样的矛盾。此次审计在两阶段KTO训练集没有发现精确/JSON键序归一化后的同输入同输出相反标签。
三个阶段均未发现完全相同样本直接跨train/val复制；问题重叠与完整样本重复是不同的检查。

这些是**当前保存的配置和数据**证据，尚未认证历史CLI覆盖参数、预处理缓存或训练时的数据原始字节。
因此不把它们擅自归因于2026-09-06的某次模型操作。
不改原始标签、不删历史记录；事后重新分组也不能使已经见过这些问题的现有模型变成“未见过”。

## 18条同证据配对提示实验：不采用

现有strict9每题两个版本，同一显式采样seed、顺序交替，完整可见证据/E-ID映射/模型/2048输出预算不变。
新版本只追加通用读题、主客体和否定条件检查，没有发送旧答案、gold或judge结论。
空闲4090持续60秒无进程后才启动。18条全部正常stop，token逐项核对无输入裁切。

新版实质内容口径（仍为自动暂判）：

| 版本 | 记录 | 完整 | 部分 | 不支持 | 未提供实质回答 | 审核失败 |
|---|---:|---:|---:|---:|---:|---:|
| 控制 | 9 | 3 | 1 | 1 | 4 | 0 |
| 读题提示 | 9 | 2 | 0 | 1 | 6 | 0 |

控制与提示组都保留了“威尔辱骂芙蕾达为叛徒”的错误，实际原文是威尔转述汤姆的辱骂并替芙蕾达辩护。
提示组比较题甚至将final_answer写成“阿米娅和阿米娅在对待战斗的态度上有何不同？”。
因此不扩大这一失败提示方案，也不以结构合法数或引用放行数宣传提升。
这些是反复检查过的开发题，非独立E2E测试；3/9与2/9不能作为系统整体正确率。

## 已纠正审核工具缺陷：未答不能只看动作

旧rubric明定`no_answer=动作不是answer_directly`；旧validator又强制二者等价。
因此，即使审核模型已指出纯问句“没有任何实质回答”，也无法合法返回no_answer，
被迫落入supported_partial等类别。**这是旧评测设计的问题，不应全甩给审核模型。**

新版协议`asa_visible_answer_semantics_v2`：

- 将answer_content与action分开；`answer_directly`也可以是无实质回答。
- complete/partial必须有至少一条逐字来自展示答案、确实提供回答信息的supported_answer_spans。
- 空内容、纯问句、套话不能靠“没有错误断言”算部分正确。
- 动作选择、证据充分性仍独立评价；精确摘句只是完整性检查，不保证模型判断永远正确。

8项工具测试通过，包含旧validator真实复现拒绝“direct action + no_answer”的回归。
相同18条实际答案重评18/18成功：只有上述纯问句从旧partial变成新no_answer，其余17条类别不变。
旧结果原文件不改，新协议、rubric、transport和分析输入均以哈希绑定。
这修正的是测量口径，不是模型突然多答对一题。

## 逐条事实与所引完整E块的语义检查：7条小探针

选择两个已知问题状态中的全部规范化事实（5+2），每条单独审核。
只发question、原样claim、其实际引用的完整可见E块，不发final_answer、旧judge、参考答案或隐藏源文。
临时审核工具有7项测试，强制claim不得被换主客体或改写，判断引用须来自所给E块逐字连续原文。

首次6成功、1因审核器把原文空格删掉而引用不匹配失败；只重试失败的1条，7/7最终成功：

- 4条由引用块支持。
- 2条在所引块中未充分建立：均涉及只引用E1却断言阿米娅不愿战斗，相关依据实际上在其他块；
  不能把“所引块不支持”直接说成整个剧情中必然为假。
- 1条与原文相反：威尔被误写成辱骂者，E1明确说辱骂者是汤姆。

这是识别了一类真实错误，不是验证器的独立正确率。
审核器对解释细节也可能过宽，仍需逐字源文复核。
不直接把所有结构化facts拼成答案：引用匹配本身没有证明这些facts真实，重建可能放出额外无依据内容。
目前没有上线语义gate、自动删fact、自动修答案或训练这些新标签。

## 后续优先顺序

1. 检查241上已存在且训练格式匹配的候选权重，追溯数据和训练契约后做隔离对照，不改默认模型。
2. 扩展逐条引用块语义检查的正例与错误类型；区分编号/引用机械错误、所引证据不足、真实语义矛盾。
3. 将“生成修复”与“再检索”分开实验；不得以重新检索掩盖已有证据下的表达/角色错误。
4. 新训练之前先使用现有数据审计门禁检查问题/剧情家族隔离、同输入输出矛盾标签、当前协议和token预算；
   不复用未审定的自动judge标签，不把重新分组的旧题冒充当前模型的全新测试题。
5. 继续保留完整分母，最终仍需新检索、真实多轮与final fallback的完整E2E评测。

规范凭据目录：

```text
{run}
```

关键凭据：`adapter_lineage_and_label_conflicts_v2.json`、
`question_focus_paired18_ctx12288_v1/semantic_summary18_content_v2.json`、
`cited_claim_entailment_probe7_retry1.json`、
`answer_content_semantics_tests_v2.xml`、`claim_entailment_probe_tests_v1.xml`。
"""
destinations = [
    run / "TRAINING_CONTRACT_CONTENT_CHECKPOINT_v5.md",
    repo / "docs/CANONICAL_TRAINING_CONTRACT_AND_CONTENT_AUDIT_20260908_irPugk_v5.md",
]
for path in destinations:
    with path.open("x") as output:
        output.write(text)
assert sha(destinations[0]) == sha(destinations[1])
with (run / "contract_content_checkpoint_v5_artifacts.json").open("x") as output:
    output.write(json.dumps({
        "report_sha256": sha(destinations[0]), "lineage_sha256": sha(lineage_path),
        "content_summary_sha256": sha(summary_path), "claim_probe_sha256": sha(claims_path),
        "applied_state_sha256": sha(run / "canonical_inferred_integrity_patch_v2/applied_verified.json"),
        "production_source_changed_this_checkpoint": False, "training_ready": False, "goal_complete": False,
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": str(destinations[1]), "sha256": sha(destinations[0]), "goal_complete": False}))
