"""New checkpoint only; no source, training-data or previous-report rewrites."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
replay = json.loads((HERE / "diagnostic_terms_replay_comparison_v2.json").read_text())
applied = json.loads((HERE / "applied_verified.json").read_text())
baseline = ET.parse(HERE / "baseline_diagnostic_terms_tests_v2.xml").getroot().find("testsuite")
assert int(baseline.attrib["tests"]) == 28 and int(baseline.attrib["failures"]) == 10
assert applied["canonical_tests"] == {"tests": 214, "failures": 0, "errors": 0, "skipped": 0}
states = replay["state_replay"]
assert states["states"] == 421
assert states["changes_by_source"] == {"probe": 2, "historical_training_input": 35}
assert not states["new_normalization_errors"]
assert all(not case["grounded_changed_ids"] and not case["parsed_changed_ids"] for case in replay["comparisons"])
files = "\n".join(
    f"- `{path}`：`{replay['candidate_source_hashes'][path]}`"
    for path in replay["changed_source_files"]
)
text = f"""# ASA 已有 hypothesis 诊断词传播修复检查点 v9

- UTC 时间：{datetime.now(timezone.utc).isoformat()}。
- 规范源：241 的 `${REMOTE_HOME}/ASA-ArknightStoryAgent`，HEAD `{applied['head']}`。
- 证据目录：`{HERE}`。

## 已落地的最小规则

v8 探针确认：旧 release 能清理新增 missing_slots 中的诊断，但会保留已有 hypothesis 的诊断实体/关键词。本阶段新增共享的检索词清理函数，并应用到模型 hypothesis 规范化、状态合并、启发式 follow-up、证据词扩展、主检索查询与 follow-up 查询构建。

只识别明确的控制字段/动作、grounding 诊断前缀与“缺少可校验”等诊断片段。不会将所有英文、下划线词或未知人物名都删掉。

- Logos、W、Kal'tsit、S.W.E.E.P.、AMa-10、AMa_10 有保留测试。
- 用户原问题明确要求的词仍保留，包括用户在问 `quote`、`answer_directly` 或 `missing_supported_facts` 时。
- “支撑”和 `quote` 本身可能有正常含义。候选 v1 曾按宽泛诊断上下文删除；落地的 v2 收紧为识别到“缺少可校验”片段才处理这两个碎片。只有其他机器错误时仍保留它们。
- 输入对象不被原地修改；合并先清理再分配有限词项位置。模型 entities 全是诊断时沿用已有 ModelOutputError 路径，不能把机器诊断当作可用实体。

这不是通用上下文污染检测器：未知诊断写法、已污染的历史 graph/minirag_hints、错误剧情实体或没有可辨诊断标记的普通词，并没有因此得到语义纠正。

## 验证结果

新增 28 个测试：旧规范实现 10 失败、18 通过；候选 release 156/156 通过；规范仓库 release 加 v7 构建/合并测试总计 **214/214 通过**。

只读状态回放共 421 项：

| 来源 | 状态数 | 有变化 |
| --- | ---: | ---: |
| 合成/历史实例探针 | 3 | 2 |
| 现有五组开发矩阵 | 183 | 0 |
| 历史训练输入中的去重 hypothesis | 235 | 35 |

历史训练源共有 791 条记录，390 条无已有 hypothesis 区段，166 条是重复 hypothesis 状态，剩余 235 个去重状态。**35 是状态数，不是 35 道新增正确问题**。

被清理的合并词项是 `answer_directly`、`missing_supported_facts`、“缺少可校验”、条件匹配的 `quote`/“支撑”及 supported_fact 开头的完整或截断诊断前缀。回放没有新增 hypothesis 规范化错误。原训练 JSON 的输入与标签一字未改，清理只发生在隔离回放/新运行代码中。

对 79 + 9 + 68 + 18 + 9 = 183 条冻结模型输出再次做同分析器前后对照，解析结果、grounded 结果、动作与展示答案全相同。这些是重复开发状态，不是独立测试题。原 79 状态仍是 39 直接回答、31 继续检索、9 弃答；不是 39/79 的语义正确率。

本阶段没有实际重新检索、生成或训练，不能用清理状态数声称 RAG 召回率或答案正确率提升。需在后续受控实验中测量真实效果。

## 范围与保护

4 个既有运行源码、1 个新增共享模块、1 个新增测试文件。校验了 {applied['source_files_checked']} 个运行 Python 文件；其他已有 dirty 文件、历史数据与冻结矩阵/预测哈希均未变。没有修改默认模型、提交/push、重启服务或向 248 发起训练。源码已修复不代表既有服务已热加载。

运行源码：

{files}

新增测试：`release/ASA-ArknightStoryAgent/tests/test_retrieval_diagnostic_terms.py`。

## 未解决与下一步

1. 完整 79 状态的旧语义审核有明确假阳性；下一步应使用内容口径 v2（driver `judge_visible_answers_content_v3.py`）重做完整审核，再分析具体生成错误。不能沿用旧 19/79 当真实正确率。
2. 历史训练标签/graph hints 不会因运行时清理自动变成正确标签；禁止直接训练未经独立复核的样本。
3. `clean_soda_api_verifier_dataset.py` 的旧按 prompt 切分、定点旧式答案重写，以及训练构建脚本独立 fallback 诊断处理，仍需分别审核；本阶段没有调用这些旧路径。
4. 单条引用关键词归属检查仍不能代替主客体、因果和回答覆盖度的语义验证。

## 复核文件

- `original/src/`、`candidate/src/`：本阶段前后完整运行源码副本。
- `baseline_diagnostic_terms_tests_v2.xml`、`candidate_diagnostic_terms_tests_v2.xml`、`canonical_diagnostic_terms_tests_v1.xml`。
- `baseline_query_states_v2.json`、`candidate_query_states_v2.json`：421 状态的完整前后结果。
- `diagnostic_terms_replay_comparison_v2.json`：源码/输入哈希、35 个历史变化、五组 grounding 对照。
- `preapply_manifest.json`、`applied_verified.json`：落地前后校验。

状态：持续改进中的检查点，不是项目完成声明。
"""
path = ROOT / "docs/CANONICAL_DIAGNOSTIC_PROPAGATION_REPAIR_20260908_irPugk_v9.md"
with path.open("x", encoding="utf-8") as output:
    output.write(text)
print(json.dumps({"report": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))
