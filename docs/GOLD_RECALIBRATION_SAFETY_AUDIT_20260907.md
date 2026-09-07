# GOLD_RECAL 全量校准安全复核（2026-09-07）

## 结论

`GOLD_RECAL` 的 GLM judging 确实完成了，不是断线后漏跑：

- `facts_full.jsonl`：1929 行、1929 个唯一 `fact_id`；
- `judgements_full.jsonl`：1929 行、1929 个唯一 `fact_id`；
- 两者一一对应，没有 missing/extra；
- 1929 个 merged 和 2259 个 individual judgement 均为 `status=ok`；
- 411 个 fact 变更和 50 个动作翻转也能从 judgement 完整复算。

但首轮 judge 对每个 E-ID 只提交了前 600 个字符。完整 manifest 中有 247 个
fact 的被引文本超过该上限，其中 172 个被判为 `unsupported`。另有 86 个
merged/individual 相互矛盾的 fact。因此历史 `1518/123/288` 是可复现的
**strict-v1 运行结果**，不是可直接训练的干净真值；由它生成的 strict 数据、
clean-SFT adapter 和 calibrated 指标只能保留为历史对照。

生产 runtime 未因本次复核发生任何变化。

## 机械复核结果

| 项目 | 数量 |
| --- | ---: |
| 全部 fact | 1929 |
| strict-v1 `clean_support` | 1518 |
| strict-v1 `partial` | 123 |
| strict-v1 `unsupported` | 288 |
| judge 输入发生 600 字截断 | 247 |
| merged/individual 冲突 | 86 |
| 两类风险去重后的隔离集 | 326 |
| 完整输入且无冲突的保守可用集 | 1603 |

截断高度集中在负标签：

| strict-v1 标签 | 总数 | judge 输入被截断 | 完整输入且无冲突 |
| --- | ---: | ---: | ---: |
| `clean_support` | 1518 | 40 | 1478 |
| `partial` | 123 | 35 | 88 |
| `unsupported` | 288 | 172 | 37 |

所以不能继续说“确认有 288 个 unsupported”。在不追加完整证据复判的条件下，
只能说 37 个 `unsupported` 同时满足“原 judge 看全输入”和“内部判定无冲突”；
它们仍是单个 GLM 的候选标签，不等于人工 gold。

50 个历史动作翻转中，46 个依赖至少一个被隔离 fact，只有 4 个不依赖截断或
内部冲突。`answer_directly → retrieve_more/abstain` 必须在完整证据复判后按题
重新计算，不能沿用 strict-v1 的 50。

此次 1929 个 fact 只来自 train（1706）和 val（223）。test 没有可校准 fact，
不能表述成“train/val/test 1929 条全部校准”。

## 确定性假阴性

这不是仅凭分布推断。完整 evidence 中能直接找到被截断掉的支持原文：

- `bb9741a237c6a19cd12ec3a0`：E3 共 2886 字；“工厂没炸掉”在字符 1569
  附近，“城镇议会问我拨给我的钱去哪儿了”在 1821 附近，“被感染者炸了”
  在 1849 附近。judge 只看前 600 字后却以“未提及炸毁工厂、嫁祸感染者”
  为由删除该 fact。
- `15312cee6b76a47cda023d28`：主张“海嗣重围之下一座摇摇欲坠的危城”
  在 E1 字符 720 处逐字出现，仍被判 `unsupported`。
- `3db7614f8407996f74c2ce83`：学生被押送到凛冬所在校园的完整主张在 E1
  字符 1366 处逐字出现，仍被判 `unsupported`。
- `c07ffc76b58843ddae1b4dc2`：儿女不孝、亲友觊觎财产的完整主张在 E1
  字符 1891 处逐字出现，仍被判 `unsupported`。
- 其他人工可直接确认的后段支持还包括：真理本名、PCS 统一种族、火龙未被
  感染、凯尔希对阿戈尔接触风险的判断、U-Gigamax 的开发目的。

仅逐字完全匹配就找到 4 个假阴性；其余大量主张是正常释义，不能用 exact
substring 计数代替完整复判。

## 第三次歧义裁决也需重新生成

历史 ambiguity sidecar 对 86 个冲突项进行了第三次 GLM 裁决，但当时脚本仍把
每个 E-ID 截到 1800 字。86 项中有 3 项超过 1800 字，且 sidecar 没有保存
“完整 evidence 内容指纹”，无法证明裁决与当前 manifest 完全一致。

本次已修改 `scripts/adjudicate_gold_binding_ambiguities.py`：

- `--include-truncated` 会选择截断或冲突的 326 个 fact；
- 对截断项采用 blind rejudge，不把错误的旧 verdict/reason 提供给新 judge；
- 提交每个 E-ID 的完整文本，不再使用 600/1800 字静默截断；
- 输出 `full_evidence=true`、逐 E-ID 字符数和 `evidence_sha256`；
- 下游只有在指纹与 manifest 完全一致时才接受该 full-evidence 裁决。

复判命令：

```bash
python3 scripts/adjudicate_gold_binding_ambiguities.py \
  --facts /path/to/facts_full.jsonl \
  --judgements /path/to/judgements_full.jsonl \
  --output /path/to/gold_recalibrated_full_evidence_adjudications.jsonl \
  --include-truncated
```

当前本机没有 `BIGMODEL_API_KEY`，且 248 的新 SSH 命令没有可用的非交互认证，
所以本次提交没有伪造 326 条新 verdict；训练门禁保持关闭。

## 下一轮 paired SFT 的数据边界

完整证据复判完成前：

- 1478 个无截断、无冲突的 `clean_support` 可作为候选正例；
- 88 个无截断、无冲突的 `partial` 只能保留已确认 E-ID，并继续做 set-level
  复核；
- 37 个保守 `unsupported` 仍建议二次裁决或人工抽样后才作为 hard target；
- 326 个隔离项不能删除 gold、翻转动作、构造 confirmed negative 或进入
  “clean”训练集；
- strict-v1 clean-SFT 继续作为历史 adapter 对照，不作为下一轮初始化标签真值。

full-evidence 复判后按如下规则合并：

- `supported_by_union`：保留完整 E-ID 联集；
- `supported_by_some`：原复合 claim 不能作为正例；先拆出受支持的原子子句，
  再独立复核并绑定 `keep_eids`；
- `ambiguous`：隔离或人工复核，不转 hard negative；
- `unsupported`：移除 fact；只有一题所有 fact 均可靠移除后，才重新判断动作。

同一 question、`question_key`、事件及其 Full/distractor/missing/decoy 变体必须
先分组再划 split，避免 paired SFT 的同源状态泄漏到验证集。

## 产物与复现

只读审计入口：

```bash
python3 scripts/audit_gold_recalibration.py \
  --facts /path/to/facts_full.jsonl \
  --judgements /path/to/judgements_full.jsonl \
  --changes /path/to/gold_recalibrated_changes.jsonl \
  --report /path/to/report.json \
  --quarantine /path/to/quarantine.jsonl
```

仓库内产物：

- `data/eval/gold_recalibration_safety_report_20260907.json`
- `data/eval/gold_recalibration_quarantine_20260907.jsonl`

输入 SHA256：

| 输入 | SHA256 |
| --- | --- |
| facts | `f99aa73524a16ccb075f193e21495b128a38da8c1faa2d024d877064f22306f1` |
| judgements | `8ebc106c4d2505735992028fa926218ecac7c1f6e0f80f06b92b0da81efb36c6` |
| changes | `76e8ffcd19a67caaf19d63f8911b9128c06a30e63732febcfff989bf6e53874f` |

输出 SHA256：

| 输出 | SHA256 |
| --- | --- |
| safety report | `90932b648afeb0f63a96f9e22d2d24e03c128bd472089c70259d35137d59e038` |
| quarantine | `4dc2204287968e80c1b4ca29eab221cccce19dcbb36a2f7c4ef06287cc9ccb5c` |
