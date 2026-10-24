# Hard-negative v2 全证据安全复核（2026-09-07）

## 结论

历史 hard-negative v2 的 dense、同场景和 GLM 验证流程确实产出了完整文件，
但 GLM 实际看到的每个 candidate 在判断前被执行了 `text[:2000]`。因此
历史的 1005 个 `unsupported` 和 125 个 `supported` 不是全证据标签，基于
它们训练和评估的 augmented binding verifier v2 只能作为历史对照。

本次已恢复完整 source prompt 并完成 480 个隔离 candidate 的盲复判，没有
修改生产 runtime。复判结果仍是单一 GLM-5.3 裁决，不等于人工 gold；只有
通过四元组连接、完整文本前缀和 SHA 校验的结果才可进入离线 pair builder。

## 机械复核

| 项目 | 数量 |
| --- | ---: |
| fact rows / 唯一 fact key | 1706 / 1706 |
| source records | 1005 |
| dense top-4 candidates | 6718 |
| 历史 `unsupported` | 1005 |
| 历史 `supported` | 125 |
| `not_run` | 5588 |
| 被 2000 字截断的 candidate | 480 |
| 已判且被截断 | 345 |

按历史 verdict 拆分：

| verdict | 历史数量 | 输入被截断 | 看到了完整 candidate |
| --- | ---: | ---: | ---: |
| `unsupported` | 1005 | 286 | 719 |
| `supported` | 125 | 59 | 66 |
| `not_run` | 5588 | 135 | 5453 |

所以 `1005/1130 = 88.94%` 只能描述历史 judge 输出比例，不能再称为
“已验证负例纯度”。尤其不能把 1005 个 `unsupported` 全部用于训练。

## 确定性尾部假阴性

完整 source prompt 直接证明截断不只是理论风险：

- 贝希曼炸厂候选 E1 共 3288 字；“城镇议会问我拨给我的钱去哪儿了”和
  “被感染者炸了”分别在字符 2637、2663 附近，历史 verdict 却是
  `unsupported`。
- PCS 统一种族候选 E2 共 3947 字，“同一种族”在字符 3109 附近；E6
  共 3964 字，相同关键信息在 3126 附近，两者都被判 `unsupported`。
- 斯第奇雨林判断候选 E2 共 3336 字，“历史性地宏观”和“千百万年之后”
  分别在字符 2782、2795 附近，历史 verdict 为 `unsupported`。
- 凯特说明卡拉顿议会拨款的候选 E2 共 3216 字，关键原文在字符 2108
  附近，历史 verdict 为 `unsupported`。
- 塞雷娅评价实验的候选 E3 共 3233 字，“冒进而无谋”在字符 2231
  附近，历史 verdict 为 `unsupported`。

同类风险还出现在瓦列里与卓娅、火龙是否感染、真理本名等记录中。以上例子
只用于证明旧标签存在确定性错误；全部 480 个隔离候选现已完成完整证据复判。

## 与 gold 门禁取交集

修正后的 `scripts/build_binding_verifier_pairs.py` 默认同时拒绝：

- gold judge 的 600 字截断项；
- merged/individual 冲突但没有 full-evidence SHA 裁决的项；
- hard-negative 中达到旧 2000 字上限、却没有完整证据指纹的 candidate；
- `supported_by_some` 的原始复合 claim。

用相同历史输入重建（未接入 full-evidence sidecar）：

| 项目 | 数量 |
| --- | ---: |
| 保守 pair | 393 |
| record groups | 65 |
| train | 360 |
| eval | 33 |
| clean-support pair | 369 |
| partial pair | 24 |

复现命令：

```bash
python3 scripts/build_binding_verifier_pairs.py \
  --hardneg /path/to/binding_hardneg_v2_20260904.jsonl \
  --manifest /path/to/facts_full.jsonl \
  --judgements /path/to/judgements_full.jsonl \
  --out-dir /path/to/safe_pairs
```

接入已校验的完整证据 sidecar：

```bash
python3 scripts/build_binding_verifier_pairs.py \
  --hardneg /path/to/binding_hardneg_v2_20260904.jsonl \
  --manifest /path/to/facts_full.jsonl \
  --judgements /path/to/judgements_full.jsonl \
  --ambiguity /path/to/gold_recalibrated_full_evidence_adjudications_20260907.jsonl \
  --full-evidence-rejudgements /path/to/binding_hardneg_full_evidence_rejudgements_20260907.jsonl \
  --include-suspected-missed-positives \
  --out-dir /path/to/safe_pairs_with_missed
```

这 393 条仍不是最终训练集。接入 full-evidence sidecar 后，默认不纳入
suspected missed positives 时得到 817 对、123 个 record group（744 train /
73 eval）；显式加入 98 条通过完整性门禁的 GLM-supported missed-positive
变体（包括新 full-evidence 复判及旧的未截断支持判定，并非全是新复判）后得到 1019
对、123 个 group（907 train / 112 eval）。builder 仍只按 `record_id` 隔离；
原 hard-neg 1706 行中有 1331 行没有 `question_key`，也没有完成 story-family
分组，因此这些规模不能替代正式 family-held-out 训练集。

## 历史 verifier v2 的处置

历史 augmented v2 的 766 pairs、93.9% missed-positive accuracy 和 0.904 AUC
都继承了两类问题：

1. hard-negative 的 2000 字输入截断；
2. 旧 builder 把 `supported_by_some` 的完整复合 claim 当成正例。

这些数字可以保留用于复现实验过程，但不能作为真实 binding 泛化能力、
question-document retrieval 能力或生产切换依据。生产 runtime 保持冻结。

## 复判和训练边界

完整证据复判已完成。每条结果保存：

- `record_id`、`fact_index`、claim、E-ID；
- 完整 candidate 文本或可复原 source locator；
- `full_evidence=true`；
- claim/E-ID/完整文本的 `evidence_sha256`；
- judge model、reasoning effort、status、verdict 和 reason。

只有 `status=ok`、指纹与当前 source 完全一致的结果才能解封。此次 480/480
通过校验：新 verdict 为 `supported=131`、`unsupported=349`；旧→新翻转矩阵
见 `data/eval/binding_hardneg_full_evidence_rejudgements_20260907.report.json`。
API 错误、超时和截断结果不计入完成数。2026-09-08 的续审发现旧
`supported_by_some` 混淆“单段支持整条”与“主张只部分成立”，因此必须先
盲重判支持范围；真正的 partial 再拆成原子 claim 复核。不能通过保留某个
E-ID 把原复合主张变成正例，也不能将旧 96 个 some 全称作真实 partial。
详见 `SYSTEM_ACCURACY_DIAGNOSIS_20260908.md` 的训练入口续审。

## 产物与 SHA256

仓库内只读审计产物：

- `data/eval/binding_hardneg_safety_report_20260907.json`
- `data/eval/binding_hardneg_quarantine_20260907.jsonl`
- `data/eval/binding_hardneg_full_evidence_rejudgements_20260907.jsonl`
- `data/eval/binding_hardneg_full_evidence_rejudgements_20260907.report.json`

输入：

| 输入 | SHA256 |
| --- | --- |
| hard-negative v2 | `fd3ef76db717cb5c312de3fccf6f83cd2f48e15baf8ce02854455054740127b8` |
| source rows | `3a2da70987c1413895f3708cb79cc3d95a59d4d690414c3f1ff6689399f0e541` |

输出：

| 输出 | SHA256 |
| --- | --- |
| safety report | `4d1fe5a671c6add63541ea2f9dc42a989ea1d9ea1132dd0cc38d6213cdd8f34a` |
| quarantine | `8eb6705da668ba384aba2a48ae51174efbdbed2ea8a4f0088bf8854213ea9f91` |
| full-evidence rejudgements | `9f5ff5fa0795723142efdb4ff5abd15f1129a26680e61c28a6a3b544d55815dd` |
| full-evidence rejudgement report | `ad4d1ba048a83a4f7e3cdfaccce2c38bd6a19b83740ec5a360021547e6758c60` |
