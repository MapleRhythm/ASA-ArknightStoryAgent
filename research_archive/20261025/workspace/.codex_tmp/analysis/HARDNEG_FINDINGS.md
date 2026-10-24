# Claim-citation hard-negative 数据集 v2 质量报告

执行日期：2026-09-04（Asia/Shanghai）

## 产物

- 远端数据集：`${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl`
- 远端质量 JSON：`${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.report.json`
- 远端构建脚本：`${REMOTE_STORE}/exx_grounding_v1/tools/build_binding_hardneg_v2.py`
- 本地复现脚本：[build_binding_hardneg_v2.py](${ASA_ROOT}/.codex_tmp/analysis/build_binding_hardneg_v2.py)

数据集已按任务要求写入 248 的指定路径。本地 smoke 产物也保留在远端：

- `binding_hardneg_v2_20260904_smoke20.jsonl`
- `binding_hardneg_v2_20260904_smoke20.report.json`

## 执行流程

1. 从 `exx_grounding_v3_sft_minimal559_glm53_20260903/train.json` 解析
   `grounded_action_generation` 且 `next_action=answer_directly` 的可见证据池和
   `supported_facts` gold 绑定。
2. 使用 `${REMOTE_STORE}/rank_v8_coverage_241_mirror/Qwen3-Embedding-0.6B`
   （Sentence-Transformers，claim 使用 query prompt，证据使用 document 编码）计算
   每个 claim 与同一可见池中所有非 gold chunk 的余弦相似度。
3. 选取 top-4：同场景优先，其次同 source，再以 dense cosine 降序打破优先级内的平局；
   场景/source 从证据正文中可见的故事头解析，无法可靠解析时不强行推断。
4. smoke 先跑 20 个源记录；全量 dense 选取覆盖全部源记录，GLM 只验证前 300 个
   claim/fact row 以控制成本。每个 fact row 的 4 个候选合并为一次 GLM 请求。
5. GLM 使用 `glm-5.3`、`reasoning_effort=medium`、`max_tokens=4096`，只保留
   `supports=false` 的验证候选为干净 hard negative；`supports=true` 的候选移入
   `suspected_missed_positives`。GLM 预算之外的候选保留为 `glm_verdict=not_run`
   （明确标记为 provisional）。

## Smoke（20 条源记录）

- 源记录：20；fact row：53
- dense 候选：205（每个 fact row 最多 4 个）
- dense cosine：均值 `0.558603`，p50 `0.559298`，p90 `0.689289`，范围
  `0.193049–0.770731`
- GLM：183 `unsupported`，22 `supported`，0 error
- GLM 验证负例纯度：`183 / 205 = 89.27%`
- 疑似漏标正例：22 个候选，分布在 18 个 fact row

Smoke 证明 dense 编码、场景优先选取、GLM 请求/解析和输出 schema 均可运行后才执行全量。

## 全量结果

- 源记录：555
- fact row / 输出行：1706
- 唯一 claim：1697
- 唯一证据 chunk：3145
- dense top-4 候选：6718
- 同场景候选：130（候选阶段占 1.94%）
- 同 source 候选：272（候选阶段占 4.05%）
- 输出文件大小：约 14 MB

### Dense 相似度对比

| 选取方式 | 均值 | p50 | p90 | 最大值 |
|---|---:|---:|---:|---:|
| v1 char-trigram | 0.007571 | — | — | 0.337 |
| v2 Qwen3 dense cosine | 0.560142 | 0.565410 | 0.695948 | 0.853815 |

v2 的 dense 均值约为 v1 trigram 均值的 74 倍，且最低候选 cosine 也为
0.185460，已不再是旧版的近随机低重叠选择。

### GLM 验证（前 300 个 fact row）

- 验证 fact row：300
- 验证候选：1130
- `unsupported`：1005
- `supported`：125
- 最终验证负例纯度：`1005 / 1130 = 88.94%`
- 疑似漏标正例：125 个候选，分布在 98 个 fact row（300 行中的 32.67%）
- 最终 GLM error：0

因此，前 300 行中有 125 个“gold 之外其实也支持”的候选；这些候选没有混入
`hard_negatives`，而是写入每行的 `suspected_missed_positives` 和
`suspected_missed_positive_eids`。这个数是对 exact-evidence-set 假阴性/标注多解性的
直接抽样证据，但因为 GLM 只覆盖前 300 个 fact row，不能把 125 外推成全量精确计数。

全量输出中的 hard-negative 状态：

- 已 GLM 确认干净（`glm_verdict=unsupported`）：1005
- 尚未 GLM 验证、明确标记 provisional（`glm_verdict=not_run`）：5588
- 被判支持并移出的疑似漏标正例：125
- 没有保留 hard negative 的输出行：5（这些行的 top-4 候选均被 GLM 判为支持）

## 完整性检查

- JSONL 可逐行解析：1706/1706
- 必需字段齐全：1706/1706
- `positive_eids` 与 `positive_texts` 长度一致：1706/1706
- `suspected_missed_positive_eids` 与对象列表一致：1706/1706
- 输出中 `glm_verdict=error`：0
- 每行 hard-negative 数量不超过 4

初次全量并发时有一个 4-candidate 请求收到 GLM HTTP 429；脚本随后将错误缓存视为
不可复用，等待后以较低并发重跑。最终报告中的 `error_count=0`，但缓存文件保留了
历史错误行以便审计。

## 结论与使用建议

v2 已完成并满足 dense+场景选取与 GLM 验证要求。dense 相似度分布相对 v1 有数量级
提升，GLM 验证候选的“不支持”比例约 89%，说明负例整体是“相关但不支持”而非随机
背景。训练时建议：

- 优先使用 1005 个已 GLM 确认的 `unsupported` 样本；
- `not_run` 样本作为未审计的候选集使用，并保留其 `glm_verdict` 标记；
- 单独利用 125 个 `suspected_missed_positives` 做 gold 绑定复核/多解性分析，不要把
  它们当作 hard negative。
