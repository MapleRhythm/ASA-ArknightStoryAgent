# Exx gold evidence binding recalibration

执行日期：2026-09-04（GLM 产物生成时间；2026-09-06 复核）

## 结论

248 上的全量 judging 已完成并可复核：`facts_full.jsonl` 与
`judgements_full.jsonl` 均为 1929 行，`fact_id` 唯一且一一对应，所有 merged
judgement 均为 `status=ok`。strict 标签集和变更清单均已生成。

严格判定结果显示原 gold 绑定并非全部可靠，但问题比例低于 AGREEMENT 中
“gold positive 的 48.4% 不支持”这一 250 条 pair 抽样结果。原因是本任务按
完整 fact 的合并证据重判，并对多 E-ID 事实做了单段判定；两者分母不同，不能
直接比较为同一个比例。

## 全量统计

| 指标 | 数值 |
|---|---:|
| 原始 fact 数 | 1929 |
| clean_support | 1518（78.69%） |
| partial（删除不支持的 E-ID 后保留） | 123（6.38%） |
| unsupported | 288（14.93%） |
| 清洗后保留 fact | 1641 |
| 多 E-ID fact | 330 |
| 其中 partial | 123（多 E-ID 的 37.27%） |
| 变更条目 | 461 |
| 因事实被全部移除而翻转动作的题目 | 50 |

动作翻转后，train 的 `answer_directly` 为 555→510，`retrieve_more` 为
129→171，`abstain` 为 99→102；val 的 `answer_directly` 为 67→62，
`retrieve_more` 为 7→12，`abstain` 保持 5。

## 分集

- train：1706 facts，unsupported 259（15.18%），清洗后 1447 facts，动作翻转 45。
- val：223 facts，unsupported 29（13.00%），清洗后 194 facts，动作翻转 5。
- test：源文件没有可校准行，输出文件为空；不能据此声称 test 已完成校准。

train 的 unsupported 率按 query type：

- causality 15.91%
- fact 16.46%
- reasoning 14.10%
- relation 16.16%
- reveal 10.00%
- mystery 0%（该类样本量很小）

30 条 smoke 的 merged fact unsupported 为 12/30（40%）；按单段 binding 计算
为 19/37（51.35%），落在任务要求的 40–55% 交叉验证区间。

## 产物

远端目录：

`${REMOTE_STORE}/exx_grounding_v1/data/gold_recalibrated_20260904/`

- `gold_recalibrated_v1_strict.jsonl`
- `gold_recalibrated_changes.jsonl`
- `gold_recalibrated_20260904.report.json`
- `judgements_full.jsonl`

本地复核脚本：[gold_recalibrate.py](${ASA_ROOT}/.codex_tmp/analysis/gold_recalibrate.py)

## 使用边界

strict 数据可以作为重新训练/评测的候选输入，但仍应保留 GLM 原始理由和
变更清单。被判 unsupported 的 fact 不应直接改写成“正确答案”；动作翻转只
表示当前证据不足，后续仍需独立验证 `retrieve_more` 与 `abstain` 的标签。
