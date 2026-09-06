# v3 语义审计分析（2026-09-06）

审计目录（241 挂载盘）：

`/mnt/wwn-0x5000cca295f3594f/zhb_asa/relevance_data_20260906/audit_full_v3_20260906/`

## 结果

| 指标 | clean-SFT | relevance-aware |
|---|---:|---:|
| 有效请求 | 76/79 | 78/79 |
| answer_directly | 62 | 68 |
| complete / answer_directly | 25/62 (40.3%) | 31/68 (45.6%) |
| 含非 entailed fact 的回答 | 24/62 (38.7%) | 21/68 (30.9%) |
| 含关键未支持主张的回答 | 17/62 (27.4%) | 15/68 (22.1%) |
| 含无关 fact 的回答 | 9/62 (14.5%) | 2/68 (2.9%) |
| 证据不足却 answer_directly | 4/62 | 6/68 |

`fact_support` 的 unsupported 比例不能直接当幻觉率：一个 partial fact
可能遗漏主体、原因、时间或第二分句。因此验收使用回答级“任一非 entailed”
和“任一关键未支持主张”两个指标。

## 泄漏与配对限制

relevance 微调的 short4096 训练集为 32 行、19 个问题族。它与本次 79 条
评测有 34 条规范化问题重合。按问题是否重合分组：

- 重合组：clean 12/34 complete，relevance 18/34；
- 未重合组：clean 13/42 complete，relevance 13/44。

因此总体 `25→31` 的 complete 增益主要来自训练重合组，不能作为泛化提升。
这批 79 条只能作为回归/诊断集，不得继续生成训练样本或作为独立盲测。

## 决策

relevance-aware 版本确实减少了无关扩写，并略微增加完整回答，但回答级关键
未支持率仍为 22.1%，未达到 20% 门槛；同时证据不足时抢答从 clean 的
4/62 增至 6/68。因此不切生产，也不以此结果启动 RLVR。

下一步应使用完全不重合的 family-held-out 盲测，并优先修正：

1. evidence 到 prompt 的集合覆盖；
2. evidence-sufficient 时的集合级校验；
3. 删除坏 fact 后必须重新做 set-level 审计，不能直接将半答案标为 SFT 正例；
4. `retrieve_more`/`abstain` 的动作边界，避免为了降低幻觉率大量弃答。

