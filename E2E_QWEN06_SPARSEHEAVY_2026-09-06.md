# Qwen0.6B sparse-heavy 端到端对照（2026-09-06）

## 配置

在 241 使用 Qwen3-Embedding-0.6B 旁路索引和 BGE reranker，设置
`dense_weight=0.5, sparse_weight=1.0`，其余参数与 intended baseline
一致（10 条证据、单条 1000 字、MiniRAG 关闭、同一 8 条盲测）。

原始产物：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/runs/e2e_full_evidence_codex_20260906/qwen06_quote_sparseheavy/results_sparseheavy.jsonl
```

## 观察

- 平均耗时 21.78s（首题冷启动 71.27s，后 7 题平均约 14.71s）；
- 古米训练题的第三击和照顾方式恢复得更好；
- 霜星题保持正确，凯尔希人名/日期题保持克制弃答；
- 房东身份、浊心斯卡蒂、调岗动机仍出现明显无证据推断；
- 雷德题只覆盖到部分“示威者经历”，多子问仍不完整。

## 判断

1. sparse-heavy 在离线 Recall@1/10/20 与 MRR 上优于默认权重，但端到端事实正确率没有同步提升。
2. 当前瓶颈已经不是单纯 dense/sparse 权重，而是：
   - 固定 top-k 证据无法覆盖每个子问题；
   - 4B 会把相邻剧情或角色关系拼成答案；
   - quote/词法校验不能证明 claim 被证据蕴含。
3. 不能因为 Recall 提升而切生产，也不能把 sparse-heavy 当作最终方案。

## 后续实验优先级

1. evidence-id/Exx 输出协议：模型只输出 `fact + evidence_ids`，程序负责引用重建，消除 quote 格式失败；
2. 训练加入身份、因果、多子问覆盖和“证据未写明” hard negatives；
3. 对 prompt evidence 做子问题覆盖选择，而非简单 top-k；
4. 用 GLM/本地 verifier 做离线 claim-level entailment 审计，再决定 SFT/KTO/RLVR；
5. 重新跑 79 条固定集和自拟题，只有幻觉率低于 20% 且延迟达标才考虑生产。
