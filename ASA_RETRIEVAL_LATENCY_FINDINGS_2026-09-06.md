# ASA 检索延迟与召回评审（2026-09-06）

## 测试范围

241 上使用已保存的 7 条端到端追踪，比较四种首轮检索配置。测试只读，不修改生产配置。

## 结果

| 配置 | 均值 | P95 | Top-3 Jaccard（相对当前） | Top-10 Jaccard |
|---|---:|---:|---:|---:|
| full_current | 18.73s | 20.31s | 1.000 | 1.000 |
| compact_graph_query | 9.76s | 11.77s | 0.743 | 0.805 |
| compact_graph_query_no_scoped_search | 9.63s | 11.55s | 0.600 | 0.633 |
| no_chapter_expansion | 7.02s | 7.65s | 0.486 | 0.298 |

原始完整结果：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/audits/retrieval_latency_ablation_20_20260906.json
```

## 判断

1. 当前主要延迟来自图扩展/章节二次检索，而不是 BGE reranker 本身。
2. 完全关闭章节扩展会显著改变候选集合，不能作为默认优化。
3. 压缩图查询可以把首轮检索延迟降低约 48%，同时保留约 80% 的 Top-10 候选重合，是目前最值得做 A/B 的折中方案。
4. `compact_graph_query_no_scoped_search` 比 `compact_graph_query` 只快约 0.14s，却额外损失 Top-3/Top-10 重合，因此不建议关闭 scoped chapter search。
5. 上述耗时仅为首轮检索，不包含回答生成；在“单题 GPU 20s”预算下，生产方案必须采用并行召回、缓存和超时降级。

## 建议实施顺序

1. 先在旁路配置启用 compact graph query，保留 scoped chapter search，做 79 条固定集的召回与事实正确率 A/B。
2. 若 Recall@20 不下降超过 1-2pp，再考虑作为默认配置。
3. 将图扩展查询构建改为结构化短字段（实体、章节、事件词），避免把整段证据拼进 BM25 查询。
4. 对同一问题的 embedding、BM25、章节过滤结果做缓存；召回并行化，给图扩展设置 2-3s 超时，超时直接使用首轮融合结果。
5. 重新测量完整端到端耗时，确认生成阶段仍能留出预算。

## 不确定性

- 当前延迟样本只有 7 条，需在 79 条固定集及若干自拟问题上复测。
- Qwen3-Embedding-0.6B 的召回已优于当前 BGE（200 条：Recall@20 82.5% vs 78.5%），但其在线编码延迟尚未完成同口径测量。
- Qwen3-Embedding-4B 在 100 条测试上 Recall@20=80.0%、MRR=0.527，未显示优于 0.6B；模型加载约 299s，不适合在线动态加载。
