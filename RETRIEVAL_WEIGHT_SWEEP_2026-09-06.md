# Qwen0.6B dense/sparse 权重扫参（2026-09-06）

## 测试

同一 `reranker_listwise.jsonl` 的 100 条样本，Qwen3-Embedding-0.6B 旁路索引、同一 BGE reranker 和候选规模，仅改变 dense/sparse 融合权重。

## 结果

| dense weight | sparse weight | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.8 | 42% | 64% | 74% | 79% | 0.5228 |
| 0.5 | 1.0 | **48%** | 65% | **77%** | **80%** | **0.5579** |
| 1.5 | 0.5 | 44% | **68%** | 76% | 79% | 0.5443 |

原始结果：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/audits/recall_sweep_w_dense1_sparse08_100_20260906.json
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/audits/recall_sweep_w_dense05_sparse1_100_20260906.json
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/audits/recall_sweep_w_dense15_sparse05_100_20260906.json
```

## 结论

- Qwen0.6B 下 sparse-heavy（0.5/1.0）在首位命中、Recall@10/20 和 MRR 均最好；
- dense-heavy 只提高 Recall@5，未提高深层召回；
- `dense=0.5, sparse=1.0` 可作为下一轮端到端 A/B 的候选配置，但不直接覆盖生产；
- 下一轮必须补测事实正确率、premature answer、弃答率和 p50/p95 延迟；召回提升不等于幻觉率下降。
