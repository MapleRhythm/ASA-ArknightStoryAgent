## accuracy-baseline-20260826

该 revision 固定 2026-08-26 正确率基线，配套开发分支 tag `accuracy-baseline-20260826` 与 Hugging Face LoRA revision `accuracy-baseline-20260826`。

### Runtime

- 本地生成模型：`Qwen/Qwen3.5-4B` + `teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2`。
- 检索主链路：Qwen3-Embedding-0.6B 1024 维 dense 旁路 + 多 lane BM25 + 来源配额融合 + BGE reranker。
- 不做代词改写；每轮最多 4 条 query；prompt 使用 12 条完整证据，不做单文档或总字符截断。
- 默认关闭 MiniRAG、章节 scope、故事线 scope、neighbor expansion 和 same-story sweep。
- 使用 minimal conclusion prompt、strict grounding，最多两轮召回。

### Evaluation Snapshot

8 道新题小样本回归中，本地 4B 严格正确 4/8、宽松可用 6/8、明确幻觉 0/8、错误弃答 2/8，平均 52.616 秒；DeepSeek 对照分别为 4/8、5/8、1/8、1/8 和 11.904 秒。这不是正式 benchmark，仅作为后续正确率修复的固定回归点。

### Known Limits

- 复合问题的不同子主张仍可能争抢 query 与 prompt evidence 配额。
- `quote_not_found` 仍可能触发不必要的第二轮检索。
- 本地 4B 对“现有证据足以证明原文未说明”的负事实判断偏保守。
- 当前配置优先正确率和完整证据，尚未达到单题 20 秒目标。

