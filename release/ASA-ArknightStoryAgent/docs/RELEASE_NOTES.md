# Release Notes

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


## 20260607-cutoff6656

当前 GPU LoRA 发布仓库：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

本地训练产物：

```text
model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered
```

### Runtime

- GPU 默认 LoRA 路径保持为 `model/lora/asa-arknightstoryagent-4b-lora/`，部署脚本不暴露训练产物长路径。
- 默认 `max_tokens` 调整为 `1536`，conclusion 生成在 runtime 内使用 `1536-2048` 的保护区间，降低 JSON 截断。
- 默认 `answer_grounding_mode` 调整为 `quote`，conclusion 输出按 grounded action schema 解析。
- 发布版 runtime 已同步 JSON 截断恢复、`supported_facts` 压缩、quote 80 字上限和 max 6 facts / max 2 refs per fact 的适配。
- GPU 默认关闭 web context，避免发布环境依赖外部网页。

### Evaluation Snapshot

训练环境中的 `eval50 + hard10` 快照：

```text
outputs/eval50_hard10_mergedbase_cutoff6656_trainenv_20260607_175322
```

当时结果：

- `eval50`: 50 条，JSON errors 3，abstain_like 13，answer_directly 34。
- `hard10`: 10 条，JSON errors 1，abstain_like 3，answer_directly 6。
- 后续 runtime JSON 截断恢复 probe：4 条 truncation-prone 问题，errors 0/4。

注意：完整 `eval50 + hard10` 尚未在截断恢复 patch 后重跑。当前版本仍可能在低置信检索、证据链缺失或主观价值判断问题上外推。

### Known Limits

- `fusion_score=0`、无 dense/sparse/minirag/evidence_chain 的证据不应支撑强事实结论；答案端仍需继续用偏好数据和 verifier 改进。
- “坏事/好人/是否正确”一类主观问题需要按立场区分事实行为与道德评价。
- CPU GGUF 尚未在本次发布中重新确认由 `20260607-cutoff6656` 权重合并导出；不要把旧 GGUF 标注为当前最新版。
