# Release Notes

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
