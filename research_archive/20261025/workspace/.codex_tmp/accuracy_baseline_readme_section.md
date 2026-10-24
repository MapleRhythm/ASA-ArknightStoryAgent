## 2026-08-26 正确率基线冻结版

这一版先固定一套可回归的正确率基线，再继续处理复合问题漏召回和错误弃答。对应 Git tag 为 `accuracy-baseline-20260826`，开发分支使用 `configs/runtime_accuracy_baseline_20260826.json`；本地生成 adapter 发布在 [MapleRhythm/asa-arknightstoryagent-4b-lora](https://huggingface.co/MapleRhythm/asa-arknightstoryagent-4b-lora)，HF revision 为 `accuracy-baseline-20260826`。

冻结协议：

- 基座为 `Qwen/Qwen3.5-4B`，LoRA 为 `teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2`。
- dense 旁路使用 `Qwen3-Embedding-0.6B` 的 1024 维索引；sparse 使用 char / 领域词 / 字段精确匹配多 lane BM25，并在融合时保留 dense/sparse 来源配额。
- 每轮最多 4 个检索 query，保留用户原问题，不做代词改写。
- reranker 对融合后的 120 个候选打分，保留 32 个候选；结论 prompt 选择 12 条完整证据，不截断单条证据或证据总文本。
- MiniRAG、章节隔离、storyline scope、neighbor expansion 和 same-story sweep 默认关闭；当前消融中这些扩展没有稳定提高召回，且会引入额外延迟和噪声。
- 最多两轮召回，使用 minimal conclusion prompt 与 strict grounding；此配置优先正确率与可审计性，尚未达到 20 秒延迟目标。

8 道未用于仓库训练的新题快照如下。它只用于记录当前回归点，不是正式 benchmark，也不能代表全集表现：

| 生成端 | 严格正确 | 宽松可用 | 明确幻觉 | 错误弃答 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本地 Qwen3.5-4B + 当前 LoRA | 4/8 | 6/8 | 0/8 | 2/8 | 52.616s |
| DeepSeek API 对照 | 4/8 | 5/8 | 1/8 | 1/8 | 11.904s |

当前主要问题不是“4B 看见直接证据也不会作答”，而是复合子问题的 query / evidence coverage、低质量 `retrieve_more`、把 `quote_not_found` 错当成缺证，以及对“原文未说明”的否定性回答判断不足。两道错误弃答中，一道关键证据位于候选第 23/24 而未进 prompt，另一道关键 chunk 未进入最终候选池。后续修复会优先保持这套冻结协议可回归，并使用不同剧情的新题检查泛化。

公开复现示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.python_packages/train:src \
python scripts/run_cpu_inference.py \
  --runtime-config configs/runtime_accuracy_baseline_20260826.json \
  --answer-only \
  "博士为什么要关闭全舰防御系统"
```

