# OpenClaw + Lobster Cluster

本目录保存本项目的 OpenClaw 专家 agent 配置与 Lobster 工作流。

## 专家 Agent

- `sft-data-curator`：生成、清洗、划分 SFT 数据集
- `lora-trainer`：生成 `LLaMA-Factory` 微调配置与执行方案
- `rag-retrieval-optimizer`：优化剧情 RAG 检索与评测

对应 workspace：

- `.openclaw-agents/sft-data-curator/`
- `.openclaw-agents/lora-trainer/`
- `.openclaw-agents/rag-retrieval-optimizer/`

## Lobster 工作流

- `workflows/sft_data_curator.lobster`
- `workflows/lora_trainer.lobster`
- `workflows/rag_retrieval_optimizer.lobster`
- `workflows/arknights_dev_cluster.lobster`

## 运行方式

单个专家：

```bash
./.openclaw-cli/bin/lobster run --mode tool --file lobster/workflows/sft_data_curator.lobster
./.openclaw-cli/bin/lobster run --mode tool --file lobster/workflows/lora_trainer.lobster
./.openclaw-cli/bin/lobster run --mode tool --file lobster/workflows/rag_retrieval_optimizer.lobster
```

整组调用：

```bash
./.openclaw-cli/bin/lobster run --mode tool --file lobster/workflows/arknights_dev_cluster.lobster
```

自定义提示：

```bash
./.openclaw-cli/bin/lobster run \
  --mode tool \
  --file lobster/workflows/sft_data_curator.lobster \
  --args-json '{"message":"为澄闪语气与剧情事实分离设计 SFT 数据结构，并产出 data/sft 的目录规划"}'
```

## 配置片段

仓库内不保存你的 OpenClaw 凭证。可复用的 agent 配置片段位于：

- `config/openclaw.agents.fragment.json`
