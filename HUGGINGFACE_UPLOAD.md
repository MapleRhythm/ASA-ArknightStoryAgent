# 微调权重上传到 Hugging Face

建议把模型权重和大索引放到 Hugging Face，不要提交到 GitHub。

## 1. 登录

```bash
pip install -U huggingface_hub
huggingface-cli login
```

也可以使用环境变量：

```bash
export HF_TOKEN="hf_..."
```

## 2. 上传 4B LoRA

发布版部署路径：

```text
model/lora/asa-arknightstoryagent-4b-lora/
```

当前发布内容来自本地训练产物：

```text
model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered/
  adapter_config.json
  adapter_model.safetensors
  chat_template.jinja
  tokenizer_config.json
  tokenizer.json
```

默认发布用户名为 `MapleRhythm`，稳定仓库名为：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

当前版本标记：`20260607-cutoff6656`。

创建仓库并上传：

```bash
huggingface-cli repo create asa-arknightstoryagent-4b-lora \
  --type model \
  --private

huggingface-cli upload MapleRhythm/asa-arknightstoryagent-4b-lora \
  model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered \
  . \
  --repo-type model
```

建议只上传 adapter 运行所需文件和模型卡，不上传 `trainer_state.json`、`training_args.bin`、训练曲线图等本地训练细节。

部署机下载：

```bash
huggingface-cli download MapleRhythm/asa-arknightstoryagent-4b-lora \
  --local-dir model/lora/asa-arknightstoryagent-4b-lora
```

如果你希望公开发布，先确认基座模型、LoRA 数据和游戏文本衍生数据的许可都允许公开分发。默认建议保持 private。

## 3. 上传合并后的 GGUF

本地文件：

```text
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

上传：

```bash
huggingface-cli repo create asa-arknightstoryagent-4b-gguf \
  --type model \
  --private

huggingface-cli upload MapleRhythm/asa-arknightstoryagent-4b-gguf \
  model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf \
  qwen3.5-4b-lora-merged-q4_k_m.gguf \
  --repo-type model
```

下载：

```bash
huggingface-cli download MapleRhythm/asa-arknightstoryagent-4b-gguf \
  qwen3.5-4b-lora-merged-q4_k_m.gguf \
  --local-dir model/gguf
```

## 3.1 运行时 LoRA GGUF

CPU 本地发布版默认不使用运行时 LoRA GGUF，而是使用已合并 LoRA 的 GGUF：

```text
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

注意：本次已上传的是 GPU/vLLM 用 HF LoRA adapter。CPU GGUF 需要在确认合并到同一 `20260607-cutoff6656` 权重后再单独上传，否则不要把旧 GGUF 标为最新版。

## 4. 上传微调 reranker

本地目录：

```text
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
```

上传：

```bash
huggingface-cli repo create asa-evidence-chain-reranker \
  --type model \
  --private

huggingface-cli upload MapleRhythm/asa-evidence-chain-reranker \
  model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch \
  . \
  --repo-type model
```

下载：

```bash
huggingface-cli download MapleRhythm/asa-evidence-chain-reranker \
  --local-dir model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch
```

## 5. 可选：上传预构建索引

如果确认许可允许分发衍生索引，可以打包上传：

```bash
tar -czf arknights_story_indexes.tar.gz indexes/arknights_story indexes/arknights_story_minirag_v3

huggingface-cli repo create asa-arknightstoryagent-indexes \
  --type dataset \
  --private

huggingface-cli upload <你的HF用户名>/asa-arknightstoryagent-indexes \
  arknights_story_indexes.tar.gz \
  arknights_story_indexes.tar.gz \
  --repo-type dataset
```

建议默认设为 private。公开发布前需要确认原始游戏数据和衍生索引的再分发权限。

## 6. 建议写在 Hugging Face Model Card 里

- 基座模型名称和许可证。
- 权重类型：LoRA、合并 HF 权重或 GGUF。
- 用途：明日方舟剧情问答，答案应基于检索证据。
- 运行仓库地址和本地期望路径。
- 游戏文本与衍生索引的许可说明。
- 限制：模型可能幻觉，生产环境应显示引用证据。
