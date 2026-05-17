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

本地目录：

```text
model/lora/asa-arknightstoryagent-4b-lora/
  adapter_config.json
  adapter_model.safetensors
  tokenizer_config.json
  tokenizer.json
```

创建仓库并上传：

```bash
huggingface-cli repo create asa-arknightstoryagent-4b-lora \
  --type model \
  --private

huggingface-cli upload <你的HF用户名>/asa-arknightstoryagent-4b-lora \
  model/lora/asa-arknightstoryagent-4b-lora \
  . \
  --repo-type model
```

部署机下载：

```bash
huggingface-cli download <你的HF用户名>/asa-arknightstoryagent-4b-lora \
  --local-dir model/lora/asa-arknightstoryagent-4b-lora
```

如果你希望公开发布，先确认基座模型、LoRA 数据和游戏文本衍生数据的许可都允许公开分发。

## 3. 上传合并后的 GGUF

本地文件：

```text
model/gguf/qwen3.5-4b-q4_k_m.gguf
```

上传：

```bash
huggingface-cli repo create asa-arknightstoryagent-4b-gguf \
  --type model \
  --private

huggingface-cli upload <你的HF用户名>/asa-arknightstoryagent-4b-gguf \
  model/gguf/qwen3.5-4b-q4_k_m.gguf \
  qwen3.5-4b-q4_k_m.gguf \
  --repo-type model
```

下载：

```bash
huggingface-cli download <你的HF用户名>/asa-arknightstoryagent-4b-gguf \
  qwen3.5-4b-q4_k_m.gguf \
  --local-dir model/gguf
```

## 4. 上传微调 reranker

本地目录：

```text
model/reranker/bge-reranker-v2-m3-evidence-chain-answerability/
```

上传：

```bash
huggingface-cli repo create asa-evidence-chain-reranker \
  --type model \
  --private

huggingface-cli upload <你的HF用户名>/asa-evidence-chain-reranker \
  model/reranker/bge-reranker-v2-m3-evidence-chain-answerability \
  . \
  --repo-type model
```

下载：

```bash
huggingface-cli download <你的HF用户名>/asa-evidence-chain-reranker \
  --local-dir model/reranker/bge-reranker-v2-m3-evidence-chain-answerability
```

## 5. 可选：上传预构建索引

如果确认许可允许分发衍生索引，可以打包上传：

```bash
tar -czf arknights_story_indexes.tar.gz indexes/arknights_story indexes/arknights_story_minirag

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
