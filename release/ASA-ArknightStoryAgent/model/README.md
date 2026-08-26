# 模型目录

本发布版不包含模型权重。

推理时推荐目录结构如下，通常不需要手工创建，直接使用下载脚本即可：

```text
model/embeddings/Qwen3-Embedding-0.6B/
model/reranker/bge-reranker-v2-m3/
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

默认下载仓库可通过环境变量或命令行参数覆盖。默认发布仓库：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

2026-08-26 正确率冻结版使用 Hugging Face revision：

```text
accuracy-baseline-20260826
```

常用下载命令：

```bash
# CPU API：只下载 embedding
python scripts/download_models.py --runtime cpu-api

# CPU 本地：下载 embedding 和合并 LoRA 的 GGUF
python scripts/download_models.py --runtime cpu-local

# GPU 本地：下载 embedding、Qwen3.5 4B、LoRA 和微调 reranker
python scripts/download_models.py --runtime gpu
```

如需使用自己的镜像或仓库：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --runtime gpu --lora-repo MapleRhythm/asa-arknightstoryagent-4b-lora
```
