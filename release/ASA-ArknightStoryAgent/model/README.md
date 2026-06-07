# 模型目录

本发布版不包含模型权重。

推理时推荐目录结构：

```text
model/embeddings/bge-small-zh-v1.5/
model/reranker/bge-reranker-v2-m3/
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

当前 GPU LoRA 稳定下载仓库：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

当前内容版本为 `20260607-cutoff6656`。CPU GGUF 需要单独确认已经由同一权重合并导出后再发布。

公开 embedding / reranker 基座模型可用：

```bash
python scripts/download_models.py
```

微调权重上传和下载方式见根目录的 `HUGGINGFACE_UPLOAD.md`。
