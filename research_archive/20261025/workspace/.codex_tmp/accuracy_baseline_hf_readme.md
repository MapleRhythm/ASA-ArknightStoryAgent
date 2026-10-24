---
library_name: peft
license: other
base_model: Qwen/Qwen3.5-4B
tags:
- peft
- lora
- text-generation
- zh
- rag
- arknights
pipeline_tag: text-generation
---

# ASA ArknightStoryAgent 4B LoRA

这是 ASA（ArknightStoryAgent）当前本地 Qwen3.5-4B 推理使用的 LoRA adapter。`accuracy-baseline-20260826` revision 对应 GitHub 项目的同名 tag。

## 基线版本

- Base model：`Qwen/Qwen3.5-4B`
- Adapter：`teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2`
- 训练方式：在已有 SFT / SODA adapter 基础上使用 teacher-scored KTO 数据训练 2 epochs
- Learning rate：`8e-7`
- PEFT / LoRA；本仓库不是可独立运行的完整模型

推荐与 [MapleRhythm/ASA-ArknightStoryAgent](https://github.com/MapleRhythm/ASA-ArknightStoryAgent) 的 `accuracy-baseline-20260826` tag 配套使用。该 runtime 使用 Qwen3-Embedding-0.6B 1024 维 dense 索引、多 lane BM25、BGE reranker、每轮最多 4 个 query 和 12 条完整 prompt 证据；默认关闭 MiniRAG 与其他图扩展。

## 小样本回归快照

在 8 道未用于仓库训练的新题上：严格正确 4/8，宽松可用 6/8，明确幻觉 0/8，错误弃答 2/8，平均延迟 52.616 秒。该集合很小，只用于冻结开发回归点，不应被视为正式 benchmark。

## 已知限制

- 复合子问题可能因 query / evidence 配额不足而漏掉关键证据。
- 模型偶尔会把引用格式问题误判为需要继续检索。
- 对原文没有明说某事实的否定性回答偏保守。
- Adapter 面向证据约束的《明日方舟》剧情 RAG，不建议脱离检索链路作为通用聊天模型使用。

## 加载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3.5-4B"
adapter = "MapleRhythm/asa-arknightstoryagent-4b-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter, revision="accuracy-baseline-20260826")
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter, revision="accuracy-baseline-20260826")
```

使用本模型前请同时确认《明日方舟》文本数据和基座模型各自的许可与使用条件。
