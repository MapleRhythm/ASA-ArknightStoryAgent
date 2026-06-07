# 配置文件说明

三个推荐配置：

```text
configs/runtime_gpu_reranker_qwen35_4b.json       # GPU，本地 Qwen3.5 4B + reranker
configs/runtime_cpu_qwen35_4b_no_reranker.json   # CPU，本地已合并 LoRA 的 Qwen3.5 4B GGUF，无 reranker
configs/runtime_cpu_api_no_reranker.json         # CPU，本地检索 + 远程 API，无 reranker
```

配置文件分三段：`retrieval`、`generator`、`inference`。

## retrieval

```json
{
  "device": "cpu",
  "enable_reranker": false,
  "reranker_model_path": "model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch",
  "dense_top_k": 80,
  "sparse_top_k": 80,
  "enable_minirag": true,
  "minirag_index_path": "indexes/arknights_story_minirag_v3/graph.json",
  "fusion_top_k": 50,
  "rerank_top_k": 20,
  "rerank_batch_size": 4,
  "reranker_max_length": 1024,
  "enable_neighbor_expansion": false
}
```

字段说明：

- `device`：检索侧设备，`cpu` 或 `cuda`。
- `enable_reranker`：是否加载 reranker。CPU/API 版本默认关闭，GPU 版本默认开启。
- `reranker_model_path`：reranker 模型目录，仅 `enable_reranker=true` 时需要。
- `dense_top_k`：FAISS dense 召回数量。
- `sparse_top_k`：BM25 召回数量。
- `enable_minirag`：是否启用 MiniRAG 图召回。
- `minirag_index_path`：MiniRAG 图路径，发布版已内置。
- `fusion_top_k`：dense/BM25/MiniRAG 融合后保留的候选数。
- `rerank_top_k`：最终进入证据选择的候选数；无 reranker 时仍作为融合截断上限使用。
- `rerank_batch_size`：reranker batch size，仅开启 reranker 时影响明显。
- `reranker_max_length`：reranker 输入长度。
- `enable_neighbor_expansion`：是否启用 chunk 邻接扩展；CPU 默认关闭。

## generator

本地 GPU/vLLM：

```json
{
  "backend": "vllm",
  "ctx_size": 10000,
  "max_tokens": 1536,
  "temperature": 0.2,
  "top_p": 0.9,
  "repeat_penalty": 1.05,
  "vllm": {
    "base_model_path": "model/qwen3.5-4b",
    "lora_path": "model/lora/asa-arknightstoryagent-4b-lora",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.52,
    "max_model_len": 10000,
    "dtype": "auto",
    "max_num_batched_tokens": 4096,
    "enforce_eager": true
  }
}
```

本地 CPU/llama.cpp：

```json
{
  "backend": "llama.cpp",
  "ctx_size": 8192,
  "max_tokens": 512,
  "llama_cpp": {
    "llama_cli_path": "third_party/llama.cpp/build-cpu/bin/llama-completion",
    "gguf_model_path": "model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf",
    "lora_path": null,
    "device": "cpu",
    "gpu_layers": "0"
  }
}
```

CPU/API：

```json
{
  "backend": "openai_compatible_api",
  "api_base_url": "https://api.openai.com/v1/chat/completions",
  "api_key_env": "OPENAI_API_KEY",
  "model": "gpt-4.1-mini",
  "timeout": 120,
  "max_tokens": 4096,
  "response_format_json": true
}
```

字段说明：

- `backend`：`vllm`、`llama.cpp`、`openai_compatible_api` 或 `responses_api`。
- `ctx_size` / `max_model_len`：上下文窗口。
- `max_tokens`：单次生成最大 token 数。
- `temperature` / `top_p`：采样参数。
- `repeat_penalty`：本地模型重复惩罚。
- `base_model_path`：HF 格式 Qwen3.5 4B 基座目录。
- `lora_path`：LoRA 路径。vLLM 使用 Hugging Face LoRA 目录；CPU/llama.cpp 推荐使用已合并 LoRA 的 GGUF，并设为 `null`。
- `max_num_batched_tokens` / `enforce_eager`：vLLM 运行参数。当前发布默认偏保守，优先降低显存占用和 JSON 截断风险。
- `gguf_model_path`：llama.cpp GGUF 文件。
- `api_base_url`：API endpoint。`openai_compatible_api` 使用 chat completions；`responses_api` 使用 responses。
- `api_key_env`：读取 API key 的环境变量名。
- `response_format_json`：provider 支持 JSON mode 时设为 `true`；不支持时设为 `false` 或命令行加 `--no-json-response-format`。

## inference

```json
{
  "pipeline_mode": "answer_then_retrieve_refine",
  "initial_prompt_hint": "直接回答用户问题。提示词越少越好；可以使用你已有的剧情知识，但不确定的细节不要写死。",
  "initial_answer_max_tokens": 1024,
  "refine_answer_max_tokens": 1536,
  "max_retrieval_rounds": 2,
  "prompt_evidence_top_k": 10,
  "prompt_evidence_max_chars_per_doc": 1000,
  "prompt_conclusion_evidence_max_total_chars": 12000,
  "enable_mmr": true,
  "mmr_lambda": 0.72,
  "enable_pyramid_order": true,
  "enable_crag_refinement": true,
  "crag_refine_top_sentences": 4,
  "crag_refine_max_sentences": 24,
  "self_consistency_samples": 1,
  "self_consistency_temperature": 0.7,
  "conclusion_prompt_mode": "minimal",
  "answer_grounding_mode": "quote",
  "use_model_hypothesis": true,
  "use_model_conclusion_generation": true
}
```

字段说明：

- `pipeline_mode`：API 模式可设为 `answer_then_retrieve_refine` 或 `standard`。前者先让 API 直接回答，再用初答检索证据并校正；后者走本地 4B 同款 hypothesis/conclusion 多轮 RAG。
- `initial_prompt_hint`：API 初答阶段的极简提示词。
- `initial_answer_max_tokens`：API 初答 token 上限。
- `refine_answer_max_tokens`：证据校正阶段 token 上限。
- `max_retrieval_rounds`：最多多轮检索次数。核心链路会 clamp 到最多 2 轮。
- `prompt_evidence_top_k`：最终塞给生成模型的证据数量。
- `prompt_evidence_max_chars_per_doc`：每条证据进入 prompt 的最大字符数。
- `prompt_conclusion_evidence_max_total_chars`：conclusion 阶段证据包总字符数上限。
- `enable_mmr`：是否做证据去冗余。
- `mmr_lambda`：MMR 相关性/多样性权重，越高越偏相关性。
- `enable_pyramid_order`：是否按证据结构重排。
- `enable_crag_refinement`：是否按句子精炼证据包。API/GPU 可开，纯 CPU 默认关。
- `crag_refine_top_sentences`：每个 chunk 保留的高分句数。
- `crag_refine_max_sentences`：整个证据包最多保留句数。
- `self_consistency_samples`：自一致采样次数。GPU 可设 3-5，CPU/API 建议 1。
- `conclusion_prompt_mode`：当前本地 4B 推荐 `minimal`。
- `answer_grounding_mode`：当前 LoRA 推荐 `quote`，要求答案通过结构化 facts/quotes 支撑。
- `use_model_hypothesis`：是否让模型生成首轮检索假设。
- `use_model_conclusion_generation`：是否让模型做 conclusion/retrieve_more/abstain 判断。

## 调参建议

- CPU 慢：降低 `dense_top_k`、`sparse_top_k`、`fusion_top_k`，关闭 `enable_crag_refinement`。
- 召回不够：提高 `dense_top_k`/`sparse_top_k` 到 120，保持 MiniRAG 开启。
- reranker 显存不足：降低 `rerank_batch_size` 或 `reranker_max_length`。
- API 不支持 JSON mode：把 `response_format_json` 设为 `false`。
- 生成太短：提高 `generator.max_tokens`。
