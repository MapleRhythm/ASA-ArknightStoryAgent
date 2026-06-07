# ASA-ArknightStoryAgent

面向《明日方舟》剧情问答的本地 / API 混合 RAG Agent。项目目标不是泛聊天，而是用剧情原文、档案、语音和可追踪证据回答剧情事实、因果、关系、时间线与阴谋真相类问题。

核心原则：

- 答案优先基于检索证据，不能把模型记忆或二创设定写成官方剧情。
- 证据足够时应回答“可确认部分”，不要因缺少完整背景直接拒答。
- 证据不足时必须说明不足，不能把推测包装成事实。
- 本地 4B 与 API teacher 使用同一套 `CPUInferencePipeline`，差异只在生成器后端。

## 当前架构

主代码位于 `src/goldenglow/`。

- `src/goldenglow/config.py`：路径配置与 `QueryConfig`。
- `src/goldenglow/retrieval/hybrid.py`：dense / BM25 / MiniRAG / reranker 混合召回。
- `src/goldenglow/retrieval/minirag.py`：活动级图检索、章节 scope、关系扩展。
- `src/goldenglow/retrieval/storyline.py`：故事线标签与 sparse scope。
- `src/goldenglow/inference/cpu_pipeline.py`：多轮 RAG 主链路。
- `api-mode/run_api_inference.py`：API 生成器入口，复用本地检索链路。

当前标准推理流程：

```text
用户问题
-> user_question_hypothesis_generation
-> dense + BM25 + MiniRAG 召回
-> MiniRAG 章节隔离 / 图扩展 / 二次 scoped retrieval
-> 可选故事线 sparse scope、neighbor expansion、web context
-> fusion + reranker
-> prompt evidence 去重、降权、MMR / pyramid / pinning
-> conclusion_generation
-> answer_directly / retrieve_more / clarify_user / abstain
-> follow_up_hypothesis_generation 后进入下一轮，最多 2 轮召回
```

达到最大轮次后，链路会基于当前证据输出可确认部分或证据不足说明，而不是机械返回“达到检索轮次上限”。
当前核心链路会把外部传入的 `max_retrieval_rounds` clamp 到 `2`，避免第三轮召回只增加延迟。

## 技术栈

- 基座模型：`model/qwen3.5-4b`
- 本地生成：`vLLM` 或 `llama.cpp`
- 微调：LoRA + LLaMA-Factory
- 向量模型：`model/embeddings/bge-small-zh-v1.5`
- 稀疏检索：BM25
- 向量索引：FAISS
- 重排器：BGE reranker 系列，当前按 runtime config 选择
- API teacher：OpenAI-compatible Chat Completions / Responses API，当前常用 DeepSeek

## 当前本地 4B 发布模型

当前 GPU/vLLM 发布版使用稳定 LoRA 目录：

```text
model/lora/asa-arknightstoryagent-4b-lora/
```

Hugging Face 稳定仓库：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

当前内容版本为 `20260607-cutoff6656`，对应本地训练产物：

```text
model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered
```

配套 runtime 已包含 grounded action JSON 截断恢复和 quote 压缩适配。CPU GGUF 需要单独确认由同一权重重新合并导出后再标记为当前版。

## 关键目录

- `data/ArknightsGameData/zh_CN/gamedata/story/`：剧情原文。
- `indexes/arknights_story/`：documents、FAISS、BM25、别名表。
- `indexes/arknights_story_minirag/`、`indexes/arknights_story_minirag_v3/`：MiniRAG 图索引。
- `model/qwen3.5-4b/`：基座模型。
- `model/lora/`：LoRA 训练输出。
- `model/merged/`：合并后的完整模型。
- `model/reranker/`：证据链 reranker。
- `configs/runtime_inference_gpu.json`：本地 GPU / vLLM 默认运行配置。
- `api-mode/runtime_deepseek_api.json`：DeepSeek API 模式配置。
- `release/ASA-ArknightStoryAgent/`：推理发布版源码、配置、Web UI 与 MiniRAG v3 图。
- `src/config/`：LLaMA-Factory 训练配置。
- `outputs/`：评测、运行轨迹、API 请求日志。

## 环境

推荐使用 `train` conda 环境运行训练、GPU 检索、vLLM 和 API 模式：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train
export PYTHONPATH=.python_packages/train:src
```

安装或重建 vLLM overlay：

```bash
PYTHON_OVERLAY_DIR=.python_packages/train bash scripts/install_train_vllm.sh
```

训练时常用 3 卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 构建索引

构建基础检索索引：

```bash
python scripts/build_retrieval_index.py --device cpu
```

构建 MiniRAG 图索引：

```bash
python scripts/build_minirag_index.py
```

基础调试：

```bash
python scripts/query_retrieval.py "炎景公主一事具体是什么"
```

## 本地推理

GPU / vLLM 推理使用：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.python_packages/train:src \
python scripts/run_cpu_inference.py \
  --runtime-config configs/runtime_inference_gpu.json \
  --answer-only \
  "博士为什么要关闭全舰防御系统"
```

常用配置在 `configs/runtime_inference_gpu.json`：

- `retrieval.reranker_model_path`
- `retrieval.minirag_index_path`
- `retrieval.minirag_chapter_isolation`
- `retrieval.minirag_auto_second_retrieval`
- `retrieval.enable_storyline_sparse_scope`
- `inference.max_retrieval_rounds`
- `inference.prompt_evidence_top_k`
- `inference.conclusion_prompt_mode`
- `inference.answer_grounding_mode`
- `inference.web_context`

## API 模式

API 模式入口：

```bash
export DEEPSEEK_API_KEY="sk-..."

PYTHONPATH=.python_packages/train:src \
python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_deepseek_api.json \
  --answer-only \
  "真龙为什么要启动不反"
```

注意：

- DeepSeek key 使用纯 `sk-...`；如果外部记录为 `ds:sk-...`，运行脚本前应去掉 `ds:`。
- API 模式现在和本地链路透传同一组关键推理配置，包括 MiniRAG 章节隔离、故事线 scope、web context、MMR、pyramid、evidence pinning、conclusion prompt mode。
- JSON 任务会附加轻量 teacher system 约束；direct-answer 任务会切换到自然语言 system，避免最终答案被强行输出 JSON。
- API 请求日志默认写入 `outputs/api_mode_runs/` 或指定 `--log-dir`。

## SODA 半在线黑盒蒸馏

当前推荐用 SODA/KTO 修正本地 4B 的在线错误，而不是继续盲目补 SFT。

实现入口：

- `scripts/generate_soda_blackbox_distillation.py`
- `scripts/run_soda_blackbox_distill.sh`
- `src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml`

机制：

1. 当前本地 4B 按真实 pipeline 运行。
2. 脚本记录每一次 runtime prompt 和学生输出。
3. API teacher 对同一个 prompt 生成输出。
4. teacher 输出作为 KTO positive，student 输出作为 KTO negative。
5. 导出 LLaMA-Factory ShareGPT KTO 数据。

先跑小样本：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_SAMPLE=50 \
GEN_CUDA_VISIBLE_DEVICES=0 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_soda_blackbox_distill.sh
```

第一版正式建议：

```bash
SODA_SAMPLE=300 \
GEN_CUDA_VISIBLE_DEVICES=0 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_soda_blackbox_distill.sh
```

不建议一开始上千问题；先抽样检查 `data/processed/llama_factory/soda_blackbox_deepseek_v1/raw_pairs.jsonl` 和 `audit_records.jsonl`。

## SFT / KTO 训练

历史 SFT 和 API-grounded 数据生成脚本仍保留：

- `scripts/generate_api_grounded_sft_from_retrieval.py`
- `scripts/generate_online_teacher_chain_sft.py`
- `scripts/merge_online_teacher_chain_sft_datasets.py`

当前常用训练配置示例：

- `src/config/llama_factory_online_teacher_chain_v2_quality_fix3_plus_detail_conclusion_patch_v1_config.yaml`
- `src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml`
- `src/config/llama_factory_opd_kto_full_chain_sample500_v3_config.yaml`

启动训练：

```bash
DISABLE_VERSION_CHECK=1 \
PYTHONPATH=.python_packages/train:src \
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m llamafactory.cli train \
  src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml
```

## Reranker 训练与评测

主要脚本：

- `scripts/build_rank_mix_reranker_dataset.py`
- `scripts/build_rank_clean_reranker_dataset.py`
- `scripts/train_evidence_chain_reranker.py`
- `scripts/evaluate_evidence_chain_reranker.py`

训练示例：

```bash
PYTHONPATH=.python_packages/train:src \
python scripts/train_evidence_chain_reranker.py \
  --model-name-or-path model/reranker/bge-reranker-v2-m3-rank-mix-v5-warm \
  --train-file data/processed/evidence_chain_reranker/rank_mix_v6_small_patch/reranker_pairwise.jsonl \
  --output-dir model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch \
  --num-train-epochs 1 \
  --per-device-train-batch-size 8 \
  --learning-rate 2e-5 \
  --max-length 1024 \
  --bf16
```

## 评测

多轮召回 / 答案链路评测：

```bash
PYTHONPATH=.python_packages/train:src \
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_multiround_retrieval_recall.py \
  --runtime-config configs/runtime_inference_gpu.json \
  --output outputs/eval_multiround_retrieval/current_sample50.json \
  --sample 50 \
  --max-rounds 2 \
  --device cuda
```

API 对照评测可用同一脚本的 API backend，或直接用 `api-mode/run_api_inference.py` 批量跑问题文件。

## 常见问题

### vLLM OOM

降低以下参数：

- `generator.vllm.gpu_memory_utilization`
- `generator.vllm.max_model_len`
- `inference.prompt_conclusion_evidence_max_total_chars`
- `retrieval.rerank_top_k`

同时检查 `nvidia-smi` 是否已有旧进程占显存。

### API 返回空 content

DeepSeek 等 reasoning 模型可能把内容放到 `reasoning_content`。API runner 已做一次重试，要求最终内容写入 assistant `content`。

### JSON 字段漂移

优先检查：

- 当前 LoRA 是否为最新 SFT/SODA 版本。
- `conclusion_prompt_mode` 是否与训练数据一致。
- API / local 是否使用同一套 runtime config。
- 训练数据里是否混入 extra fields，如 `confidence`、`decision`、`slot_values`。

## 开发规则

- 不要为单个问题、角色或活动写硬编码修复。
- 任何检索策略变化都应配置化，并保留 trace 方便复盘。
- 修改 retrieval / reranker / prompt / 数据清洗后至少跑 `py_compile` 和小样本评测。
- 训练数据必须保留 raw、clean、summary、rejected/failed 记录。
- 不要把 API key、私有 endpoint token 或本地密钥写进仓库。
