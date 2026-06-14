# ASA-ArknightStoryAgent

面向《明日方舟》剧情问答的本地 / API 混合 RAG Agent。

本仓库当前分为两个层次：

- `dev-training`：开发与训练分支，保留训练、评测、数据构建、实验脚本和发布包构建目录。
- `main`：发布版分支，只保留 `release/ASA-ArknightStoryAgent/` 中的可部署内容，并以发布包内容作为仓库根目录。


## 当前架构

开发分支同时保留工程源码和发布版源码：

```text
src/goldenglow/                       # 工程分支运行源码，供训练、评测、实验脚本使用
release/ASA-ArknightStoryAgent/       # 面向用户的发布包
release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/
                                       # 发布版拆分后的最小运行源码
```


### 工程源码

`src/goldenglow/` 是开发分支的内部运行层：

- `config.py`：路径配置与 `QueryConfig`。
- `data/`：剧情文本、档案、语音和别名表构建。
- `retrieval/hybrid.py`：dense / BM25 / MiniRAG / reranker 混合召回入口。
- `retrieval/minirag.py`：活动级图索引、章节 scope、关系扩展。
- `retrieval/storyline.py`：故事线标签与 sparse scope。
- `inference/cpu_pipeline.py`：多轮 RAG 主链路，供本地模型和 API 模式复用。

### 发布版源码

`release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/` 是当前发布版运行层，已经按职责拆分：

- `data/`：故事文本解析、元数据解析、干员别名构建。
- `retrieval/`：混合检索入口、MiniRAG 入口、reranker 与故事线 scope。
- `retrieval/hybrid_components/`：dense / sparse 初召回、RRF 融合、证据链构建、query 分析、打分调整。
- `retrieval/minirag_components/`：实体抽取、图构建、关系打分、PPR 传播、章节 scope。
- `inference/pipeline/`：推理状态、轮次编排、结果生成。
- `inference/retrieval/`：推理时检索轮次、MiniRAG 扩展、rerank 组合。
- `inference/generation/`：假设生成、结论生成、prompt 渲染。
- `inference/payload/`：模型 JSON / 类 JSON 输出解析与规范化。
- `inference/evidence/`：证据准备、选择、渲染、CRAG 精炼。
- `inference/grounding/`：答案 grounding、引用检查、回退策略。
- `inference/planning/`：问题理解、实体抽取、续检索规划。
- `inference/web_context/`：可选网页上下文检索。
- `inference/model_runtime/`：llama.cpp 与 vLLM 运行器。
- `inference/common/`：共享词典、正则和文本工具。

发布版代码边界详见：

```text
release/ASA-ArknightStoryAgent/ARCHITECTURE.md
```

### 推理链路

当前标准链路：

```text
用户问题
-> hypothesis_generation
-> dense + BM25 + MiniRAG 初召回
-> MiniRAG 章节隔离 / 图扩展 / scoped second retrieval
-> 可选故事线 sparse scope / neighbor expansion / web context
-> fusion + reranker
-> prompt evidence 去重、降权、MMR / pyramid / pinning
-> conclusion_generation
-> answer_directly / retrieve_more / clarify_user / abstain
-> follow_up_hypothesis_generation 后进入下一轮，最多 2 轮召回
-> grounding / quote validation / final answer
```

达到最大轮次（2轮）后，链路会基于当前证据输出可确认部分或证据不足说明。

## 技术栈

- 基座模型：Qwen3.5 4B。
- 本地生成：`vLLM` 或 `llama.cpp`。
- 微调：LoRA + LLaMA-Factory。
- 向量模型：BGE small zh。
- 稀疏检索：BM25。
- 向量索引：FAISS。
- 图检索：MiniRAG 活动 / 章节级异构图。
- 重排器：BGE reranker 系列，按 runtime config 选择。
- API teacher：OpenAI-compatible Chat Completions / Responses API，当前常用 DeepSeek。

## 关键目录

- `data/ArknightsGameData/zh_CN/gamedata/story/`：剧情原文，当前约 5314 个文本文件、约 4358 万字符。
- `data/processed/`：训练、蒸馏、reranker 和评测中间数据。
- `indexes/arknights_story/`：documents、FAISS、BM25、别名表。
- `indexes/arknights_story_minirag/`、`indexes/arknights_story_minirag_v3/`：MiniRAG 图索引。
- `model/qwen3.5-4b/`：基座模型。
- `model/lora/`：LoRA 训练输出。
- `model/merged/`：合并后的完整模型。
- `model/gguf/`：llama.cpp 使用的 GGUF 模型。
- `model/reranker/`：证据链 reranker。
- `configs/runtime_inference_gpu.json`：开发分支本地 GPU / vLLM 默认运行配置。
- `api-mode/runtime_deepseek_api.json`：开发分支 DeepSeek API 模式配置。
- `src/config/`：LLaMA-Factory 训练配置。
- `scripts/`：训练、评测、索引构建、蒸馏、发布辅助脚本。
- `outputs/`：评测、运行轨迹、API 请求日志。
- `release/ASA-ArknightStoryAgent/`：发布版源码、配置、Web UI、部署脚本和 MiniRAG v3 图。

## 环境

开发分支推荐使用 `train` conda 环境运行训练、GPU 检索、vLLM 和 API 模式：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train
export PYTHONPATH=.python_packages/train:src
```

安装或重建 vLLM overlay：

```bash
PYTHON_OVERLAY_DIR=.python_packages/train bash scripts/install_train_vllm.sh
```

训练时常用多卡：

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

基础检索调试：

```bash
python scripts/query_retrieval.py "炎景公主一事具体是什么"
```

## 本地推理

开发分支 GPU / vLLM 推理：

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

两张 GPU 可设置 tensor parallel：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=.python_packages/train:src \
python scripts/run_cpu_inference.py \
  --runtime-config configs/runtime_inference_gpu.json \
  --tensor-parallel-size 2 \
  --answer-only \
  "炎景公主一事具体指什么"
```

常用配置项在 `configs/runtime_inference_gpu.json`：

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
- API 模式和本地链路透传同一组关键推理配置，包括 MiniRAG 章节隔离、故事线 scope、web context、MMR、pyramid、evidence pinning、conclusion prompt mode。
- JSON 任务会附加轻量 teacher system 约束；direct-answer 任务会切换到自然语言 system，避免最终答案被强行输出 JSON。
- API 请求日志默认写入 `outputs/api_mode_runs/` 或指定 `--log-dir`。

## 发布版

发布包位于：

```text
release/ASA-ArknightStoryAgent/
```

发布版 README：

```text
release/ASA-ArknightStoryAgent/README.md
```

发布版提供三套部署配置：

- `configs/runtime_gpu_reranker_qwen35_4b.json`：GPU 本地，vLLM + Qwen3.5 4B + LoRA + reranker。
- `configs/runtime_cpu_qwen35_4b_no_reranker.json`：CPU 本地，llama.cpp + 合并 LoRA GGUF，无 reranker。
- `configs/runtime_cpu_api_no_reranker.json`：CPU API，本地检索 + OpenAI 兼容 API，无 reranker。

发布版当前不是开发分支根目录；同步到 `main` 时，应让 `main` 的仓库根目录等于 `release/ASA-ArknightStoryAgent/` 的内容。

## Verifier-Aware SODA

当前推荐使用 verifier-aware SODA/KTO 修正本地 4B 的在线错误，而不是继续盲目补 SFT 或只做普通 blackbox SODA。它的核心区别是：teacher replay 之后还会经过 evidence-only API verifier 重标注，只有证据支持、action 合理的 pair 才进入训练。

### 背景与问题定义

普通 SODA 的基本思路是“学生在真实链路里犯错，API teacher 在同一个 prompt 上给出更好的输出，然后用 teacher 作为 positive、student 作为 negative 做 KTO”。这个思路能快速修正在线错误，但在剧情 RAG 任务里有几个风险：

- Teacher 可能偷用模型先验。即使 prompt evidence 不够，API teacher 也可能凭训练记忆补出看似正确的剧情答案；如果直接当 positive 训练，本地 4B 会学到“缺证据也可以答”。
- Teacher 也可能答错 action。比如应该 `retrieve_more` 时过早 `answer_directly`，或者 evidence 只支持局部结论时给出完整断言。
- Student 的负样本不一定总是负样本。有些 student 输出虽然不够漂亮，但 action 是合理的；直接全量打成 negative 会伤害模型的保守性和证据边界。
- SODA 数据来自真实链路，prompt 很长、字段复杂、round state 多，单靠人工抽查很难系统发现 unsupported answer、over-retrieve、premature answer 等模式。

Verifier-aware SODA 把“teacher 是否值得学”这件事显式拆出来：再调用一个 evidence-only verifier，只允许它根据当前 prompt evidence 判断 student / teacher 的 action、答案和引用是否成立。训练数据不再盲信 teacher，而是经过 verifier 过滤、重标注和审计。

### 方法收益

- 降低幻觉迁移：把 teacher 依赖外部知识、证据不足却强答的样本过滤掉，避免把 API 模型的幻觉蒸馏进 4B。
- 改善 action 决策：专门区分 `answer_directly`、`retrieve_more`、`clarify_user`、`abstain` 是否合理，让模型不仅学答案，也学什么时候不该答。
- 提升 KTO 信号质量：positive / negative 不再只由 teacher/student 身份决定，而由 verifier 判断谁更符合 evidence，使偏好数据更干净。
- 保留可审计性：`api_verifier_records.jsonl`、`build_summary.json` 和审计报告会记录每条样本为什么保留、丢弃或重标注，方便回溯。
- 更适合小模型：4B 容量有限，脏 positive 对行为边界伤害很大；先清洗再 KTO，通常比继续堆 SFT 更稳定。
- 支持分阶段扩展：可以先小样本 verifier smoke test，再扩大到 eval50、hard question pool 或三卡分片，不需要一次性押大规模训练。

### 端到端流程

端到端流程分六步：

1. 准备问题集。通常来自 eval50、hard question pool、人工失败样本或线上错误样本。
2. Student rollout。当前本地 4B 按真实 RAG pipeline 运行，记录每轮 prompt、retrieval trace、学生输出和最终答案。
3. Teacher replay。API teacher 在同一个 runtime prompt 上生成候选输出，形成 teacher positive / student negative 的初始 pair。
4. Evidence-only verifier。verifier 只看 prompt evidence，不允许使用外部剧情知识，判断 student / teacher 的 action 和答案是否被证据支持。
5. KTO 数据重标注。脚本根据 verifier 判定保留、丢弃或改写 pair，输出 LLaMA-Factory 可训练数据。
6. 审计和训练。先看 verifier 统计和失败样本，再用 verifier-aware 数据启动 KTO。

### Blackbox Rollout 前置阶段

普通 SODA rollout 负责记录当前学生在真实 pipeline 中遇到的 prompt、学生输出和 teacher replay 输出，是 verifier-aware SODA 的前置数据来源。

入口：

- `scripts/generate_soda_blackbox_distillation.py`
- `scripts/run_soda_blackbox_distill.sh`
- `src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml`

机制：

1. 当前本地 4B 按真实 pipeline 运行。
2. 脚本记录每一次 runtime prompt 和学生输出。
3. API teacher 对同一个 prompt 生成输出。
4. teacher 输出作为 KTO positive，student 输出作为 KTO negative。
5. 导出 LLaMA-Factory ShareGPT KTO 数据。

小样本：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_SAMPLE=50 \
GEN_CUDA_VISIBLE_DEVICES=0 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_soda_blackbox_distill.sh
```

正式批量前应先抽样检查 `raw_pairs.jsonl` 和 `audit_records.jsonl`，不要直接上大规模训练。

### Evidence-Only Verifier 重标注

主要入口：

- `scripts/build_soda_api_verifier_dataset.py`：读取 SODA `audit_records.jsonl`，调用 evidence-only verifier，生成重标注后的 KTO 数据。
- `scripts/run_soda_api_verifier_v1.sh`：对已有 SODA blackbox 数据执行 verifier relabel。
- `scripts/run_soda_eval50_len1800_api_verifier_flow.sh`：单卡完整流程，包含 student rollout、teacher replay、API verifier 和审计报告。
- `scripts/run_soda_eval50_len1800_api_verifier_flow_3gpu.sh`：多卡分片流程，最后合并 verifier 数据。
- `scripts/analyze_soda_api_verifier_dataset.py`：生成 verifier-aware 数据审计报告。
- `scripts/clean_soda_api_verifier_dataset.py`：修补已确认的弱接受样本。

默认数据目录示例：

```text
data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1/
data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged/
```

已有 SODA rollout 后单独跑 verifier：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_INPUT_DIR=data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel \
SODA_VERIFIER_OUT_DIR=data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1 \
bash scripts/run_soda_api_verifier_v1.sh
```

从问题集开始跑完整 verifier-aware flow：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_FLOW_LIMIT=50 \
SODA_FLOW_VERIFIER_LIMIT=50 \
SODA_FLOW_GEN_CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_soda_eval50_len1800_api_verifier_flow.sh
```

三卡分片：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_PARALLEL_GPUS=0,1,2 \
SODA_PARALLEL_RUN_ID=v2_scoped_sweep_soda_lora \
bash scripts/run_soda_eval50_len1800_api_verifier_flow_3gpu.sh
```

对应训练配置示例：

- `src/config/llama_factory_soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_config.yaml`
- `src/config/llama_factory_soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_config.yaml`

审计时重点看：

- `api_verifier_records.jsonl`：每条 pair 的 verifier 判定。
- `build_summary.json`：最终 KTO 样本数量和过滤数量。
- `api_verifier_audit_report.md` 或 `outputs/soda_flow_reports/*_api_verifier_audit.md`：action 错误、unsupported answer、teacher prior knowledge risk 等统计。

### Verifier 消融实验

消融协议见 `docs/soda_verifier_ablation.md`。当前实验采用 paired label-construction ablation：固定 student rollout、teacher replay、prompt evidence 和 verifier records，只替换标签构造策略，比较 Raw SODA、output-only heuristic 与 verifier-aware relabel。

生成数据质量报告：

```bash
python scripts/analyze_soda_verifier_ablation.py
```

严格模型效果对照：

```bash
RUN_TEACHER_SCORE=1 \
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
GPUS=0,1 \
bash scripts/run_soda_verifier_model_ablation.sh
```

GPU 驱动不可用时，可先跑 CPU/GGUF hard10 fallback：

```bash
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
RUN_EVAL50=0 \
RUN_HARD=1 \
bash scripts/run_soda_verifier_cpu_gguf_ablation.sh
```

当前数据质量消融显示，`eval50_plus_extra300_qc_v2` 中 Raw SODA 的 teacher-as-chosen 正样本有 `108/500 = 21.60%` 被 verifier 标为 unsafe，output-only heuristic 全部漏检；verifier-aware relabel 将 chosen action 重建为 verifier correct action，当前 action mismatch 为 `0/500`。严格模型效果对照会在两套 LoRA 同运行时输出后，再用 evidence-only scorer 计算 unsupported、premature、over-abstain 和 paired score delta。

报告还纳入一组 hard10 历史同题 probe：Raw SODA `outputs/retrieval_eval/soda_blackbox_hard10_20260530.jsonl` 与 verifier-aware SODA `outputs/eval_soda_api_verifier_v2/hard10_answers.jsonl` 问题完全同序，Raw abstain-like 为 `3/10`，verifier-aware 为 `1/10`。该结果只能作为补充行为证据，因为两次运行的 web context、eager 和 batch-token 设置不同；严格结论仍以 `run_soda_verifier_model_ablation.sh` 生成的同运行时结果为准。

## SFT / KTO 训练

历史 SFT 和 API-grounded 数据生成脚本仍保留：

- `scripts/generate_api_grounded_sft_from_retrieval.py`
- `scripts/generate_online_teacher_chain_sft.py`
- `scripts/merge_online_teacher_chain_sft_datasets.py`

常用训练配置示例：

- `src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml`
- `src/config/llama_factory_soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_config.yaml`
- `src/config/llama_factory_opd_kto_full_chain_sample500_v3_config.yaml`
- `src/config/llama_factory_soda_targeted_human_20260606_v3_200_current_chain_kto_mergedbase_rank8_cutoff6656_config.yaml`

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

### dev 分支和 main 分支是什么关系？

`dev-training` 是工程分支，包含训练数据、实验脚本、评测和发布包目录。`main` 是发布分支，应该只包含发布包根目录内容。不要把整个开发分支 cherry-pick 到 `main`。

### vLLM OOM 怎么处理？

降低以下参数：

- `generator.vllm.gpu_memory_utilization`
- `generator.vllm.max_model_len`
- `inference.prompt_conclusion_evidence_max_total_chars`
- `retrieval.rerank_top_k`

同时检查 `nvidia-smi` 是否已有旧进程占显存。

### API 返回空 content 怎么处理？

DeepSeek 等 reasoning 模型可能把内容放到 `reasoning_content`。API runner 已做重试，要求最终内容写入 assistant `content`。

### JSON 字段漂移怎么排查？

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
- 发布版更新先落到 `release/ASA-ArknightStoryAgent/`，再同步到 `main`。
- 不要把 API key、私有 endpoint token、本地密钥或上传说明写进仓库。
