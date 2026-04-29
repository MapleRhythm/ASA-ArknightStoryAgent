# ASA-ArknightStoryAgent

一个面向《明日方舟》剧情问答的本地 Agent 项目。核心目标不是泛知识聊天，而是基于剧情证据进行中文问答，并在表达层面注入较轻的“澄闪”风格。

当前主链路：

1. 用户问题进入推理脚本
2. 模型生成初始 hypothesis
3. 触发混合检索：稠密召回 + 稀疏召回 + 融合 + 可选 reranker
4. 模型判断是否需要继续检索
5. 若需要，则生成 follow-up hypothesis 并继续召回
6. 基于最终证据生成答案

## 技术栈

- 基座模型：`Qwen3.5-4B`
- 微调方式：`LoRA`
- 主训练框架：`LLaMA-Factory`
- 向量模型：`bge-small-zh-v1.5`
- 稀疏检索：`BM25`
- 向量索引：`FAISS-CPU`
- 重排：交叉编码器 reranker（可关闭）
- 推理框架：`llama.cpp`
- 调度逻辑：原生 `Python`

## 目录概览

- `data/ArknightsGameData/zh_CN/gamedata/story/`
  明日方舟剧情原文
- `data/processed/sft_data/`
  SFT 数据、补充中间能力数据、合并后的训练数据
- `indexes/arknights_story/`
  检索索引、文档和 BM25 产物
- `model/qwen3.5-4b/`
  基座模型
- `model/lora/`
  LoRA 训练输出
- `model/gguf/`
  GGUF 推理模型
- `scripts/`
  数据生成、索引构建、推理、评测脚本
- `scripts/llama_factory/`
  LLaMA-Factory 训练与评测入口
- `src/goldenglow/retrieval/`
  检索与重排实现
- `src/goldenglow/inference/`
  推理主链路
- `configs/`
  运行时配置
- `src/config/`
  训练配置

## 环境建议

建议至少准备两个 conda 环境：

- `train`
  用于数据生成、检索 GPU 测试、LLaMA-Factory 训练
- `reasoning`
  用于 CPU 推理和实际运行

如果要在 `train` 环境下跑 GPU 推理加速，可额外安装 `vLLM` overlay：

```bash
conda activate train
bash scripts/install_train_vllm.sh
```

如果仓库所在磁盘空间不足，可以把 overlay 装到别的目录：

```bash
conda activate train
PYTHON_OVERLAY_DIR=/path/to/big-disk/vllm_overlay bash scripts/install_train_vllm.sh
```

## 1. 构建检索索引

在首次运行前，先构建剧情检索索引：

```bash
python scripts/build_retrieval_index.py --device cpu
```

索引产物会输出到：

- `indexes/arknights_story/documents.jsonl`
- `indexes/arknights_story/faiss.index`
- `indexes/arknights_story/bm25_tokens.pkl`

## 2. 生成 SFT 数据

主数据集：

```bash
python scripts/generate_sft_from_teacher.py --device cuda
```

该脚本现在会直接产出与补充脚本对齐的三类 tool 样本：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

补充中间能力数据集：

```bash
python scripts/generate_prompt_supplement_from_teacher.py \
  --config configs/sft_teacher_prompt_supplement_v2.json \
  --target-total 700 \
  --max-requests 450 \
  --device cuda \
  --concurrency 4
```

补充数据集默认输出到：

- `data/processed/sft_data/prompt_supplement_v2`

## 3. 合并训练数据集

将主数据集与补充中间能力数据集合并：

```bash
python scripts/merge_sft_datasets.py \
  --base-dir data/processed/sft_data/teacher_v2 \
  --supplement-dir data/processed/sft_data/prompt_supplement_v2 \
  --output-dir data/processed/sft_data/teacher_v2_plus_prompt_supplement_v2
```

当前推荐训练输入目录：

- `data/processed/sft_data/teacher_v2_plus_prompt_supplement_v2`

## 4. 使用 LLaMA-Factory 训练

当前主训练入口已经默认指向合并后的数据集。

进入训练环境后运行：

```bash
conda activate train
bash scripts/llama_factory/run_train.sh
```

该脚本会自动：

1. 将 `teacher_v2_plus_prompt_supplement_v2` 转换成 LLaMA-Factory 使用的 ShareGPT 数据
2. 启动 LoRA 训练

默认相关路径：

- 输入数据：`data/processed/sft_data/teacher_v2_plus_prompt_supplement_v2`
- 转换后数据：`data/processed/llama_factory/teacher_v2_plus_prompt_supplement_v2`
- 训练配置：`src/config/llama_factory_config.yaml`
- 输出目录：`model/lora/teacher_v2_plus_prompt_supplement_v2_qwen35_4b`

如果只想用单卡：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/llama_factory/run_train.sh
```

## 5. 推理

当前实际推理入口：

```bash
python scripts/run_cpu_inference.py "烛煌的真实身份是什么？"
```

典型示例：

```bash
conda activate reasoning

python scripts/run_cpu_inference.py \
  "烛煌的真实身份是什么？" \
  --llama-cli third_party/llama.cpp/build/bin/llama-completion \
  --gguf-model model/gguf/teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf \
  --answer-only
```

默认行为：

- 自动生成初始 hypothesis
- 自动执行混合检索
- 自动判断是否继续检索
- 如果模型返回 `retrieve_more`，则继续 follow-up retrieval
- 达到足够证据或达到安全上限后生成最终答案

GPU 加速推理：

```bash
conda activate train

bash scripts/run_gpu_inference.sh \
  --base-model model/qwen3.5-4b \
  --lora-path model/lora/teacher_v2_plus_prompt_supplement_v2_qwen35_4b \
  --answer-only
```

说明：

- `scripts/run_gpu_inference.sh` 默认优先走 `vLLM`
- 如果命令里显式传了 `--gguf-model` 或 `--llama-cli`，脚本会自动退回 `llama.cpp`
- 也可以手动指定 `--backend vllm` 或 `--backend llama.cpp`

## 6. 运行时检索配置

实际使用阶段的检索参数通过这个文件控制：

- `configs/runtime_inference.json`

当前支持的关键字段：

```json
{
  "retrieval": {
    "device": "cpu",
    "enable_reranker": true,
    "dense_top_k": 40,
    "sparse_top_k": 40,
    "fusion_top_k": 30,
    "rerank_top_k": 10,
    "rerank_batch_size": 8
  },
  "generator": {
    "backend": "vllm"
  },
  "inference": {
    "max_follow_up_rounds": 6
  }
}
```

说明：

- `enable_reranker`
  是否启用交叉编码器重排
- `generator.backend`
  生成后端，当前支持 `llama.cpp` 与 `vllm`
- `dense_top_k`
  稠密召回候选数
- `sparse_top_k`
  稀疏召回候选数
- `fusion_top_k`
  融合后进入下一阶段的候选数
- `rerank_top_k`
  重排后保留的证据数
- `max_follow_up_rounds`
  多轮检索安全上限；模型只要持续返回 `retrieve_more` 就会继续检索，直到达到上限

临时覆盖配置也可以直接走 CLI：

```bash
python scripts/run_cpu_inference.py \
  "烛煌的真实身份是什么？" \
  --llama-cli third_party/llama.cpp/build/bin/llama-completion \
  --gguf-model model/gguf/teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf \
  --no-reranker \
  --fusion-top-k 12 \
  --rerank-top-k 6 \
  --answer-only
```

## 7. 检索延迟测试

用于拆解各阶段延迟：

- `dense_encode`
- `FAISS`
- `BM25`
- `fusion`
- `rerank`
- `end-to-end`

GPU 检索测试：

```bash
conda activate train

python scripts/benchmark_retrieval_latency.py \
  --device cuda \
  --query "烛煌的真实身份是什么？" \
  --query "Logos和菈玛莲是什么关系？" \
  --query "沙卒在萨尔贡黑市的地位和影响力如何？" \
  --warmup 2 \
  --repeat 5 \
  --output outputs/retrieval_latency_train_gpu.json
```

CPU 检索测试：

```bash
conda activate reasoning

python scripts/benchmark_retrieval_latency.py \
  --device cpu \
  --query "烛煌的真实身份是什么？" \
  --warmup 2 \
  --repeat 5 \
  --output outputs/retrieval_latency_reasoning_cpu.json
```

如果要看不带重排的延迟：

```bash
python scripts/benchmark_retrieval_latency.py \
  --device cpu \
  --query "烛煌的真实身份是什么？" \
  --disable-reranker
```

## 8. 常用脚本

- `scripts/build_retrieval_index.py`
  构建剧情检索索引
- `scripts/query_retrieval.py`
  手工调试检索结果
- `scripts/generate_sft_from_teacher.py`
  生成主 SFT 数据
- `scripts/generate_prompt_supplement_from_teacher.py`
  生成 `user_question_hypothesis_generation` / `follow_up_hypothesis_generation` / `conclusion_generation` 补充数据
- `scripts/merge_sft_datasets.py`
  合并数据集
- `scripts/llama_factory/run_train.sh`
  主训练入口
- `scripts/run_cpu_inference.py`
  实际使用阶段的推理入口
- `scripts/benchmark_retrieval_latency.py`
  检索链路延迟测试

## 9. 当前训练与推理约定

- 主训练链路使用 `LLaMA-Factory`
- `scripts/transformers_peft/` 保留为兼容/调试路径，不是当前主训练入口
- 实际推理默认走 `llama.cpp`
- 实际运行时，是否继续检索由模型 planner 决定，而不是固定只补一轮

## 10. 常见问题

1. `llama.cpp CLI not found`

请确认：

- 已构建 `third_party/llama.cpp`
- 传入了真实的 `--llama-cli` 路径

2. `GGUF model not found`

请确认：

- 已导出 GGUF
- `--gguf-model` 指向真实文件

3. CPU 推理很慢

优先检查：

- `configs/runtime_inference.json` 中是否关闭了 reranker
- `fusion_top_k` 和 `rerank_top_k` 是否过大

4. 训练报数据集找不到

请先确认是否已执行：

```bash
python scripts/merge_sft_datasets.py ...
```

以及：

```bash
python scripts/llama_factory/prepare_sft_dataset.py ...
```

## 11. 当前推荐使用顺序

1. 构建索引
2. 生成主 SFT 数据
3. 生成 supplement 数据
4. 合并数据集
5. 使用 LLaMA-Factory 训练 LoRA
6. 导出 GGUF / 挂载 LoRA
7. 使用 `run_cpu_inference.py` 进行实际推理
