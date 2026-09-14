# ASA-ArknightStoryAgent

ASA（ArknightStoryAgent）是《明日方舟》的剧情研读助手：先检索原文，再回答人物关系、事件因果和剧情细节。

## 当前交付：API 固定版 `api-v1.1.0`（支持自定义接口）

默认网页现在使用 **DeepSeek Flash API**，不加载本地 4B，也不在失败时切换模型。
前端提供会话记录、中文输入法支持、复制与 Markdown 导出、完整原文面板、处理状态、
停止与手动重试，并适配手机屏幕。
「连接与配置」支持自定义 **Chat Completions 接口地址、模型 ID 和 API 密钥**。
网页提交的密钥只保留在本次服务进程内存，不写文件、不存入浏览器、不回显。

```bash
bash scripts/setup_cpu_api_no_reranker.sh
source .venv-api/bin/activate
# 按下节准备游戏数据后：
python scripts/build_retrieval_index.py --device cpu
bash scripts/run_web_ui.sh --ask-key
```

终端会隐藏输入密钥，浏览器打开 `http://127.0.0.1:7860`。
也可以提前在服务进程环境中设置 `DEEPSEEK_API_KEY`，省略 `--ask-key`。
也可不带 `--ask-key` 启动后，直接在网页「连接与配置 → API 连接」填写并保存。
切换服务商时选「通用兼容」；改地址须重新输入该接口的密钥，不会转发旧密钥。
Base URL 会自动补齐 `/chat/completions`；配置在重启服务后恢复启动默认值。
缺少索引、模型或密钥时，网页会显示准备步骤，不会把缺配置误报成“服务就绪”。

固定配置：`configs/runtime_api_stable.json`。命令行入口：

```bash
bash scripts/run_api_stable.sh "霜星为什么不再压抑自己的力量？"
```

自定义接口、启动参数与安全范围见 [API v1.1 说明](docs/API_RELEASE_V1_1.md)；
[v1.0 历史说明](docs/API_RELEASE_V1.md) 保留旧版行为。
该发布版使用 CPU 混合检索、无重排器；它不是远端 GPU 评测的同一份管线，
**不能直接套用历史20题的70%正确率**。完整原文指选中的检索 chunk，不是整章文本。

下面保留本地模型和通用 API 的历史运行方式，供研究使用；它们不是当前网页的默认入口。

## 配置要求

推荐环境：

| 模式 | 适用场景 | 最低建议 |
| --- | --- | --- |
| GPU 本地 | 研究模式，本地 Qwen3.5 4B + LoRA + reranker | Linux/WSL2、NVIDIA GPU、CUDA、16GB+ 显存 |
| CPU 本地 | 纯本地离线，llama.cpp + 合并 LoRA GGUF，无 reranker | Linux/WSL2/macOS、16GB+ 内存 |
| CPU API | 本地只做检索，生成阶段调用 OpenAI 兼容 API | Linux/WSL2/macOS、8GB+ 内存、API key |

基础软件：

- `python3`，建议 Python 3.10 或 3.11。
- `git`，用于本地克隆公开游戏数据仓库和可选 llama.cpp。
- GPU 模式需要可用的 NVIDIA 驱动和 CUDA 环境。
- CPU 本地模式如需自动编译 llama.cpp，需要 `cmake`、`make`、C/C++ 编译器。

数据与模型要求：

- 本发布包不内置原始游戏文本，需要本地准备 `data/ArknightsGameData`。
- MiniRAG v3 图已随发布包提供，主检索索引仍需根据本地游戏数据构建。
- GPU 模式会通过 `scripts/download_models.py --runtime gpu` 下载 embedding、Qwen3.5 4B、LoRA 和微调 reranker。
- CPU 本地模式会通过 `scripts/download_models.py --runtime cpu-local` 下载 embedding 和 GGUF。
- CPU API 模式只需要 embedding、本地索引和 API key。

游戏文本数据来自第三方公开仓库 `https://github.com/Kengxxiao/ArknightsGameData.git`。本发布包只提供本地克隆和建索引流程，不再分发原始游戏文本；请自行确认数据使用方式符合你的场景要求。

## 快速开始

以下命令默认从发布版仓库根目录执行。

### 1. 准备游戏文本

如果本地还没有游戏数据：

```bash
mkdir -p data
git clone --depth 1 https://github.com/Kengxxiao/ArknightsGameData.git data/ArknightsGameData
```

最终应存在：

```text
data/ArknightsGameData/zh_CN/gamedata/story/
data/ArknightsGameData/zh_CN/gamedata/excel/
```

### 2. GPU 推荐流程

GPU 流程会安装依赖、安装 vLLM、下载模型、构建索引并运行一次问题。

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

bash scripts/setup_gpu_reranker_qwen35_4b.sh
source .venv-gpu/bin/activate

python scripts/build_retrieval_index.py --device cuda
bash scripts/run_gpu_reranker_qwen35_4b.sh --answer-only "岁兽是什么？"
```

正式提问：

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh "炎景公主一事具体指什么"
```

如需使用 2 张 GPU，可在运行时设置 tensor parallel：

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh \
  --tensor-parallel-size 2 \
  "炎景公主一事具体指什么"
```

### 3. CPU 本地流程

CPU 本地模式使用已合并 LoRA 的 GGUF，不加载 reranker。

```bash
bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
source .venv-cpu/bin/activate

python scripts/build_retrieval_index.py --device cpu
bash scripts/run_cpu_qwen35_4b_no_reranker.sh "炎景公主一事具体指什么"
```

如需同时自动准备 CPU 版 llama.cpp：

```bash
BUILD_LLAMA_CPP=1 bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
```

### 4. CPU API 流程

CPU API 模式只在本地做检索，生成阶段调用 OpenAI 兼容 API。

```bash
bash scripts/setup_cpu_api_no_reranker.sh
source .venv-api/bin/activate

export OPENAI_API_KEY="你的 key"
python scripts/build_retrieval_index.py --device cpu
bash scripts/run_cpu_api_no_reranker.sh "炎景公主一事具体指什么"
```

如果使用第三方 OpenAI 兼容服务，修改：

```text
configs/runtime_cpu_api_no_reranker.json
```

### 5. Web UI（API-only，默认 DeepSeek，可自定义）

准备好 API 环境、检索模型和索引后，可启动本地前端（不再切换本地 4B）：

```bash
bash scripts/run_web_ui.sh --host 127.0.0.1 --port 7860 --ask-key
```

使用其他兼容服务，也可通过启动参数指定（密钥隐藏输入，不放命令参数）：

```bash
bash scripts/run_web_ui.sh \
  --api-base-url https://your-provider.example/v1 \
  --model your-model-id --request-profile standard --ask-key
```

浏览器打开：

```text
http://127.0.0.1:7860
```

### 6. 检索调试

只查看检索结果：

```bash
python scripts/query_retrieval.py "岁兽是什么，为什么会成为危机"
```

## 配置说明

发布版提供三套推荐配置：

| 配置文件 | 模式 | 说明 |
| --- | --- | --- |
| `configs/runtime_gpu_reranker_qwen35_4b.json` | GPU 本地 | vLLM + Qwen3.5 4B + LoRA + reranker，质量优先 |
| `configs/runtime_cpu_qwen35_4b_no_reranker.json` | CPU 本地 | llama.cpp + 合并 LoRA GGUF，无 reranker |
| `configs/runtime_cpu_api_no_reranker.json` | CPU API | 本地检索 + 远程 OpenAI 兼容 API，无 reranker |

关键目录：

```text
api-mode/                         # API 模式入口
configs/                          # 三种运行模式的 runtime 配置
data/                             # 放 ArknightsGameData
docs/                             # 更详细的部署与配置说明
indexes/arknights_story_minirag_v3/ # 内置 MiniRAG v3 图
model/                            # 放 embedding、reranker、4B、LoRA、GGUF
scripts/                          # 环境、建索引、运行脚本
src/asa_arknight_story_agent/     # 推理最小源码
web/                              # 本地浏览器前端
```

链路源码的分层边界见：

```text
ARCHITECTURE.md
```

当前 GPU LoRA 默认目录：

```text
model/lora/asa-arknightstoryagent-4b-lora/
```

当前 GPU 模式需要的主要模型目录：

```text
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
model/embeddings/bge-small-zh-v1.5/
```

CPU 本地模式需要：

```text
third_party/llama.cpp/build-cpu/bin/llama-completion
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
model/embeddings/bge-small-zh-v1.5/
```

主索引构建后应存在：

```text
indexes/arknights_story/documents.jsonl
indexes/arknights_story/faiss.index
indexes/arknights_story/bm25_tokens.pkl
indexes/arknights_story/operator_aliases.json
```

MiniRAG v3 图已内置：

```text
indexes/arknights_story_minirag_v3/graph.json
```

如需基于本地 documents 重建 MiniRAG 图：

```bash
python scripts/build_minirag_index.py --progress --output indexes/arknights_story_minirag_v3/graph.json
```

关键配置项：

- `retrieval.device`：检索侧设备，`cpu` 或 `cuda`。
- `retrieval.enable_reranker`：是否加载 reranker。GPU 默认开启，CPU/API 默认关闭。
- `retrieval.dense_top_k` / `retrieval.sparse_top_k`：dense 和 BM25 初召回数量。
- `retrieval.enable_minirag`：是否启用 MiniRAG 图召回。
- `retrieval.minirag_index_path`：MiniRAG 图路径。
- `retrieval.fusion_top_k`：多路召回融合后保留的候选数。
- `retrieval.rerank_top_k`：最终进入证据选择的候选数。
- `retrieval.enable_neighbor_expansion`：是否启用 chunk 邻接扩展。
- `inference.max_retrieval_rounds`：最多检索轮数，核心链路会限制为最多 2 轮。
- `inference.prompt_evidence_top_k`：塞给生成模型的证据数量。
- `inference.answer_grounding_mode`：当前 LoRA 推荐 `quote`。
- `generator.backend`：`vllm`、`llama.cpp` 或 `openai_compatible_api`。
- `generator.max_tokens`：单次生成最大 token 数。
- `generator.vllm.gpu_memory_utilization`：vLLM 显存使用比例。

更完整字段说明见：

```text
docs/CONFIG_REFERENCE.md
docs/GPU_RERANKER_QWEN35_4B.md
docs/CPU_QWEN35_4B_NO_RERANKER.md
docs/CPU_API_NO_RERANKER.md
```

## 常见问题

### 能不能一键克隆游戏数据？

可以。快速开始中的 `git clone --depth 1 https://github.com/Kengxxiao/ArknightsGameData.git data/ArknightsGameData` 会在本地克隆公开游戏数据仓库。本发布包不内置原始游戏文本，也不负责再分发授权；如果你的使用场景对数据来源或授权有额外要求，请改为手动准备数据。

### vLLM 显存不足怎么办？

优先降低：

- `--gpu-memory-utilization`
- `--ctx-size`
- `generator.vllm.max_model_len`
- `generator.vllm.max_num_batched_tokens`
- `inference.prompt_conclusion_evidence_max_total_chars`
- `retrieval.rerank_top_k`

同时用 `nvidia-smi` 检查是否有旧进程占用显存。

### 可以使用 2 张 GPU 吗？

可以。GPU 运行脚本支持：

```bash
--tensor-parallel-size 2
```

如果仍然 OOM，继续降低 `--gpu-memory-utilization` 或 `--ctx-size`。如果只是单卡 24GB 显存，默认 `tensor_parallel_size=1` 更简单。

### API key 放在哪里？

默认读取环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

不要把 API key 写入配置文件或提交到仓库。第三方 OpenAI 兼容服务通常只需要改 `api_base_url`、`api_key_env` 和 `model`。

### 召回不到目标原文怎么办？

先运行：

```bash
python scripts/query_retrieval.py "你的问题"
```

再检查 runtime trace 中的：

- `retained_chapter_scope`
- `retained_storyline_scope`
- `minirag_chapter_expansion`
- `evidence_summary`

如果前两轮都没有新增有效证据，第三轮通常只会增加延迟，因此当前发布版不会继续第三轮召回。

### 输出不是最终答案而是一大段 JSON？

默认输出完整 JSON，包含问题、检索 trace、证据和答案。只想看最终答案时加：

```bash
--answer-only
```

### 国内网络下载模型慢怎么办？

可设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

## 链路说明

GPU 本地标准链路：

```text
用户问题
-> user_question_hypothesis_generation
-> dense + BM25 + MiniRAG v3 混合召回
-> MiniRAG 章节隔离 / 图扩展 / scoped second-pass retrieval
-> storyline sparse scope
-> 可选 neighbor expansion / web context
-> fusion + reranker
-> prompt evidence 去重、降权、截断
-> minimal conclusion_generation
-> answer_directly / retrieve_more / clarify_user / abstain
-> 最多 2 轮召回；达到上限后基于当前证据输出可确认部分
```

CPU 本地模式复用同一套检索与推理框架，但关闭 reranker，并使用 llama.cpp 运行已合并 LoRA 的 GGUF。CPU API 模式默认使用“先答后检索再校正”链路：API 模型先给出初答，本地检索用原问题和初答召回证据，再让 API 模型按证据修正答案；如需回到完整 planning 流程，可把 `inference.pipeline_mode` 改为 `standard`。

`max_retrieval_rounds` 在核心链路中会限制为最多 2 轮。达到上限后，系统不会继续机械检索，而是根据当前证据输出可确认部分，并明确缺少哪些直接证据。
