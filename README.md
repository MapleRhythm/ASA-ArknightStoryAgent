# ASA-ArknightStoryAgent 推理发布版

这是 ASA-ArknightStoryAgent 的可部署推理版，只保留运行问答链路需要的代码、配置、Web UI、建索引脚本和 MiniRAG 图，不包含训练数据生成、SFT/KTO/reranker 训练、wandb 或 LLaMA-Factory 配置。

当前发布链路面向《明日方舟》剧情问答，原则是：优先依据剧情原文、档案、语音和联网补充证据回答；证据能支持部分内容时先回答可确认部分；证据不足时明确说明缺口，不把模型记忆或二创内容写成官方事实。

## 当前模型版本

GPU 发布版默认使用稳定 LoRA 目录：

```text
model/lora/asa-arknightstoryagent-4b-lora/
```

该目录对应 Hugging Face 仓库：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
```

当前上传内容为 `20260607-cutoff6656` adapter，本地训练产物来自：

```text
model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered
```

此版本配套的 runtime 已包含 grounded action JSON 解析、截断恢复和证据 quote 压缩适配；模型本身仍可能在低置信检索或主观问题上外推，生产使用应展示证据与 trace。

## 当前链路

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

`max_retrieval_rounds` 在核心链路中硬限制为最多 2 轮，即使外部传入更大的值也不会继续第三轮召回。

内置 MiniRAG v3 图：

```text
indexes/arknights_story_minirag_v3/graph.json
```

仍需本地构建主检索索引：

```text
indexes/arknights_story/documents.jsonl
indexes/arknights_story/faiss.index
indexes/arknights_story/bm25_tokens.pkl
indexes/arknights_story/operator_aliases.json
```

## 三种运行模式

| 模式 | 生成方式 | reranker | 默认配置 | 运行脚本 | 用途 |
| --- | --- | --- | --- | --- | --- |
| GPU | vLLM + Qwen3.5 4B + LoRA | 开启 | `configs/runtime_gpu_reranker_qwen35_4b.json` | `scripts/run_gpu_reranker_qwen35_4b.sh` | 质量优先 |
| CPU 本地 | llama.cpp + 合并 LoRA GGUF | 关闭 | `configs/runtime_cpu_qwen35_4b_no_reranker.json` | `scripts/run_cpu_qwen35_4b_no_reranker.sh` | 纯本地部署 |
| CPU API | 本地 CPU 检索 + OpenAI-compatible API | 关闭 | `configs/runtime_cpu_api_no_reranker.json` | `scripts/run_cpu_api_no_reranker.sh` | 轻量部署 |

也可以启动 Web UI：

```bash
bash scripts/run_web_ui.sh --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

## 目录结构

```text
api-mode/                         # API 模式入口
configs/                          # 三种运行模式的 runtime 配置
data/                             # 放 ArknightsGameData
docs/                             # 部署与配置说明
indexes/arknights_story_minirag_v3/ # 内置 MiniRAG v3 图
model/                            # 放 embedding、reranker、4B、LoRA、GGUF
scripts/                          # 环境、建索引、运行脚本
src/asa_arknight_story_agent/     # 推理最小源码
web/                              # 本地浏览器前端
```

## 一键部署（GPU 推荐）

以下命令适用于从发布版仓库根目录开始的首次 GPU 部署。它会在本地克隆 `ArknightsGameData`、安装 GPU 环境、下载模型、构建主检索索引，并跑一次测试问题。

`ArknightsGameData` 来自第三方公开仓库，本发布包不内置原始游戏文本；执行下面命令即表示你确认在本地获取并使用该数据。若不希望脚本克隆游戏数据，请跳过其中的 `git clone` 部分，按下一节手动准备。

```bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -d data/ArknightsGameData/zh_CN/gamedata/story && -d data/ArknightsGameData/zh_CN/gamedata/excel ]]; then
  echo "[bootstrap] game data already exists"
elif [[ ! -e data/ArknightsGameData ]]; then
  mkdir -p data
  git clone --depth 1 https://github.com/Kengxxiao/ArknightsGameData.git data/ArknightsGameData
else
  echo "[bootstrap] data/ArknightsGameData exists but is incomplete; fix or remove it first" >&2
  exit 1
fi

bash scripts/setup_gpu_reranker_qwen35_4b.sh
source .venv-gpu/bin/activate

python scripts/build_retrieval_index.py --device cuda
bash scripts/run_gpu_reranker_qwen35_4b.sh --answer-only "岁兽是什么？"
```

完成后可直接提问：

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh "炎景公主一事具体指什么"
```

如果没有可用 CUDA/vLLM 环境，使用下面的 CPU 本地或 CPU API 分步流程。

## 1. 准备游戏文本

游戏文本数据来自：

```text
https://github.com/Kengxxiao/ArknightsGameData.git
```

下载到发布目录：

```bash
git clone https://github.com/Kengxxiao/ArknightsGameData.git data/ArknightsGameData
```

最终应存在：

```text
data/ArknightsGameData/zh_CN/gamedata/story/
data/ArknightsGameData/zh_CN/gamedata/excel/
```

本仓库不内置原始游戏数据。请自行确认游戏数据及衍生索引的再分发许可。

## 2. 安装环境

GPU 版本：

```bash
bash scripts/setup_gpu_reranker_qwen35_4b.sh
source .venv-gpu/bin/activate
```

CPU 本地模型版本：

```bash
bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
source .venv-cpu/bin/activate
```

如需同时准备 llama.cpp：

```bash
BUILD_LLAMA_CPP=1 bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
source .venv-cpu/bin/activate
```

CPU API 版本：

```bash
bash scripts/setup_cpu_api_no_reranker.sh
source .venv-api/bin/activate
```

## 3. 构建主检索索引

GPU 环境：

```bash
python scripts/build_retrieval_index.py --device cuda
```

CPU / API 环境：

```bash
python scripts/build_retrieval_index.py --device cpu
```

MiniRAG v3 图已内置，一般不需要重建。如需基于本地 documents 重建：

```bash
python scripts/build_minirag_index.py --progress --output indexes/arknights_story_minirag_v3/graph.json
```

## 4. 准备模型或 API

GPU 版本需要：

```text
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
model/embeddings/bge-small-zh-v1.5/
```

CPU 本地模型版本需要：

```text
third_party/llama.cpp/build-cpu/bin/llama-completion
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
model/embeddings/bge-small-zh-v1.5/
```

CPU API 版本需要：

```bash
export OPENAI_API_KEY="你的 key"
```

如使用第三方 OpenAI 兼容服务，修改：

```text
configs/runtime_cpu_api_no_reranker.json
```

国内网络只下载模型时可设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 5. 运行

GPU：

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh "炎景公主一事具体指什么"
```

CPU 本地：

```bash
bash scripts/run_cpu_qwen35_4b_no_reranker.sh "炎景公主一事具体指什么"
```

CPU API：

```bash
bash scripts/run_cpu_api_no_reranker.sh "炎景公主一事具体指什么"
```

如果不传问题，会进入交互模式。加 `--answer-only` 只输出最终答案。

## 6. 检索调试

```bash
python scripts/query_retrieval.py "岁兽是什么，为什么会成为危机"
```

## 关键配置

GPU 默认配置：

```text
configs/runtime_gpu_reranker_qwen35_4b.json
```

当前关键项：

- `retrieval.reranker_model_path`: `model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch`
- `retrieval.minirag_index_path`: `indexes/arknights_story_minirag_v3/graph.json`
- `retrieval.minirag_chapter_isolation`: `true`
- `retrieval.minirag_auto_second_retrieval`: `true`
- `retrieval.enable_storyline_sparse_scope`: `true`
- `retrieval.enable_neighbor_expansion`: GPU 默认 `true`
- `inference.max_retrieval_rounds`: `2`
- `inference.conclusion_prompt_mode`: `minimal`
- `inference.answer_grounding_mode`: `quote`
- `inference.web_context.enabled`: GPU 默认 `false`
- `generator.max_tokens`: `1536`
- `generator.vllm.gpu_memory_utilization`: `0.52`
- `generator.vllm.max_num_batched_tokens`: `4096`

## 常见问题

### vLLM OOM

优先降低：

- `generator.vllm.gpu_memory_utilization`
- `generator.vllm.max_model_len`
- `inference.prompt_conclusion_evidence_max_total_chars`
- `retrieval.rerank_top_k`

同时用 `nvidia-smi` 检查是否有旧进程占显存。

### API key

默认读取：

```bash
export OPENAI_API_KEY="sk-..."
```

不要把 API key 写入配置文件或提交到仓库。

### 召回不到目标原文

先用 `scripts/query_retrieval.py` 检查静态召回，再检查 runtime trace 里的：

- `retained_chapter_scope`
- `retained_storyline_scope`
- `minirag_chapter_expansion`
- `evidence_summary`

如果前两轮都没有新增有效证据，第三轮通常只增加延迟，因此当前发布版不再启用第三轮召回。
