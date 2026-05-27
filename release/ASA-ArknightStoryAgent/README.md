# ASA-ArknightStoryAgent 明日方舟剧情问答推理版

这是 ASA-ArknightStoryAgent 的 GitHub 推理发布版。目录内只保留部署和推理所需内容，不包含 SFT 数据生成、teacher 标注生成、DPO/reranker 训练、wandb、LLaMA-Factory 或训练脚本。

发布版已内置 MiniRAG 图：

```text
indexes/arknights_story_minirag/graph.json
```

你仍需要本地构建 Dense / BM25 / FAISS 主索引。

## 三个推理版本

| 版本 | 生成方式 | reranker | 环境脚本 | 运行脚本 | 说明 |
| --- | --- | --- | --- | --- | --- |
| GPU | vLLM + Qwen3.5 4B + LoRA | 开启 | `scripts/setup_gpu_reranker_qwen35_4b.sh` | `scripts/run_gpu_reranker_qwen35_4b.sh` | 质量优先 |
| CPU 本地 | llama.cpp + 已合并 LoRA 的 Qwen3.5 4B GGUF | 关闭 | `scripts/setup_cpu_qwen35_4b_no_reranker.sh` | `scripts/run_cpu_qwen35_4b_no_reranker.sh` | 纯本地 CPU |
| CPU API | 本地 CPU 检索 + 远程 API | 关闭 | `scripts/setup_cpu_api_no_reranker.sh` | `scripts/run_cpu_api_no_reranker.sh` | 部署最轻 |

也可以启动本地 Web 前端，在浏览器里切换这三种模式：

```bash
bash scripts/run_web_ui.sh --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

接口说明：

```text
docs/GPU_RERANKER_QWEN35_4B.md
docs/CPU_QWEN35_4B_NO_RERANKER.md
docs/CPU_API_NO_RERANKER.md
```

配置说明：

```text
docs/CONFIG_REFERENCE.md
```

## 目录结构

```text
api-mode/                         # API 模式入口
configs/                          # 三个版本的 runtime 配置
data/                             # 放 ArknightsGameData
docs/                             # 接口说明和配置说明
indexes/arknights_story_minirag/  # 已内置 MiniRAG 图
model/                            # 放 embedding、reranker、4B、LoRA、GGUF
scripts/                          # 环境、建索引、运行脚本
src/asa_arknight_story_agent/     # 推理最小源码
web/                              # 本地浏览器前端
```

## 1. 获取游戏数据

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

## 2. 选择环境

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

CPU API 版本：

```bash
bash scripts/setup_cpu_api_no_reranker.sh
source .venv-api/bin/activate
```

## 3. 构建主检索索引

GPU 环境可用：

```bash
python scripts/build_retrieval_index.py --device cuda
```

CPU / API 环境用：

```bash
python scripts/build_retrieval_index.py --device cpu
```

生成后应看到：

```text
indexes/arknights_story/documents.jsonl
indexes/arknights_story/faiss.index
indexes/arknights_story/bm25_tokens.pkl
indexes/arknights_story/operator_aliases.json
```

MiniRAG 图已内置，一般不需要重建。如需基于本地 documents 重建：

```bash
python scripts/build_minirag_index.py --progress
```

## 4. 准备模型或 API

GPU 版本需要：

```text
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/reranker/bge-reranker-v2-m3-evidence-chain-answerability/
```

CPU 本地模型版本需要：

```text
third_party/llama.cpp/build-cpu/bin/llama-completion
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

对应环境脚本会自动从 Hugging Face 下载默认模型：

```text
MapleRhythm/asa-arknightstoryagent-4b-lora
MapleRhythm/asa-arknightstoryagent-4b-gguf
MapleRhythm/asa-evidence-chain-reranker
```

国内网络只下载时可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果你 fork 了模型仓库，可用环境变量覆盖：

```bash
export ASA_HF_OWNER=你的HF用户名
```

CPU API 版本需要：

```bash
export OPENAI_API_KEY="你的 key"
```

如使用第三方 OpenAI 兼容接口，修改：

```text
configs/runtime_cpu_api_no_reranker.json
```

微调权重上传到 Hugging Face 的方法见：

```text
HUGGINGFACE_UPLOAD.md
```

## 5. 运行

Web 前端：

```bash
bash scripts/run_web_ui.sh --host 127.0.0.1 --port 7860
```

前端会调用已有推理脚本。CPU 本地模式首次回答会加载索引和模型，可能需要几分钟；CPU API 模式需要先设置 API key。

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
