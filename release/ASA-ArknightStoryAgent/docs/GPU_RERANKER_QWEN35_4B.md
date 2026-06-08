# GPU 版本：reranker + Qwen3.5 4B

适用场景：有 NVIDIA GPU，希望使用本地 Qwen3.5 4B + LoRA，并开启证据链 reranker 获取更好的证据排序。

## 环境脚本

```bash
bash scripts/setup_gpu_reranker_qwen35_4b.sh
source .venv-gpu/bin/activate
```

脚本会安装基础依赖、vLLM，并下载公开 embedding/reranker 基座。若 vLLM 需要按 CUDA 版本单独安装：

```bash
SKIP_VLLM_INSTALL=1 bash scripts/setup_gpu_reranker_qwen35_4b.sh
```

## 必要文件

```text
data/ArknightsGameData/zh_CN/gamedata/story/
data/ArknightsGameData/zh_CN/gamedata/excel/
indexes/arknights_story/
indexes/arknights_story_minirag_v3/graph.json
model/qwen3.5-4b/
model/lora/asa-arknightstoryagent-4b-lora/
model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/
```

当前 `model/lora/asa-arknightstoryagent-4b-lora/` 对应 HF 稳定仓库 `MapleRhythm/asa-arknightstoryagent-4b-lora`，内容版本为 `20260607-cutoff6656`。

主索引构建：

```bash
python scripts/build_retrieval_index.py --device cuda
```

## 运行脚本

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh "炎景公主一事具体指什么"
```

交互模式：

```bash
bash scripts/run_gpu_reranker_qwen35_4b.sh
```

## 命令接口

位置参数：

```text
question    可选。传入时直接回答一次；不传时进入交互模式。
```

常用参数：

```bash
--answer-only
--dialogue-context "user: ...\nassistant: ..."
--runtime-config configs/runtime_gpu_reranker_qwen35_4b.json
--base-model model/qwen3.5-4b
--lora-path model/lora/asa-arknightstoryagent-4b-lora
--reranker-model model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch
--tensor-parallel-size 1
--gpu-memory-utilization 0.52
--max-tokens 1536
--ctx-size 10000
```

默认配置文件：

```text
configs/runtime_gpu_reranker_qwen35_4b.json
```

输出：默认输出完整 JSON，包含 `question`、`hypothesis`、`retrieval_trace`、`evidence`、`answer`。加 `--answer-only` 只输出最终回答。

当前默认关闭 web context，并使用 `answer_grounding_mode=quote`。如果显存不足，优先降低 `--gpu-memory-utilization`、`--ctx-size` 或 `retrieval.rerank_top_k`。
