# CPU 版本：已合并 LoRA 的 Qwen3.5 4B，无 reranker

适用场景：纯 CPU 部署，本地 llama.cpp 跑已合并 LoRA 的 Qwen3.5 4B GGUF；不加载运行时 LoRA，也不加载 reranker，降低部署复杂度和内存成本。

## 环境脚本

```bash
bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
source .venv-cpu/bin/activate
```

如需顺带编译 CPU 版 llama.cpp：

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
BUILD_LLAMA_CPP=1 bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
```

## 必要文件

```text
data/ArknightsGameData/zh_CN/gamedata/story/
data/ArknightsGameData/zh_CN/gamedata/excel/
indexes/arknights_story/
indexes/arknights_story_minirag_v3/graph.json
third_party/llama.cpp/build-cpu/bin/llama-completion
model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
```

注意：CPU 发布版推荐使用已合并 LoRA 的 GGUF。llama.cpp 对部分 Qwen3.5 LoRA 的运行时 LoRA GGUF 转换兼容性有限，合并后再导出 GGUF 更稳。

主索引构建：

```bash
python scripts/build_retrieval_index.py --device cpu
```

## 运行脚本

```bash
bash scripts/run_cpu_qwen35_4b_no_reranker.sh "炎景公主一事具体指什么"
```

交互模式：

```bash
bash scripts/run_cpu_qwen35_4b_no_reranker.sh
```

## 命令接口

位置参数：

```text
question    可选。传入时直接回答一次；不传时进入交互模式。
```

常用参数：

```bash
--answer-only
--threads 16
--ctx-size 8192
--max-tokens 512
--gguf-model model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf
--llama-cli third_party/llama.cpp/build-cpu/bin/llama-completion
--dense-top-k 80
--sparse-top-k 80
--fusion-top-k 50
```

默认配置文件：

```text
configs/runtime_cpu_qwen35_4b_no_reranker.json
```

输出：默认输出完整 JSON；加 `--answer-only` 只输出最终回答。
