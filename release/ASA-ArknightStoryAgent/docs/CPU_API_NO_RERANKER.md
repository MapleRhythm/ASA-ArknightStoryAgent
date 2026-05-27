# CPU 版本：API 生成，无 reranker

适用场景：本地只做 CPU 检索，生成阶段调用 OpenAI 兼容 API；不加载 reranker，部署最轻。

默认流程是“先答后检索再校正”：

1. 远程 LLM 直接回答原问题，只加极少提示词。
2. 用“原问题 + 初答”作为检索 query，在本地 RAG/MiniRAG 中召回证据。
3. 再把初答和证据交给远程 LLM，要求它按证据修正幻觉和错误。

这条链路不会让 API 模型先生成 hypothesis/conclusion JSON，更适合强 API 模型。

## 环境脚本

```bash
bash scripts/setup_cpu_api_no_reranker.sh
source .venv-api/bin/activate
```

## 必要文件

```text
data/ArknightsGameData/zh_CN/gamedata/story/
data/ArknightsGameData/zh_CN/gamedata/excel/
indexes/arknights_story/
indexes/arknights_story_minirag/graph.json
```

主索引构建：

```bash
python scripts/build_retrieval_index.py --device cpu
```

## 配置 API

默认配置文件：

```text
configs/runtime_cpu_api_no_reranker.json
```

OpenAI 官方兼容接口示例：

```json
{
  "generator": {
    "backend": "openai_compatible_api",
    "api_base_url": "https://api.openai.com/v1/chat/completions",
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini"
  }
}
```

设置 key：

```bash
export OPENAI_API_KEY="你的 key"
```

第三方 OpenAI 兼容服务通常只需要改：

```json
{
  "api_base_url": "https://你的服务/v1/chat/completions",
  "api_key_env": "OPENAI_API_KEY",
  "model": "你的模型名"
}
```

## 运行脚本

```bash
bash scripts/run_cpu_api_no_reranker.sh "炎景公主一事具体指什么"
```

交互模式：

```bash
bash scripts/run_cpu_api_no_reranker.sh
```

## 命令接口

位置参数：

```text
question    可选。传入时直接回答一次；不传时进入交互模式。
```

常用参数：

```bash
--answer-only
--pipeline-mode answer_then_retrieve_refine
--pipeline-mode standard
--api-base-url https://api.openai.com/v1/chat/completions
--api-key-env OPENAI_API_KEY
--model gpt-4.1-mini
--timeout 120
--max-tokens 4096
--no-json-response-format
--no-save-run
```

输出：默认输出完整 JSON，并把 API 请求、初答、召回证据和校正结果保存到 `outputs/api_mode_runs/`。加 `--answer-only` 只输出最终回答；加 `--no-save-run` 不保存运行日志。

如果想回到旧版完整 RAG planning 流程，把配置里的 `inference.pipeline_mode` 改成 `standard`，或命令行传：

```bash
bash scripts/run_cpu_api_no_reranker.sh --pipeline-mode standard "炎景公主一事具体指什么"
```
