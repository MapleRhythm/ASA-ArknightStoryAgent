# API Mode

This mode keeps local retrieval, BM25/dense fusion, neighbor expansion, and evidence-chain rerank, but replaces the local fine-tuned 4B generator with a remote OpenAI-compatible chat completion API.

## Usage

```bash
export OPENAI_API_KEY="..."

/home/zhb/miniconda3/envs/train/bin/python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_api.json \
  "乌尔比安为什么离开阿戈尔"
```

For an OpenAI-compatible provider, override the endpoint and model:

```bash
export MY_API_KEY="..."

python api-mode/run_api_inference.py \
  --api-base-url "https://your-provider.example/v1/chat/completions" \
  --api-key-env MY_API_KEY \
  --model "your-strong-model" \
  "艾泽尔和塞西莉亚关系"
```

## Notes

- This does not load local `qwen3.5-4b`, vLLM, llama.cpp, or LoRA.
- Retrieval still uses the local index and reranker configured in `api-mode/runtime_api.json`.
- The API runner converts the existing local ChatML prompts into normal chat messages and adds an API-mode system instruction so the remote model knows whether it is producing hypothesis JSON, follow-up JSON, conclusion JSON, or a grounded final answer.
- If your provider does not support `response_format: {"type": "json_object"}`, pass `--no-json-response-format`.
