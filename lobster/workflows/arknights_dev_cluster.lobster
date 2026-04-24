name: arknights-dev-cluster
args:
  sft_message:
    default: >
      为当前项目生成 SFT 数据构建、清洗、去重与切分方案，并优先指出缺失样本类型。
  lora_message:
    default: >
      为当前项目生成 Qwen3.5-4B 的 LLaMA-Factory LoRA 微调配置与执行方案，
      优先保证 llama.cpp 兼容性。
  rag_message:
    default: >
      为当前项目优化 RAG 检索链路，重点提升剧情问答的召回质量、重排质量和错误分析能力。
steps:
  - id: sft_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_SFT_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent sft-data-curator
      --message "$TASK_PROMPT"
      --json

  - id: lora_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_LORA_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent lora-trainer
      --message "$TASK_PROMPT"
      --json

  - id: rag_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_RAG_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent rag-retrieval-optimizer
      --message "$TASK_PROMPT"
      --json

  - id: emit_cluster_result
    pipeline: json
    stdin: $rag_agent.stdout
