name: lora-trainer
args:
  message:
    default: >
      基于当前仓库，为 Qwen3.5-4B 设计本地 LLaMA-Factory LoRA 微调方案。
      要求兼容 llama.cpp，覆盖澄闪语气、意图识别、多轮对话、function calling，
      并输出训练配置、数据集映射、启动命令、导出与验证命令。
steps:
  - id: invoke_lora_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent lora-trainer
      --message "$TASK_PROMPT"
      --json

  - id: emit_result
    pipeline: json
    stdin: $invoke_lora_agent.stdout
