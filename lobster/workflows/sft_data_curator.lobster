name: sft-data-curator
args:
  message:
    default: >
      基于当前仓库的数据结构，为明日方舟剧情问答项目生成 SFT 数据方案。
      重点覆盖：剧情事实问答、角色关系、多轮追问、澄清样本、拒答样本、
      function calling 样本，并给出清洗与 train/val/test 划分策略。
steps:
  - id: invoke_sft_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent sft-data-curator
      --message "$TASK_PROMPT"
      --json

  - id: emit_result
    pipeline: json
    stdin: $invoke_sft_agent.stdout
