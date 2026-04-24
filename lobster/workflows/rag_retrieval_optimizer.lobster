name: rag-retrieval-optimizer
args:
  message:
    default: >
      审查当前仓库的 RAG 实现并优化剧情检索质量。
      重点关注：剧情解析、chunk 策略、BM25 分词、RRF 融合、
      BGE-Reranker-v2-m3 重排、假设文档驱动检索和离线评测集。
steps:
  - id: invoke_rag_agent
    env:
      TASK_PROMPT: "$LOBSTER_ARG_MESSAGE"
    run: >
      /home/zhb/goldenglow/.openclaw-cli/bin/openclaw agent --local
      --agent rag-retrieval-optimizer
      --message "$TASK_PROMPT"
      --json

  - id: emit_result
    pipeline: json
    stdin: $invoke_rag_agent.stdout
