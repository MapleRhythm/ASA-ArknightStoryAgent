#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<'MSG'
请选择一个明确的推理版本安装环境：

1. GPU：reranker + Qwen3.5 4B
   bash scripts/setup_gpu_reranker_qwen35_4b.sh
   source .venv-gpu/bin/activate

2. CPU：Qwen3.5 4B，无 reranker
   bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
   source .venv-cpu/bin/activate

3. CPU：API 生成，无 reranker
   bash scripts/setup_cpu_api_no_reranker.sh
   source .venv-api/bin/activate

接口说明：
   docs/GPU_RERANKER_QWEN35_4B.md
   docs/CPU_QWEN35_4B_NO_RERANKER.md
   docs/CPU_API_NO_RERANKER.md

配置说明：
   docs/CONFIG_REFERENCE.md
MSG

if [[ "${RUN_DEFAULT_SETUP:-0}" == "1" ]]; then
  exec "$ROOT_DIR/scripts/setup_cpu_api_no_reranker.sh"
fi
