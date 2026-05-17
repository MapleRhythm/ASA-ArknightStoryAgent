#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-api}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python "$ROOT_DIR/scripts/download_models.py" --skip-reranker
fi

cat <<'MSG'

CPU API 环境安装完成。

需要准备：
1. 主索引：
   python scripts/build_retrieval_index.py --device cpu
2. API key：
   export OPENAI_API_KEY="你的 key"
3. 如使用其它 OpenAI 兼容服务，修改：
   configs/runtime_cpu_api_no_reranker.json

运行：
   bash scripts/run_cpu_api_no_reranker.sh "炎景公主一事具体指什么"

说明：此版本只在本地做 CPU 检索，不加载 reranker，生成阶段走远程 API。

MSG
