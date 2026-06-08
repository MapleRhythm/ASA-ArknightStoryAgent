#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-api}"

if [[ "${EUID:-$(id -u)}" -eq 0 && "${ALLOW_ROOT_SETUP:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
不要用 sudo 运行此脚本。

请执行：
  bash scripts/setup_cpu_api_no_reranker.sh
MSG
  exit 2
fi

if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[setup] found incomplete venv, recreating: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    cat >&2 <<'MSG'
[setup] 创建 Python 虚拟环境失败。

Debian/Ubuntu/WSL 通常需要先安装：
  sudo apt update
  sudo apt install -y python3-venv python3-pip
MSG
    exit 1
  fi
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python "$ROOT_DIR/scripts/download_models.py" --runtime cpu-api
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
