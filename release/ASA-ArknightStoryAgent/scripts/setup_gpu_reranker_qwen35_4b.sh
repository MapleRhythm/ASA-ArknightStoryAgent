#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-gpu}"

if [[ "${EUID:-$(id -u)}" -eq 0 && "${ALLOW_ROOT_SETUP:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
不要用 sudo 运行此脚本。

请执行：
  bash scripts/setup_gpu_reranker_qwen35_4b.sh
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

if [[ "${SKIP_VLLM_INSTALL:-0}" != "1" ]]; then
  python -m pip install -U vllm
fi

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python "$ROOT_DIR/scripts/download_models.py" --runtime gpu
fi

cat <<'MSG'

GPU 环境安装完成。

需要准备：
1. 主索引：
   python scripts/build_retrieval_index.py --device cuda
2. Qwen3.5 4B 基座：
   model/qwen3.5-4b/
3. 4B LoRA：
   model/lora/asa-arknightstoryagent-4b-lora/
4. 微调 reranker：
   model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch/

运行：
   bash scripts/run_gpu_reranker_qwen35_4b.sh "炎景公主一事具体指什么"

如果 vLLM 需要按 CUDA 版本手动安装，可设置 SKIP_VLLM_INSTALL=1 后自行安装。

MSG
