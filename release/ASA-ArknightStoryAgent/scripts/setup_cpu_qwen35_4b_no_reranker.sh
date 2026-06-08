#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-cpu}"

if [[ "${EUID:-$(id -u)}" -eq 0 && "${ALLOW_ROOT_SETUP:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
不要用 sudo 运行此脚本。

虚拟环境应创建在当前用户目录下，否则后续运行会遇到权限和路径问题。
请执行：
  bash scripts/setup_cpu_qwen35_4b_no_reranker.sh

如果之前已经用 sudo 跑出残缺目录，先执行：
  sudo rm -rf .venv-cpu
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

然后重新执行：
  bash scripts/setup_cpu_qwen35_4b_no_reranker.sh
MSG
    exit 1
  fi
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python "$ROOT_DIR/scripts/download_models.py" --runtime cpu-local
fi

if [[ "${BUILD_LLAMA_CPP:-0}" == "1" ]]; then
  bash "$ROOT_DIR/scripts/build_llama_cpp_cpu.sh"
fi

LLAMA_COMPLETION="$ROOT_DIR/third_party/llama.cpp/build-cpu/bin/llama-completion"
if [[ -f "$LLAMA_COMPLETION" && ! -x "$LLAMA_COMPLETION" ]]; then
  chmod +x "$LLAMA_COMPLETION" || true
fi

cat <<'MSG'

CPU 本地模型环境安装完成。

需要准备：
1. 主索引：
   python scripts/build_retrieval_index.py --device cpu
2. llama.cpp 可执行文件：
   third_party/llama.cpp/build-cpu/bin/llama-completion
3. 已合并 LoRA 的 Qwen3.5 4B GGUF：
   model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf

运行：
   bash scripts/run_cpu_qwen35_4b_no_reranker.sh "炎景公主一事具体指什么"

说明：此版本使用已合并 LoRA 的 GGUF，不加载运行时 LoRA，也不加载 reranker，适合纯 CPU 部署。MiniRAG 图已内置。

MSG
