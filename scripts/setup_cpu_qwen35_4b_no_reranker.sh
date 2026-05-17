#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-cpu}"

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

if [[ "${BUILD_LLAMA_CPP:-0}" == "1" ]]; then
  bash "$ROOT_DIR/scripts/build_llama_cpp_cpu.sh"
fi

cat <<'MSG'

CPU 本地模型环境安装完成。

需要准备：
1. 主索引：
   python scripts/build_retrieval_index.py --device cpu
2. llama.cpp 可执行文件：
   third_party/llama.cpp/build/bin/llama-completion
3. Qwen3.5 4B GGUF：
   model/gguf/qwen3.5-4b-q4_k_m.gguf

运行：
   bash scripts/run_cpu_qwen35_4b_no_reranker.sh "炎景公主一事具体指什么"

说明：此版本不加载 reranker，适合纯 CPU 部署。MiniRAG 图已内置。

MSG
