#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to clone llama.cpp into $LLAMA_CPP_DIR." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

CUDA_NVCC="${CUDA_NVCC:-}"
if [[ -z "$CUDA_NVCC" ]]; then
  for candidate in \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-12.1/bin/nvcc \
    /usr/local/cuda-12.0/bin/nvcc \
    /usr/local/cuda-11.8/bin/nvcc \
    /usr/local/cuda-11.7/bin/nvcc
  do
    if [[ -x "$candidate" ]]; then
      CUDA_NVCC="$candidate"
      break
    fi
  done
fi

if [[ -z "$CUDA_NVCC" || ! -x "$CUDA_NVCC" ]]; then
  echo "CUDA compiler not found." >&2
  echo "Set CUDA_NVCC=/abs/path/to/nvcc and rerun." >&2
  exit 1
fi

CUDA_ROOT="$(cd "$(dirname "$CUDA_NVCC")/.." && pwd)"

mkdir -p "$BUILD_DIR"

echo "Using nvcc: $CUDA_NVCC"
echo "Using CUDA root: $CUDA_ROOT"

"$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_NATIVE=ON

"$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS" --target llama-cli llama-completion llama-quantize

echo
echo "CUDA llama.cpp build completed."
echo "Binaries:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-completion"
