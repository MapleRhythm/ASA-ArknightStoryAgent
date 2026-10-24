#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build-cpu}"
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

if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

"$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_METAL=OFF

"$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS" --target llama-cli llama-completion llama-quantize

echo
echo "CPU llama.cpp build completed."
echo "Binaries:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-completion"
