#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$BUILD_DIR"

"$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON

"$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS" --target llama-cli llama-completion llama-quantize

echo
echo "CPU llama.cpp build completed."
echo "Binaries:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-completion"
