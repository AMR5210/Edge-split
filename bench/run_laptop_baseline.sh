#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/llama_cpp.env"

: "${MODEL_PATH:?Set MODEL_PATH to the GGUF model file}"
: "${MODEL_NAME:?Set MODEL_NAME, e.g. Qwen3-0.6B-Instruct}"
: "${QUANT:?Set QUANT, e.g. Q4_K_M}"
: "${PROMPT:?Set PROMPT}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
PYTHON="${PYTHON:-python3}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"

"$PYTHON" "$ROOT/bench/run_cli_baseline.py" \
  --db "$ROOT/bench/edgesplit.sqlite3" \
  --llama-cli "$LLAMA_CPP_DIR/build-cuda/bin/llama-cli" \
  --model "$MODEL_PATH" \
  --device "${DEVICE_NAME:-rtx-4050-laptop}" \
  --config laptop-only \
  --model-name "$MODEL_NAME" \
  --quant "$QUANT" \
  --llama-cpp-commit "$LLAMA_CPP_COMMIT" \
  --prompt "$PROMPT" \
  --output-tokens "$OUTPUT_TOKENS" \
  --gpu-layers "${GPU_LAYERS:-999}"

