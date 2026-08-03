#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to the WSL llama.cpp checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to a GGUF model path}"
HOST="${PREFILL_HOST:-0.0.0.0}"
PORT="${PREFILL_PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-999}"
SLOT_SAVE_PATH="${PREFILL_SLOT_SAVE_PATH:-$ROOT/state}"

mkdir -p "$SLOT_SAVE_PATH"
exec "$LLAMA_CPP_DIR/build-cuda/bin/llama-server" \
  --model "$MODEL_PATH" --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" --slot-save-path "$SLOT_SAVE_PATH"
