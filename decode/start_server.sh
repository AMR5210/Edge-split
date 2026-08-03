#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a GGUF model path}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-$HOME/llama-b10034/bin}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_BIN_DIR/llama-server}"
HOST="${DECODE_HOST:-0.0.0.0}"
PORT="${DECODE_PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-4096}"
DECODE_THREADS="${DECODE_THREADS:-2}"
ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-$ROOT/state}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not executable: $LLAMA_SERVER" >&2
  echo "Set LLAMA_BIN_DIR to the official b10034 archive's bin directory." >&2
  exit 2
fi

export LD_LIBRARY_PATH="$LLAMA_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$SLOT_SAVE_PATH"
termux-wake-lock
exec "$LLAMA_SERVER" \
  --model "$MODEL_PATH" --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" \
  --threads "$DECODE_THREADS" --no-mmap --slot-save-path "$SLOT_SAVE_PATH"
