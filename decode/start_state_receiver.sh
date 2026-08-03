#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
STATE_DIR="${SLOT_SAVE_PATH:-$ROOT/state}"
HOST="${STATE_RECEIVER_HOST:-0.0.0.0}"
PORT="${STATE_RECEIVER_PORT:-8090}"

termux-wake-lock
exec python "$ROOT/decode/state_receiver.py" \
  --bind "$HOST" --port "$PORT" --state-dir "$STATE_DIR"
