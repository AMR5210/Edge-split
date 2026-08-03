#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Phone 2 only: this listener requires the same pinned llama.cpp source plus
# patches/llama.cpp/0001-edgesplit-v2-raw-sequence-endpoints.patch. V1's
# start_state_receiver.sh remains the untouched phone-1 file handoff path.
ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
MODEL_ID="${EDGESPLIT_MODEL_ID:-Qwen3-0.6B}"
LLAMA_CPP_COMMIT="${EDGESPLIT_LLAMA_CPP_COMMIT:-e920c523e3b8a0163fe498af5bf90df35ff51d25}"
PHONE_SERVER_URL="${EDGESPLIT_PHONE_SERVER_URL:-http://127.0.0.1:8081}"
V2_PORT="${EDGESPLIT_V2_PORT:-8091}"
V2_SLOT="${EDGESPLIT_V2_SLOT:-0}"

termux-wake-lock
exec python "$ROOT/kv_transfer/receiver.py" \
  --listen-host 0.0.0.0 \
  --port "$V2_PORT" \
  --phone-server "$PHONE_SERVER_URL" \
  --slot "$V2_SLOT" \
  --model-id "$MODEL_ID" \
  --llama-commit "$LLAMA_CPP_COMMIT"
