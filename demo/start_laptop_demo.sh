#!/usr/bin/env bash
set -euo pipefail

# Start the laptop half of the recordable EdgeSplit demo. Phone services are
# intentionally manual: see demo/README.md. This script does not modify or run
# anything in decode/.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf}"
PHONE_HOST="${EDGESPLIT_PHONE_HOST:-YOUR_PHONE_IP}"
LAPTOP_PORT="${EDGESPLIT_LAPTOP_PORT:-8080}"
ROUTER_PORT="${EDGESPLIT_ROUTER_PORT:-8083}"
ROUTER_PYTHON="${ROUTER_PYTHON:-$ROOT/.venv/bin/python}"
LLAMA_SERVER="$LLAMA_CPP_DIR/build-cuda/bin/llama-server"

test -x "$LLAMA_SERVER" || {
  echo "Missing CUDA llama-server: $LLAMA_SERVER" >&2
  echo "Run ./prefill/build_laptop_wsl.sh first." >&2
  exit 2
}
test -x "$ROUTER_PYTHON" || {
  echo "Missing router virtualenv: $ROUTER_PYTHON" >&2
  exit 2
}
test -f "$MODEL_PATH" || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }

mkdir -p "$ROOT/state"
"$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$LAPTOP_PORT" \
  --ctx-size 4096 \
  --n-gpu-layers 999 \
  --parallel 1 \
  --slot-save-path "$ROOT/state" &
LAPTOP_PID=$!

cleanup() {
  kill "${ROUTER_PID:-}" "$LAPTOP_PID" 2>/dev/null || true
  wait "${ROUTER_PID:-}" "$LAPTOP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  curl -fsS "http://127.0.0.1:${LAPTOP_PORT}/health" >/dev/null && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${LAPTOP_PORT}/health" >/dev/null || {
  echo "Laptop llama-server did not become healthy." >&2
  exit 1
}

(
  cd "$ROOT/router"
  EDGESPLIT_LAPTOP_URL="http://127.0.0.1:${LAPTOP_PORT}" \
  EDGESPLIT_PHONE_URL="http://${PHONE_HOST}:8081" \
  EDGESPLIT_PHONE_UPLOAD_URL="http://${PHONE_HOST}:8090" \
  EDGESPLIT_V2_PHONE_URL="http://${PHONE_HOST}:8081" \
  EDGESPLIT_V2_PHONE_HOST="$PHONE_HOST" \
  EDGESPLIT_V2_PHONE_PORT=8091 \
  EDGESPLIT_LAPTOP_STATE_DIR="$ROOT/state" \
  EDGESPLIT_BENCH_DB="$ROOT/bench/edgesplit.sqlite3" \
  EDGESPLIT_QUANT=Q4_K_M \
  EDGESPLIT_RUNTIME_LABEL="laptop:e920c523e3b8a0163fe498af5bf90df35ff51d25;phone:e920c523e3b8a0163fe498af5bf90df35ff51d25;path:v1-file" \
  EDGESPLIT_V2_RUNTIME_LABEL="laptop:e920c523e3b8a0163fe498af5bf90df35ff51d25;phone:e920c523e3b8a0163fe498af5bf90df35ff51d25;patch:edgesplit-v2" \
  "$ROUTER_PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "$ROUTER_PORT"
) &
ROUTER_PID=$!

for _ in $(seq 1 45); do
  curl -fsS "http://127.0.0.1:${ROUTER_PORT}/healthz" >/dev/null && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${ROUTER_PORT}/healthz" || {
  echo "Router did not become healthy." >&2
  exit 1
}

cat <<EOF

Demo services are ready.

V1: curl -fsS -X POST http://127.0.0.1:${ROUTER_PORT}/v1/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"Explain why the sky is blue in one sentence.","n_predict":16,"slot_id":0,"temperature":0,"seed":42}'

V2: curl -fsS -X POST http://127.0.0.1:${ROUTER_PORT}/v2/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"Explain why the sky is blue in one sentence.","n_predict":16,"slot_id":0,"temperature":0,"seed":42}'

Press Ctrl-C when the recording is complete.
EOF

wait "$ROUTER_PID"
