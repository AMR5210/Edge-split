#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EDGESPLIT_DASHBOARD_HOST:-127.0.0.1}"
PORT="${EDGESPLIT_DASHBOARD_PORT:-8084}"

exec python3 "$ROOT/dashboard/server.py" --host "$HOST" --port "$PORT"
