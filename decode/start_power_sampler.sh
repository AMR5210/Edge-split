#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
HOST="${EDGESPLIT_POWER_HOST:-0.0.0.0}"
PORT="${EDGESPLIT_POWER_PORT:-8092}"
INTERVAL_MS="${EDGESPLIT_POWER_INTERVAL_MS:-750}"
COMMAND_TIMEOUT_SECONDS="${EDGESPLIT_POWER_COMMAND_TIMEOUT_SECONDS:-3}"
MIN_SAMPLES="${EDGESPLIT_POWER_MIN_SAMPLES:-3}"
MAX_CURRENT_AMPS="${EDGESPLIT_POWER_MAX_CURRENT_AMPS:-10}"
MIN_VOLTAGE_VOLTS="${EDGESPLIT_POWER_MIN_VOLTAGE_VOLTS:-2.5}"
MAX_VOLTAGE_VOLTS="${EDGESPLIT_POWER_MAX_VOLTAGE_VOLTS:-5.0}"

if ! command -v termux-battery-status >/dev/null 2>&1; then
  echo "termux-battery-status is required; install/configure Termux:API first" >&2
  exit 1
fi

termux-wake-lock
exec python "$ROOT/decode/power_sampler.py" \
  --host "$HOST" --port "$PORT" --interval-ms "$INTERVAL_MS" \
  --command-timeout-seconds "$COMMAND_TIMEOUT_SECONDS" --min-samples "$MIN_SAMPLES" \
  --max-current-amps "$MAX_CURRENT_AMPS" \
  --min-voltage-volts "$MIN_VOLTAGE_VOLTS" --max-voltage-volts "$MAX_VOLTAGE_VOLTS"
