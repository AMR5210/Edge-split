#!/data/data/com.termux/files/usr/bin/bash
# Capture a symbol-level CPU profile for one long, warmed decode on Phone 2.
# Run this manually on the rooted phone only; it does not modify llama.cpp.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the GGUF model path}"

ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-$HOME/src/llama.cpp/build-termux/bin}"
LLAMA_CLI="${LLAMA_CLI:-$LLAMA_BIN_DIR/llama-cli}"
DECODE_THREADS="${DECODE_THREADS:-2}"
WARMUP_TOKENS="${PROFILE_WARMUP_TOKENS:-64}"
PROFILE_TOKENS="${PROFILE_OUTPUT_TOKENS:-512}"
PROFILE_PROMPT="${PROFILE_PROMPT:-Explain why the sky is blue in one sentence.}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/results}"
PROFILE_MODE="${PROFILE_MODE:-root-launch}"
SU_BIN="${SU_BIN:-su}"
ATTACH_DELAY_SECONDS="${PROFILE_ATTACH_DELAY_SECONDS:-5}"
ATTACH_DURATION_SECONDS="${PROFILE_ATTACH_DURATION_SECONDS:-120}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PROFILE_DIR="$PROFILE_ROOT/neon-profile-$STAMP"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "llama-cli not executable: $LLAMA_CLI" >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 2
fi

if [[ -n "${SIMPLEPERF_BIN:-}" ]]; then
  SIMPLEPERF="$SIMPLEPERF_BIN"
elif command -v simpleperf >/dev/null 2>&1; then
  SIMPLEPERF="$(command -v simpleperf)"
elif [[ -x /system/bin/simpleperf ]]; then
  SIMPLEPERF=/system/bin/simpleperf
else
  echo "simpleperf is unavailable. Set SIMPLEPERF_BIN to a working simpleperf binary." >&2
  exit 2
fi

export LD_LIBRARY_PATH="$LLAMA_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$PROFILE_DIR"
termux-wake-lock

common=(
  --model "$MODEL_PATH"
  --prompt "$PROFILE_PROMPT"
  --no-mmap
  --threads "$DECODE_THREADS"
  --ignore-eos
  --single-turn
)

printf 'llama_cli=%s\nmodel=%s\nthreads=%s\nwarmup_tokens=%s\nprofile_tokens=%s\nprofile_mode=%s\nattach_delay_seconds=%s\nattach_duration_seconds=%s\n' \
  "$LLAMA_CLI" "$MODEL_PATH" "$DECODE_THREADS" "$WARMUP_TOKENS" "$PROFILE_TOKENS" "$PROFILE_MODE" "$ATTACH_DELAY_SECONDS" "$ATTACH_DURATION_SECONDS" \
  | tee "$PROFILE_DIR/metadata.txt"
"$SIMPLEPERF" --version 2>&1 | tee -a "$PROFILE_DIR/metadata.txt"

echo "Warm-up: $WARMUP_TOKENS tokens (not profiled)"
"$LLAMA_CLI" "${common[@]}" --n-predict "$WARMUP_TOKENS" \
  >"$PROFILE_DIR/warmup.stdout" 2>"$PROFILE_DIR/warmup.stderr"

echo "Profiling: $PROFILE_TOKENS tokens"
set +e
status=0
record_cmd=(
  env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  "$SIMPLEPERF" record --call-graph fp -o "$PROFILE_DIR/perf.data" --
  "$LLAMA_CLI" "${common[@]}" --n-predict "$PROFILE_TOKENS"
)
case "$PROFILE_MODE" in
  root-launch)
    if ! command -v "$SU_BIN" >/dev/null 2>&1; then
      echo "su is unavailable. Set SU_BIN or use PROFILE_MODE=direct for diagnosis." >&2
      exit 2
    fi
    printf -v root_command '%q ' "${record_cmd[@]}"
    "$SU_BIN" -c "$root_command" >"$PROFILE_DIR/profile.stdout" 2>"$PROFILE_DIR/profile.stderr"
    status=$?
    ;;
  direct)
    "${record_cmd[@]}" >"$PROFILE_DIR/profile.stdout" 2>"$PROFILE_DIR/profile.stderr"
    status=$?
    ;;
  attach)
    "$LLAMA_CLI" "${common[@]}" --n-predict "$PROFILE_TOKENS" \
      >"$PROFILE_DIR/profile.stdout" 2>"$PROFILE_DIR/profile.stderr" &
    cli_pid=$!
    printf 'attach_pid=%s\n' "$cli_pid" | tee -a "$PROFILE_DIR/metadata.txt"
    sleep "$ATTACH_DELAY_SECONDS"
    if ! kill -0 "$cli_pid" 2>/dev/null; then
      wait "$cli_pid"
      status=3
      echo "llama-cli exited before simpleperf could attach" >&2
    else
      "$SIMPLEPERF" record --call-graph fp --duration "$ATTACH_DURATION_SECONDS" \
        -p "$cli_pid" -o "$PROFILE_DIR/perf.data" \
        >"$PROFILE_DIR/simpleperf.stdout" 2>"$PROFILE_DIR/simpleperf.stderr"
      perf_status=$?
      wait "$cli_pid"
      cli_status=$?
      if [[ "$perf_status" -ne 0 ]]; then
        status="$perf_status"
      elif [[ "$cli_status" -ne 0 ]]; then
        status="$cli_status"
      fi
    fi
    ;;
  *)
    echo "Unsupported PROFILE_MODE: $PROFILE_MODE (use root-launch, attach, or direct)" >&2
    status=2
    ;;
esac
set -e
if [[ "$status" -ne 0 ]]; then
  echo "simpleperf record failed with status $status; see $PROFILE_DIR/profile.stderr" >&2
  exit "$status"
fi

"$SIMPLEPERF" report -i "$PROFILE_DIR/perf.data" -o "$PROFILE_DIR/report.txt"
grep -E 'ggml_vec_dot_(q4_0_q8_0|q8_0_q8_0|q4_K_q8_K)|ggml_compute_forward_mul_mat' \
  "$PROFILE_DIR/report.txt" >"$PROFILE_DIR/candidate-kernels.txt" || true

echo "Profile complete: $PROFILE_DIR"
echo "Candidate kernel samples:"
if [[ -s "$PROFILE_DIR/candidate-kernels.txt" ]]; then
  cat "$PROFILE_DIR/candidate-kernels.txt"
else
  echo "No candidate symbols found. Preserve report.txt and profile.stderr for diagnosis."
fi
