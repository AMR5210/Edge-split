#!/data/data/com.termux/files/usr/bin/bash
# Build comparable stock and optimized Phone 2 binaries without changing V1/V2.
set -euo pipefail

ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
SOURCE_DIR="${LLAMA_SOURCE_DIR:-$HOME/src/llama.cpp}"
COMMIT="e920c523e3b8a0163fe498af5bf90df35ff51d25"
TERMUX_ANDROID_API="${TERMUX_ANDROID_API:-28}"
ANDROID_TARGET="aarch64-linux-android$TERMUX_ANDROID_API"
NEON_PATCH="$ROOT/patches/llama.cpp/0002-edgesplit-neon-q4k-q8k-vector-accumulator.patch"
STOCK_BUILD_DIR="${STOCK_BUILD_DIR:-$SOURCE_DIR/build-termux-neon-stock}"
OPT_BUILD_DIR="${OPT_BUILD_DIR:-$SOURCE_DIR/build-termux-neon-optimized}"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  echo "llama.cpp checkout not found: $SOURCE_DIR" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "llama.cpp must be at $COMMIT" >&2
  exit 2
fi
if [[ ! -f "$NEON_PATCH" ]]; then
  echo "NEON patch not found: $NEON_PATCH" >&2
  exit 2
fi
if ! git -C "$SOURCE_DIR" apply --reverse --check "$ROOT/patches/llama.cpp/0001-edgesplit-v2-raw-sequence-endpoints.patch"; then
  echo "The required V2 patch is not applied to $SOURCE_DIR" >&2
  exit 2
fi

build() {
  local build_dir="$1"
  rm -rf "$build_dir"
  cmake -S "$SOURCE_DIR" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-target $ANDROID_TARGET" \
    -DCMAKE_CXX_FLAGS="-target $ANDROID_TARGET" \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF
  cmake --build "$build_dir" --config Release -j"$(nproc)"
}

neon_was_applied=0
if git -C "$SOURCE_DIR" apply --reverse --check "$NEON_PATCH"; then
  neon_was_applied=1
  git -C "$SOURCE_DIR" apply --reverse "$NEON_PATCH"
elif ! git -C "$SOURCE_DIR" apply --check "$NEON_PATCH"; then
  echo "NEON patch cannot be applied or reversed cleanly" >&2
  exit 2
fi

restore_neon() {
  if [[ "$neon_was_applied" -eq 1 ]] && git -C "$SOURCE_DIR" apply --check "$NEON_PATCH"; then
    git -C "$SOURCE_DIR" apply "$NEON_PATCH"
  fi
}
trap restore_neon EXIT

echo "Building stock binary: $STOCK_BUILD_DIR"
build "$STOCK_BUILD_DIR"
git -C "$SOURCE_DIR" apply "$NEON_PATCH"
echo "Building optimized binary: $OPT_BUILD_DIR"
build "$OPT_BUILD_DIR"
echo "Stock binary: $STOCK_BUILD_DIR/bin/llama-cli"
echo "Optimized binary: $OPT_BUILD_DIR/bin/llama-cli"
