#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT="${EDGESPLIT_ROOT:-$HOME/edgesplit}"
REPOSITORY="https://github.com/ggml-org/llama.cpp.git"
COMMIT="e920c523e3b8a0163fe498af5bf90df35ff51d25"
TERMUX_ANDROID_API="${TERMUX_ANDROID_API:-28}"
ANDROID_TARGET="aarch64-linux-android$TERMUX_ANDROID_API"
EDGESPLIT_PATCH_DIR="$ROOT/patches/llama.cpp"

pkg update -y
pkg install -y git cmake clang make python termux-api libandroid-spawn
termux-wake-lock

mkdir -p "$HOME/src"
if [[ ! -d "$HOME/src/llama.cpp/.git" ]]; then
  git clone "$REPOSITORY" "$HOME/src/llama.cpp"
fi
git -C "$HOME/src/llama.cpp" fetch --tags --force
git -C "$HOME/src/llama.cpp" checkout --detach "$COMMIT"
ACTUAL_COMMIT="$(git -C "$HOME/src/llama.cpp" rev-parse HEAD)"
echo "llama.cpp commit: $ACTUAL_COMMIT"
echo "Android target: $ANDROID_TARGET"

shopt -s nullglob
EDGESPLIT_PATCHES=("$EDGESPLIT_PATCH_DIR"/*.patch)
if [[ "${#EDGESPLIT_PATCHES[@]}" -eq 0 ]]; then
  echo "No EdgeSplit patches found in $EDGESPLIT_PATCH_DIR" >&2
  exit 1
fi
for patch in "${EDGESPLIT_PATCHES[@]}"; do
  if git -C "$HOME/src/llama.cpp" apply --reverse --check "$patch"; then
    echo "EdgeSplit patch already applied: $(basename "$patch")"
  elif git -C "$HOME/src/llama.cpp" apply --check "$patch"; then
    git -C "$HOME/src/llama.cpp" apply "$patch"
    echo "EdgeSplit patch applied: $(basename "$patch")"
  else
    echo "EdgeSplit patch cannot be applied cleanly: $patch" >&2
    exit 1
  fi
done

BUILD_DIR="$HOME/src/llama.cpp/build-termux"

# Match the portable release CPU dispatch and explicitly target API 28. The
# latter avoids Android API 30+ allocator instrumentation while we determine
# whether it exposes the observed Termux heap crash.
rm -rf "$BUILD_DIR"
cmake -S "$HOME/src/llama.cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-target $ANDROID_TARGET" \
  -DCMAKE_CXX_FLAGS="-target $ANDROID_TARGET" \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
"$BUILD_DIR/bin/llama-server" --version

echo "Copy $ROOT/bench/edgesplit_bench.py to this checkout before phone benchmarks."
