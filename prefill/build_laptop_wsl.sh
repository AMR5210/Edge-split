#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/llama_cpp.env"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
EDGESPLIT_V2_PATCH="$ROOT/patches/llama.cpp/0001-edgesplit-v2-raw-sequence-endpoints.patch"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable inside WSL2" >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone "$LLAMA_CPP_REPOSITORY" "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" fetch --tags --force
git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_COMMIT"
ACTUAL_COMMIT="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "llama.cpp commit: $ACTUAL_COMMIT"

if git -C "$LLAMA_CPP_DIR" apply --reverse --check "$EDGESPLIT_V2_PATCH"; then
  echo "EdgeSplit V2 patch: already applied"
elif git -C "$LLAMA_CPP_DIR" apply --check "$EDGESPLIT_V2_PATCH"; then
  git -C "$LLAMA_CPP_DIR" apply "$EDGESPLIT_V2_PATCH"
  echo "EdgeSplit V2 patch: applied"
else
  echo "EdgeSplit V2 patch cannot be applied cleanly; source checkout is not the pinned baseline" >&2
  exit 1
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build-cuda" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build-cuda" --config Release -j"$(nproc)"
"$LLAMA_CPP_DIR/build-cuda/bin/llama-server" --version
