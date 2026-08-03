#!/data/data/com.termux/files/usr/bin/bash
# Compare a Termux-built llama-cli against an official Android release binary.
# This script only inspects ELF metadata; it never executes either binary.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 /path/to/source/llama-cli /path/to/prebuilt/llama-cli" >&2
  exit 2
fi

SOURCE_BIN="$1"
PREBUILT_BIN="$2"
BUILD_DIR="$HOME/src/llama.cpp/build-termux"

for binary in "$SOURCE_BIN" "$PREBUILT_BIN"; do
  if [[ ! -f "$binary" ]]; then
    echo "binary not found: $binary" >&2
    exit 2
  fi
done

READELF="$(command -v llvm-readelf || command -v readelf || true)"
if [[ -z "$READELF" ]]; then
  echo "Neither llvm-readelf nor readelf is installed. Install binutils and retry." >&2
  exit 2
fi

section() {
  printf '\n===== %s =====\n' "$1"
}

inspect_binary() {
  local label="$1"
  local binary="$2"

  section "$label: identity"
  printf 'path: %s\n' "$binary"
  command -v file >/dev/null && file "$binary" || true
  sha256sum "$binary"

  section "$label: ELF header and program interpreter"
  "$READELF" -hW "$binary"
  "$READELF" -lW "$binary" | grep -E 'Requesting program interpreter|LOAD|GNU_STACK|GNU_RELRO' || true

  section "$label: dynamic section (linked shared objects, RPATH/RUNPATH)"
  "$READELF" -dW "$binary"

  section "$label: build notes"
  "$READELF" -nW "$binary" || true

  section "$label: unresolved dynamic symbols"
  "$READELF" -Ws "$binary" | grep ' UND ' || true
}

section "device and Termux environment"
uname -a
getprop ro.build.version.sdk || true
getprop ro.product.cpu.abi || true
printf 'TERMUX_PREFIX=%s\n' "$PREFIX"
"$PREFIX/bin/clang" --version || true
cmake --version || true

section "source-build compiler recorded by CMake"
grep -R -E 'CMAKE_(C|CXX)_COMPILER(:|_VERSION)' "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/CMakeFiles" 2>/dev/null || true

inspect_binary "from-source" "$SOURCE_BIN"
inspect_binary "official-prebuilt" "$PREBUILT_BIN"

section "diff: dynamic section (official -> from-source)"
diff -u <("$READELF" -dW "$PREBUILT_BIN") <("$READELF" -dW "$SOURCE_BIN") || true

section "diff: program headers (official -> from-source)"
diff -u <("$READELF" -lW "$PREBUILT_BIN") <("$READELF" -lW "$SOURCE_BIN") || true

section "diff: unresolved dynamic symbols (official -> from-source)"
diff -u \
  <("$READELF" -Ws "$PREBUILT_BIN" | grep ' UND ' || true) \
  <("$READELF" -Ws "$SOURCE_BIN" | grep ' UND ' || true) || true
