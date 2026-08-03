# Device notes

Hardware, runtime parity, and the two device-specific investigations that shaped
the project's scope.

## Hardware

| Role | Device | Relevant detail |
| --- | --- | --- |
| Prefill | Laptop, RTX 4050 Laptop GPU, 6 GB VRAM | WSL2, CUDA build of llama.cpp |
| Decode (V2, active) | Android phone, rooted, Termux | Exynos 7420, 4× Cortex-A57 @ 2.1 GHz + 4× Cortex-A53 @ 1.5 GHz, 3 GB LPDDR4 |
| Decode (V1, preserved) | Second Android phone, ColorOS | Official `llama-b10034-bin-android-arm64` prebuilt |

The decode device has 3 GB of RAM and an ARMv8.0-A CPU. That is the point rather
than a limitation to apologise for: decode is memory-bandwidth-bound, and this
LPDDR4 part makes the bandwidth ceiling the dominant term. Two decode threads are
used, not eight, because the four A53 efficiency cores contribute little and add
scheduling noise.

It also sets a real memory constraint. A 1B model at Q8_0 plus KV cache on a
3 GB device is tight, which is why Q8_0 completing at all is worth noting: the
preserved V1 phone OOM'd on that configuration.

## Environment gotchas

Four things that cost time and are not obvious:

- **Install Termux from F-Droid, not the Play Store.** The Play Store build is
  frozen at an older version with known compatibility problems for this kind of
  native toolchain work.
- **Clone and build inside Termux's own home directory, never `/sdcard`.**
  Shared storage does not carry Unix permissions, so builds fail on executable
  bits partway through.
- **`termux-wake-lock` must be held for the duration of any run.** Without it
  Android's Doze mode throttles or kills the background server, which looks like
  a network timeout from the laptop side. Every `decode/start_*.sh` script
  acquires it.
- **Check for router AP isolation before writing any networking code.** Many
  consumer access points block client-to-client traffic by default, so the two
  devices cannot see each other even on the same SSID. `ip addr` on both sides
  plus a ping is the fast check.

## Why the dot-product path was never available

The NEON work targeted llama.cpp's baseline ARMv8.0 kernel rather than its
dot-product kernel. That was forced by the ISA, not chosen.

Cortex-A57 and Cortex-A53 are both **ARMv8.0-A**. ARM's `SDOT`/`UDOT`
instructions arrived as an optional extension in **ARMv8.2-A**, so this SoC
cannot execute them at any clock speed.

The measurement agrees exactly:

- `/proc/cpuinfo` features on the decode process:
  `fp asimd aes pmull sha1 sha2 crc32`, the ARMv8.0-A baseline plus the crypto
  extensions, with no `asimddp`.
- `ggml_backend_score()` queried directly on each built variant:
  `libggml-cpu-android_armv8.0_1.so` scores **1**; `android_armv8.2_1` and every
  higher variant score **0**.

So llama.cpp's runtime dispatch correctly selects the baseline variant, and the
only available lever on this device was to make that baseline kernel tighter.
`decode/inspect_cpu_variants.py` performs this check; it loads each `.so` and
calls the scoring symbol through `ctypes` rather than inferring eligibility from
the filename.

## Verifying the NEON change actually took effect

A percentage difference between two builds is worthless if both processes loaded
the same library. Four checks, in order:

1. **Distinct artifacts.** The selected backend `libggml-cpu-android_armv8.0_1.so`
   had different SHA-256 values in `build-termux-neon-stock` and
   `build-termux-neon-optimized`.
2. **Distinct libraries at runtime.** `/proc/<pid>/maps` confirmed each test
   process mapped the `.so` from its own build directory.
3. **Patch state confirmed.** `git apply --reverse --check` verified the NEON
   patch was applied to the final source tree.
4. **Output unchanged.** A fixed-seed, temperature-0, 64-token gate produced
   byte-identical generated text from both binaries after excluding llama-cli's
   timing footer. An optimization that changes numerical results is a bug, not a
   speedup.

The change itself: hold Q4_K partial dot products in `int32x4_t` accumulators via
`vmlaq_n_s32` and perform one horizontal `vaddvq_s32` reduction per superblock,
instead of reducing once per 64-element block. SVE, the scalar fallback, and
runtime dispatch are untouched. See
[`patches/llama.cpp/0002-edgesplit-neon-q4k-q8k-vector-accumulator.patch`](../patches/llama.cpp/0002-edgesplit-neon-q4k-q8k-vector-accumulator.patch).

### Profiling took three attempts

Worth recording because the first two failures are Android-specific and not
obvious:

1. **Direct `simpleperf` launch.** Blocked by Android's app-UID child-process
   restriction.
2. **`su` launch.** Produced no usable elevation or record.
3. **PID attach.** Worked. `simpleperf record -p <pid>` against an already
   running warmed decode.

The resulting profile selected `ggml_vec_dot_q4_K_q8_K` at roughly 9% combined
sample share across call sites; the next matmul-level symbol was about 0.3%.
Full trail in
[`results/neon_q4_k_m_comparison.md`](../results/neon_q4_k_m_comparison.md).

## Runtime parity

Both the laptop and the active decode phone run the same pinned llama.cpp
revision:

```text
e920c523e3b8a0163fe498af5bf90df35ff51d25
```

This is not a nicety. Mismatched builds do not fail cleanly: the sequence-state
format is version-specific, so a mismatch produces a corrupt restore or a hang at
handoff rather than an error message. The V2 transfer frame therefore carries the
commit hash in its metadata and the receiver rejects any frame whose commit does
not match its own, converting a silent corruption into an explicit refusal.

The pinned revision lives in [`config/llama_cpp.env`](../config/llama_cpp.env) and
is shared by the laptop and Termux build scripts, which apply the tracked patches
idempotently.

## The preserved V1 phone's from-source build failure

The V1 device runs the official prebuilt binary rather than a source build,
because its source build crashed during model load. The investigation covered:

- portable CPU dispatch variants
- OpenSSL removal
- context size, mmap, and thread-count settings
- ELF comparison against the working official prebuilt
- an ADB malloc-level backtrace
- a GWP-ASan / API-30 allocator-instrumentation lead, retested with an explicit
  API-28 build target

None resolved it. What settled the question was running the *identical* source
revision on the Exynos 7420 phone, where it works: that establishes the failure as
device- or ROM-specific rather than a general ARM incompatibility or a bug in
EdgeSplit's code.

This is why the two phones have different roles. The V1 file-handoff proof was
recorded on the prebuilt device and is preserved as-is; all V2 work and every
paired comparison uses that phone's source build. The API-28 build target in
`decode/setup_termux.sh` is a residue of that investigation.
