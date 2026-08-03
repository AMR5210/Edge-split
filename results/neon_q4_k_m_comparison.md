# Q4_K_M NEON accumulator comparison - Phone 2

Date: 2026-07-18

## Result

One stock/optimized warm-up pair was excluded. Five retained pairs used the
same Phone 2, Qwen3-0.6B Q4_K_M model, 64 generated tokens, fixed seed 42,
temperature 0, two decode threads, `--no-mmap`, and standalone `llama-cli`.
The V1 receiver, V2 listener, and llama-server were stopped for every run.
Values are mean +/- sample standard deviation.

| Build | TTFT | Decode throughput |
| --- | ---: | ---: |
| Stock | 11.20 +/- 0.36 s | 6.28 +/- 0.22 tok/s |
| NEON accumulator | 9.96 +/- 0.07 s | 6.58 +/- 0.23 tok/s |

The optimized library reduced mean TTFT by 11.1% and increased mean decode
throughput by 4.8%. The TTFT sample standard deviation fell from 0.36 s to
0.07 s, about 5.1x lower. This is a modest, five-pair, device-specific result.

## Target and implementation

The successful PID-attach profile selected `ggml_vec_dot_q4_K_q8_K`, with about
9% combined sample share across its call sites. `ggml_compute_forward_mul_mat`
was a distant second at about 0.3%. The optimized patch is
[`0002-edgesplit-neon-q4k-q8k-vector-accumulator.patch`](../patches/llama.cpp/0002-edgesplit-neon-q4k-q8k-vector-accumulator.patch).
It retains Q4_K partial dot products in NEON `int32x4_t` accumulators and
performs one horizontal reduction per superblock. The SVE path, scalar
fallback, runtime dispatch, and all V1/V2 code paths remain unchanged.

## Verification trail

1. Profiling used three approaches: direct simpleperf launch was blocked by
   Android's app-UID child-process restriction; the `su` launch produced no
   usable elevation/record; PID attach succeeded. The first attach record used
   an unsupported `report --stdio` option, then the verified Android
   `report -o` option generated the readable profile.
2. `git apply --reverse --check` confirmed the NEON patch was applied to the
   final source tree.
3. The actual selected backend artifacts,
   `libggml-cpu-android_armv8.0_1.so`, had distinct SHA-256 values in
   `build-termux-neon-stock` and `build-termux-neon-optimized`.
4. `/proc/<pid>/maps` confirmed each test process mapped its own directory's
   distinct `armv8.0_1` library. The original successful profile also showed
   that same variant executing on Phone 2.
5. A fixed-seed, temperature-zero 64-token correctness gate produced identical
   generated text from stock and optimized binaries after excluding llama-cli's
   timing footer.
6. The final variant eligibility inspection read the Phone 2 process features
   as `fp asimd aes pmull sha1 sha2 crc32`. `android_armv8.0_1` scored 1;
   `android_armv8.2_1` and every higher Android variant scored 0. The dotprod
   path is not eligible on this runtime, so no higher-variant benchmark was
   run.

The raw phone evidence is retained on Phone 2 at
`~/edgesplit/results/neon_q4_k_m_repetitions.json`; it contains the exact
retained and excluded SQLite row IDs and all summary statistics. It is not
mirrored into this directory.
