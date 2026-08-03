# Llama-3.2-1B-Instruct V1/V2 Phone 2 comparison

## Scope

This is an additive second-model validation of the existing Phone 2 LAN
experiment. It leaves the Qwen3-0.6B comparison, the Phone 1 V1 proof, and
the same-device V2 fallback intact.

- Laptop: native WSL2 CUDA runtime, llama.cpp
  `e920c523e3b8a0163fe498af5bf90df35ff51d25`.
- Phone: Phone 2 Termux source runtime at the same pinned revision, Android
  API 28, two decode threads, and `--no-mmap`.
- Modes: V1 slot-save file plus HTTP upload and restore; V2 patched raw
  sequence-state export plus `SEQ0` TCP transfer and in-process import.
- Prompt: `Explain why the sky is blue in one sentence.`
- Seed: 42; slot: 0; generated tokens: 64.
- Method: one V1/V2 warm-up pair was explicitly excluded, followed by five
  retained paired samples per quant. Reported spread is sample standard
  deviation.

The source server, V1 receiver, and V2 raw listener were running on Phone 2
throughout. Every retained request returned HTTP 200. Q8_0 completed without
the OOM failure observed on Phone 1.

## Laptop-only baselines

| Quant | Benchmark row | TTFT | Generation |
| --- | ---: | ---: | ---: |
| Q4_0 | 50 | 2.806014 s | 171.7 tok/s |
| Q4_K_M | 51 | 2.343214 s | 176.7 tok/s |
| Q8_0 | 52 | 2.269317 s | 122.5 tok/s |

## Cross-device retained samples

| Quant | Warm-up rows (excluded) | Recorded V1 rows | Recorded V2 rows |
| --- | --- | --- | --- |
| Q4_K_M | 53, 54 | 55, 57, 59, 61, 63 | 56, 58, 60, 62, 64 |
| Q4_0 | 65, 66 | 67, 69, 71, 73, 75 | 68, 70, 72, 74, 76 |
| Q8_0 | 77, 78 | 79, 81, 83, 85, 87 | 80, 82, 84, 86, 88 |

| Quant | V1 TTFT mean +/- sd (median) | V2 TTFT mean +/- sd (median) | V1 decode mean +/- sd (median) | V2 decode mean +/- sd (median) |
| --- | ---: | ---: | ---: | ---: |
| Q4_K_M | 0.771 +/- 0.090 s (0.747) | 0.754 +/- 0.068 s (0.762) | 2.641 +/- 0.155 tok/s (2.608) | 2.580 +/- 0.119 tok/s (2.561) |
| Q4_0 | 0.839 +/- 0.354 s (0.703) | 0.659 +/- 0.070 s (0.645) | 3.303 +/- 0.135 tok/s (3.358) | 3.323 +/- 0.087 tok/s (3.374) |
| Q8_0 | 1.227 +/- 0.371 s (1.123) | 0.723 +/- 0.021 s (0.726) | 2.823 +/- 0.237 tok/s (2.924) | 2.763 +/- 0.131 tok/s (2.747) |

V2 changed mean TTFT by -2.16% (Q4_K_M), -21.45% (Q4_0), and -41.06%
(Q8_0). Mean decode changed by -2.30%, +0.59%, and -2.12%, respectively.
The mechanism changes handoff, not Phone 2 decode, so the decode figures are
reported for completeness rather than as an optimization claim.

The machine-readable retained samples and summaries are:

- `results/llama32_q4_k_m_v1_v2_repetitions.json`
- `results/llama32_q4_0_v1_v2_repetitions.json`
- `results/llama32_q8_0_v1_v2_repetitions.json`

The benchmark CSV was exported from the native WSL SQLite database after the
runs. It retains all earlier evidence, including Phone 1, Qwen, V1 smoke, V2
smoke, and NEON rows.

## Later five-pair run with laptop GPU power

This additive later pass reran the exact Llama-3.2-1B-Instruct protocol above
with laptop GPU board power sampled every 100 ms through
`nvidia-smi --query-gpu=power.draw`. It retains one excluded V1/V2 warm-up pair
and five pairs per mode/quant. It does not replace the timing-only raw
artifacts above. Values are **mean +/- sample standard deviation (median)**.

| Quant | V1 rows | V2 rows | V1 TTFT | V2 TTFT | V1 decode | V2 decode | Laptop GPU draw, V1 / V2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 115, 117, 119, 121, 123 | 116, 118, 120, 122, 124 | 0.697 +/- 0.028 s (0.702) | 0.614 +/- 0.012 s (0.616) | 2.809 +/- 0.153 (2.792) | 2.797 +/- 0.142 (2.831) | 2.62 +/- 0.55 / 2.16 +/- 0.32 W* |
| Q4_0 | 127, 129, 131, 133, 135 | 128, 130, 132, 134, 136 | 0.812 +/- 0.298 s (0.695) | 0.620 +/- 0.026 s (0.624) | 2.879 +/- 0.195 (2.909) | 2.893 +/- 0.105 (2.857) | 1.82 +/- 0.06 / 1.81 +/- 0.03 W |
| Q8_0 | 103, 105, 107, 109, 111 | 104, 106, 108, 110, 112 | 0.774 +/- 0.050 s (0.762) | 0.761 +/- 0.090 s (0.723) | 2.318 +/- 0.139 (2.310) | 2.249 +/- 0.084 (2.216) | 2.02 +/- 0.38 / 2.29 +/- 0.51 W |

Excluded warm-up rows are 113--114 (Q4_K_M), 125--126 (Q4_0), and 101--102
(Q8_0). These GPU numbers are board-power samples, not energy or whole-system
power. They cover the entire router request and therefore include time when
the phone is decoding and laptop GPU activity is low.

*The Q4_K_M GPU cells use the separate 2026-07-20 stability rerun (rows
223--232), which retained the same sampling configuration but ran through the
current model-native chat template. The table's Q4_K_M TTFT/decode cells remain
the original powered pass.

### Q4_K_M laptop-GPU-power stability rerun (2026-07-20)

The first Q4_K_M powered pass had unusually high cross-run GPU-power variation:
3.04 +/- 2.28 W for V1 (75.1% relative standard deviation) and 3.87 +/- 2.67 W
for V2 (69.1%). A separate rerun preserved that artifact and repeated the same
method: one excluded V1/V2 warm-up pair, five retained pairs, 64 output tokens,
seed 42, and 100 ms nvidia-smi polling around each whole router request. Phone
battery power remained omitted.

| Artifact | V1 retained rows | V2 retained rows | V1 laptop GPU draw | V2 laptop GPU draw |
| --- | --- | --- | ---: | ---: |
| Original powered pass | 115, 117, 119, 121, 123 | 116, 118, 120, 122, 124 | 3.04 +/- 2.28 W (75.1%) | 3.87 +/- 2.67 W (69.1%) |
| Stability rerun | 223, 225, 227, 229, 231 | 224, 226, 228, 230, 232 | 2.62 +/- 0.55 W (20.9%) | 2.16 +/- 0.32 W (14.6%) |

Warm-up rows 221--222 are explicitly invalidated. The rerun did not reproduce
the original high-variance GPU windows. Both machine-readable artifacts remain
tracked: results/llama32_q4_k_m_v1_v2_laptop_gpu_power_repetitions.json and
results/llama32_q4_k_m_v1_v2_laptop_gpu_power_repetitions_rerun_20260720.json.
This is a whole-request board-power stability check, not a system-power or
energy comparison. The warm-up, repetition, and polling parameters matched the
original pass, but the current model-native chat template produced a 46-token
prompt and stopped at 56 output tokens; the earlier powered pass used the
former raw-completion workload. It is therefore a current-path stability
follow-up, not a workload-identical power replication.

Phone 2 battery power is intentionally excluded. The initial
vendor fuel-gauge sampler yielded only 2/176 and 1/180 plausible samples in
later plugged and unplugged windows; SQLite rows 89--100 remain as invalid
calibration evidence. A replacement `termux-battery-status` sampler correctly
used its `current` (uA) and `voltage` (mV) fields, not `current_average`, and
produced plausible zero-drop samples at 750 ms. Consecutive controlled idle
windows still had large whole-device state changes, including screen-off
medians of 614.754 and 411.264 mW with ranges up to 2.443 W. The source is
therefore not a stable per-request baseline; no phone-power figure is claimed.
