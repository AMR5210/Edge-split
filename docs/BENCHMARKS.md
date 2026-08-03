# Benchmarks and method

Every number in the README comes from here. This document records the full
method, the raw artifacts, and the exact SQLite row IDs behind each claim, plus
the one metric that was measured and then deliberately excluded.

## Contents

- [Measurement protocol](#measurement-protocol)
- [What the numbers do and do not show](#what-the-numbers-do-and-do-not-show)
- [Qwen3-0.6B: V1 versus V2](#qwen3-06b-v1-versus-v2)
- [Llama-3.2-1B-Instruct: V1 versus V2](#llama-32-1b-instruct-v1-versus-v2)
- [Laptop GPU power stability rerun](#laptop-gpu-power-stability-rerun)
- [Phone battery power: measured, then excluded](#phone-battery-power-measured-then-excluded)
- [Single-device baselines](#single-device-baselines)
- [NEON kernel comparison](#neon-kernel-comparison)
- [Prompt formatting change](#prompt-formatting-change)
- [Raw artifacts](#raw-artifacts)

## Measurement protocol

Identical for every paired comparison below:

| Parameter | Value |
| --- | --- |
| Repetitions | 1 warm-up pair (excluded), then 5 retained pairs per mode |
| Ordering | Strictly paired: V1 then V2, back to back, per repetition |
| Sampling | `temperature = 0`, `seed = 42`, `slot = 0` |
| Statistics | Mean, median, sample standard deviation |
| Laptop GPU power | `nvidia-smi --query-gpu=power.draw`, polled every 100 ms |
| Storage | SQLite; every run writes one row, warm-ups included |
| Filesystem | Native WSL2 storage, never a `/mnt/c` Windows path |

The filesystem row matters. V1 writes a 1.26 MB state file per request, so DrvFs
overhead on `/mnt/c` would inflate V1's timings and invalidate the comparison.
Every number here was measured from the WSL2 home directory.

Three implementation details make the numbers auditable rather than
self-reported:

- **Warm-ups are invalidated, not deleted.** [`invalidate_run`](../bench/edgesplit_bench.py)
  sets `status = 'invalid'` and appends the reason to the row's notes. Excluded
  repetitions stay in the database and can be inspected.
- **An attempt row is written before inference starts.** If the Android process
  is killed mid-run, the attempt is durably visible as `started` and never
  `completed`, so a crash cannot silently disappear from the record.
- **Power capture failure invalidates its own run.** In
  [`run_power_split_repetitions.py`](../bench/run_power_split_repetitions.py), if
  the GPU or phone sampler fails to produce a complete window, the associated
  benchmark row is invalidated rather than kept with a partial measurement.

The JSON evidence files record the exact retained and excluded row IDs for
every summary, so any table here can be traced back to individual runs.

## What the numbers do and do not show

Read this before the tables.

**V2 lowers time-to-first-token.** That is the measured effect, and the
mechanism is not subtle: V1 writes a ~1.26 MB state file to disk, uploads it
over HTTP, and restores it from disk on the phone; V2 streams the same bytes
over one TCP connection into an in-process import. A socket beating a file
round-trip is the expected outcome, not a surprising one. The engineering claim
is that the raw sequence-state handoff works at all across two architectures,
not that the percentage is impressive.

**V2 does not target decode.** Both modes decode on the same phone with the same
runtime, so there is no mechanism by which V2 would change generation speed.
Measured deltas across the six cells below run from -10.7% to +0.5%. Most sit
inside the run-to-run spread; the exception is Qwen Q8_0 at -10.7%, where the
gap is wider than the sample standard deviations account for. That cell is not
explained, and no decode claim is made in either direction.

**The split is slower end-to-end than the laptop alone.** Compare the phone's
2.2-9.3 tok/s against the [single-device baselines](#single-device-baselines) at
122-241 tok/s. The split trades generation throughput for freeing the laptop GPU
after prefill. It is not a way to make inference faster.

**Scope.** Two model families, three quantizations, one laptop and one phone on
one LAN, five retained pairs per cell. This is not a device survey or a
statistical study.

## Qwen3-0.6B: V1 versus V2

11 prompt tokens, 16 generated tokens. Values are **mean ± sample standard
deviation (median)**. GPU draw is laptop GPU board power over the whole router
request.

| Quant | V1 TTFT | V2 TTFT | V1 decode | V2 decode | GPU draw V1 / V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 2.425 ± 1.262 s (2.022) | 0.841 ± 0.272 s (0.793) | 8.442 ± 0.409 tok/s (8.663) | 8.078 ± 0.356 tok/s (8.024) | 2.17 ± 0.16 / 2.38 ± 0.08 W |
| Q4_0 | 3.569 ± 0.326 s (3.406) | 0.632 ± 0.043 s (0.650) | 9.280 ± 0.081 tok/s (9.262) | 8.979 ± 0.227 tok/s (9.018) | 2.08 ± 0.08 / 2.49 ± 0.14 W |
| Q8_0 | 3.404 ± 0.490 s (3.681) | 0.610 ± 0.084 s (0.594) | 8.128 ± 0.346 tok/s (8.241) | 7.261 ± 0.523 tok/s (6.947) | 2.07 ± 0.11 / 2.40 ± 0.25 W |

V2 reduced mean TTFT by **65.32%** (Q4_K_M), **82.30%** (Q4_0), and **82.09%**
(Q8_0). V1's TTFT variance is consistently higher, which is consistent with
filesystem and HTTP-upload timing on the phone.

GPU draw is board power sampled across the entire request, which includes the
period when the phone is decoding and the laptop GPU is largely idle. It is not
energy and not whole-system power.

Detail, per-run rows, and the earlier timing-only pass:
[`results/qwen_v1_v2_comparison.md`](../results/qwen_v1_v2_comparison.md).

## Llama-3.2-1B-Instruct: V1 versus V2

Same protocol, second model family. 12 prompt tokens, 64 generated tokens.

| Quant | V1 TTFT | V2 TTFT | V1 decode | V2 decode | GPU draw V1 / V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 0.697 ± 0.028 s (0.702) | 0.614 ± 0.012 s (0.616) | 2.809 ± 0.153 tok/s (2.792) | 2.797 ± 0.142 tok/s (2.831) | 2.62 ± 0.55 / 2.16 ± 0.32 W [^1] |
| Q4_0 | 0.812 ± 0.298 s (0.695) | 0.620 ± 0.026 s (0.624) | 2.879 ± 0.195 tok/s (2.909) | 2.893 ± 0.105 tok/s (2.857) | 1.82 ± 0.06 / 1.81 ± 0.03 W |
| Q8_0 | 0.774 ± 0.050 s (0.762) | 0.761 ± 0.090 s (0.723) | 2.318 ± 0.139 tok/s (2.310) | 2.249 ± 0.084 tok/s (2.216) | 2.02 ± 0.38 / 2.29 ± 0.51 W |

V2 lowered mean TTFT by **11.89%** (Q4_K_M), **23.56%** (Q4_0), and **1.67%**
(Q8_0). Decode changed between -2.98% and +0.50%.

The Q4_K_M and Q8_0 TTFT gaps sit within their observed spread, so this model
family shows the handoff working across a second architecture rather than a
reliable per-model speedup. The larger model also shifts the balance: at 64
output tokens on a 1B model, phone decode dominates total request time, so the
handoff saving is a smaller fraction of the whole.

Detail and row IDs:
[`results/llama32_v1_v2_phone2_comparison.md`](../results/llama32_v1_v2_phone2_comparison.md).

[^1]: These two GPU cells come from the separate stability rerun below, not from
the same pass as the TTFT and decode cells in that row. See the next section.

## Laptop GPU power stability rerun

The first Llama Q4_K_M powered pass produced unusually wide GPU-power spread:
3.04 ± 2.28 W for V1 and 3.87 ± 2.67 W for V2, or relative standard deviations
of 75.1% (V1) and 69.1% (V2), driven by a few high-average sampling windows.

A rerun repeated the same protocol (one excluded warm-up pair, five retained
pairs, 64 output tokens, seed 42, 100 ms polling) while preserving the original
artifact rather than overwriting it.

| Artifact | V1 rows | V2 rows | V1 GPU draw | V2 GPU draw |
| --- | --- | --- | ---: | ---: |
| Original powered pass | 115, 117, 119, 121, 123 | 116, 118, 120, 122, 124 | 3.04 ± 2.28 W (75.1%) | 3.87 ± 2.67 W (69.1%) |
| Stability rerun | 223, 225, 227, 229, 231 | 224, 226, 228, 230, 232 | 2.62 ± 0.55 W (20.9%) | 2.16 ± 0.32 W (14.6%) |

The high variance did not reproduce. Warm-up rows 221-222 are invalidated.

One caveat keeps this honest: the warm-up, repetition, and polling parameters
matched, but the rerun ran through the current model-native chat template, which
produced a 46-token prompt and stopped at 56 output tokens, while the original
pass used the earlier raw-completion workload. It therefore confirms stability
under the current code path and is **not** a workload-identical replication.

## Phone battery power: measured, then excluded

No phone power figure appears anywhere in this project. Two independent sources
were built and instrumented, and both were rejected. This section records why,
because the absence is a result.

**Attempt 1: the vendor fuel-gauge sysfs node (`current_avg`).** Physically
invalid. Range filtering rejected 162 of 176 samples in one plugged window and
179 of 180 in one unplugged window, with individual readings as high as 40 A and
251.406 A on a phone battery. SQLite rows 89-100 are retained,
marked invalid, as calibration evidence.

**Attempt 2: Termux:API `termux-battery-status`.** Mechanically sound. It used
the documented `current` (µA) and `voltage` (mV) fields while deliberately
ignoring the inconsistent `current_average` field, and returned 41-45 plausible
samples per 31-34 s window at a 750 ms cadence with zero dropped samples and
generally no cadence overruns.

It was still rejected. Consecutive **idle** control windows disagreed with each
other: screen-off medians of 614.754 mW and 411.264 mW back to back, with
within-window ranges reaching 2.443 W and 2.384 W. Screen-on checks were
similarly bimodal. The instrument reports whole-device power, and on this device
that baseline does not hold still long enough to attribute the power of a
sub-second state handoff to the handoff itself.

A number could have been published. It would not have meant anything, so it
was not. The sampler itself is kept in
[`decode/power_sampler.py`](../decode/power_sampler.py) with the plausibility
bounds and drop counters that produced this conclusion.

## Single-device baselines

Same harness, one device, no handoff. These are the honest comparison point for
the split, and they show the split is a throughput regression.

**Laptop only** (RTX 4050 Laptop, 6 GB, CUDA):

| Quant | Qwen3-0.6B TTFT | Qwen3-0.6B gen | Llama-3.2-1B TTFT | Llama-3.2-1B gen |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 2.661759 s | 177.6 tok/s | 2.806014 s | 171.7 tok/s |
| Q4_K_M | 1.503534 s | 241.0 tok/s | 2.343214 s | 176.7 tok/s |
| Q8_0 | 1.788383 s | 193.3 tok/s | 2.269317 s | 122.5 tok/s |

The phone decodes the same models at 2.2-9.3 tok/s, roughly 20-100x slower.
That gap is the cost of the split and the reason this is an offload experiment
rather than an acceleration one.

## NEON kernel comparison

Separate from the V1/V2 architecture comparison and specific to this device.
Standalone `llama-cli`, all servers stopped, Qwen3-0.6B Q4_K_M, 64 generated
tokens, seed 42, temperature 0, two threads, `--no-mmap`. One warm-up pair
excluded, five retained pairs.

| Build | TTFT | Decode |
| --- | ---: | ---: |
| Stock | 11.20 ± 0.36 s | 6.28 ± 0.22 tok/s |
| NEON accumulator | 9.96 ± 0.07 s | 6.58 ± 0.23 tok/s |

Mean TTFT 11.1% lower, mean decode 4.8% higher, and TTFT sample standard
deviation about 5.1x lower.

Profiling, the four-step verification that the two builds actually differed at
runtime, the deterministic-output correctness gate, and the CPU backend
eligibility finding are in
[`results/neon_q4_k_m_comparison.md`](../results/neon_q4_k_m_comparison.md) and
[`DEVICE-NOTES.md`](DEVICE-NOTES.md).

## Prompt formatting change

The retained timing artifacts above were collected before the router rendered
chat templates, using the raw completion prompt named in each method.

The router now takes plain user text, wraps it as a structured user message,
calls the laptop model's own `/apply-template` endpoint to render it once (with
Qwen3 thinking disabled), and sends that byte-identical rendered string to both
prefill and phone decode. It also reads the model's EOS token from `/props` and
passes it as the phone's stop sequence. Rendering once on the laptop rather than
duplicating template syntax in Python is what guarantees both servers restore
the same sequence.

This changes prompt token count and workload for newly recorded runs. The
earlier artifacts are preserved rather than overwritten, and they are handoff and
timing evidence, not a claim about output quality.

## Raw artifacts

Machine-readable, one file per configuration, each containing method parameters,
excluded warm-up rows, retained rows, and summary statistics:

```text
results/
  qwen_q4_k_m_v1_v2_laptop_gpu_power_repetitions.json
  qwen_q4_0_v1_v2_laptop_gpu_power_repetitions.json
  qwen_q8_0_v1_v2_laptop_gpu_power_repetitions.json
  llama32_q4_k_m_v1_v2_laptop_gpu_power_repetitions.json
  llama32_q4_0_v1_v2_laptop_gpu_power_repetitions.json
  llama32_q8_0_v1_v2_laptop_gpu_power_repetitions.json
  llama32_q4_k_m_v1_v2_laptop_gpu_power_repetitions_rerun_20260720.json
  q4_k_m_repetitions.json, q4_0_repetitions.json, q8_0_repetitions.json
  llama32_q4_k_m_v1_v2_repetitions.json, llama32_q4_0_..., llama32_q8_0_...
  edgesplit_wsl_benchmarks.csv
```

The `*_laptop_gpu_power_*` files are the GPU-instrumented passes reported above.
The plain `*_repetitions.json` files are the earlier timing-only passes, kept as
independent evidence. The CSV is exported from the SQLite database, which is a
local artifact and not tracked.

To regenerate a comparison against live services:

```bash
python bench/run_power_split_repetitions.py \
  --db bench/edgesplit.sqlite3 \
  --quant Q4_K_M \
  --output results/my_run.json \
  --runs 5 --warmups 1
```
