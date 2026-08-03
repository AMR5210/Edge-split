# V1 versus V2: controlled phone-2 comparison

Date: 2026-07-17

## Primary five-pair statistics

For each quant, one V1/V2 warm-up pair was run and excluded, followed by five
retained pairs per configuration. Every retained pair used Qwen3-0.6B, the
same 11-token prompt, fixed seed 42, slot 0, and 16 generated tokens. Values
below are **mean ± sample standard deviation (median)**. The JSON exports hold
the exact retained and excluded SQLite row IDs:
[`q4_k_m_repetitions.json`](q4_k_m_repetitions.json),
[`q4_0_repetitions.json`](q4_0_repetitions.json), and
[`q8_0_repetitions.json`](q8_0_repetitions.json).

| Quant | V1 TTFT | V2 TTFT | V1 decode | V2 decode |
| --- | ---: | ---: | ---: | ---: |
| Q4_K_M | 2.838 ± 1.174 s (3.252) | 0.677 ± 0.064 s (0.675) | 8.695 ± 0.035 tok/s (8.681) | 8.756 ± 0.060 tok/s (8.748) |
| Q4_0 | 2.433 ± 1.527 s (1.817) | 0.614 ± 0.048 s (0.605) | 9.028 ± 0.237 tok/s (9.037) | 8.940 ± 0.504 tok/s (9.202) |
| Q8_0 | 2.461 ± 1.331 s (2.769) | 0.686 ± 0.071 s (0.681) | 8.171 ± 0.145 tok/s (8.138) | 7.737 ± 0.737 tok/s (7.803) |

V2 reduced mean TTFT by 76.16% (Q4_K_M), 74.78% (Q4_0), and 72.11% (Q8_0).
The V1 state-file path has markedly higher TTFT variance. Decode remains a
phone-side property rather than a V2 optimization target: Q4_K_M is +0.71%,
Q4_0 is -0.98%, and Q8_0 is -5.32% in V2 mean decode throughput, with the
largest spread on Q8_0 V2. No decode-speed improvement is claimed.

## Later five-pair run with laptop GPU power

This additive later pass repeats the same Qwen3-0.6B method (11 prompt tokens,
16 generated tokens, seed 42, slot 0): one excluded V1/V2 warm-up pair and
five retained pairs per quant. It records laptop GPU board power by polling
`nvidia-smi --query-gpu=power.draw` every 100 ms over the complete router
request. The timing-only results above remain preserved as their own raw
artifact family. Values are **mean +/- sample standard deviation (median)**.

| Quant | V1 rows | V2 rows | V1 TTFT | V2 TTFT | V1 decode | V2 decode | Laptop GPU draw, V1 / V2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 139, 141, 143, 145, 147 | 140, 142, 144, 146, 148 | 2.425 +/- 1.262 s (2.022) | 0.841 +/- 0.272 s (0.793) | 8.442 +/- 0.409 (8.663) | 8.078 +/- 0.356 (8.024) | 2.17 +/- 0.16 / 2.38 +/- 0.08 W |
| Q4_0 | 151, 153, 155, 157, 159 | 152, 154, 156, 158, 160 | 3.569 +/- 0.326 s (3.406) | 0.632 +/- 0.043 s (0.650) | 9.280 +/- 0.081 (9.262) | 8.979 +/- 0.227 (9.018) | 2.08 +/- 0.08 / 2.49 +/- 0.14 W |
| Q8_0 | 163, 165, 167, 169, 171 | 164, 166, 168, 170, 172 | 3.404 +/- 0.490 s (3.681) | 0.610 +/- 0.084 s (0.594) | 8.128 +/- 0.346 (8.241) | 7.261 +/- 0.523 (6.947) | 2.07 +/- 0.11 / 2.40 +/- 0.25 W |

Excluded warm-up rows are 137--138 (Q4_K_M), 149--150 (Q4_0), and 161--162
(Q8_0). The GPU figures are board-power samples rather than energy or total
system power; the whole-request window includes phone decode time when the
laptop GPU is generally idle.

Phone 2 battery power is not reported. The initial vendor fuel-gauge source
was invalid: filtering rejected 162/176 samples in one plugged window and
179/180 in one unplugged window. Rows 89--100 are retained as invalid
calibration evidence. A replacement `termux-battery-status` sampler used its
`current` (uA) and `voltage` (mV) fields, not `current_average`, and returned
plausible zero-drop samples at a 750 ms cadence. It nevertheless showed large
uncontrolled whole-device variation across consecutive screen-on and
screen-off idle windows (for example, screen-off medians of 614.754 versus
411.264 mW with ranges up to 2.443 W). It is not a stable baseline for
attributing short V1/V2 request power, so no phone-power conclusion is drawn.
