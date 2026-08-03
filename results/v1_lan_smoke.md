# Task 3 V1 LAN smoke result

Date: 2026-07-16

A complete V1 request succeeded over the LAN:

1. CUDA llama-server on the WSL laptop prefills the prompt into slot 0.
2. The router saves the slot state (1,262,444 bytes).
3. The router uploads it over HTTP to the phone state receiver.
4. The phone llama-server restores slot 0 and streams the continuation.
5. The router logs the run as `split-v1`.

| Config | Model | Quant | Prompt tokens | Output tokens | End-to-end TTFT | Phone decode |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| split-v1 | Qwen3-0.6B | Q4_K_M | 11 | 16 | 0.647003 s | 15.738845 tok/s |

The router inserted benchmark row 5 in the native WSL SQLite database. The
runtime label is deliberately mixed: laptop
`e920c523e3b8a0163fe498af5bf90df35ff51d25`; phone
`official-prebuilt-b10034`. See the README for the investigated Termux
from-source crash and this known parity limitation.
