# EdgeSplit demo package

This package is a runbook for a live end-to-end run, not a synthetic demo. It
drives the same Phone-2 V1/V2 paths that produced the tracked benchmark rows.

## Prerequisites

On Phone 2, use the source-built, patched Q4_K_M server and leave three Termux
sessions running:

```bash
# Session 1: patched source llama-server on 8081
cd ~/edgesplit
export LLAMA_BIN_DIR="$HOME/src/llama.cpp/build-termux/bin"
export MODEL_PATH="$HOME/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
export SLOT_SAVE_PATH="$HOME/edgesplit/state"
export DECODE_THREADS=2
./decode/start_server.sh

# Session 2: preserved V1 receiver on 8090
cd ~/edgesplit
SLOT_SAVE_PATH="$HOME/edgesplit/state" ./decode/start_state_receiver.sh

# Session 3: V2 TCP listener on 8091
cd ~/edgesplit
EDGESPLIT_MODEL_ID=Qwen3-0.6B ./decode/start_v2_receiver.sh
```

Confirm both phone services are reachable:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8090/healthz
```

On WSL, start the laptop components. Set `EDGESPLIT_PHONE_HOST` to the phone's
own LAN IP address; the launcher has no usable default.

```bash
cd ~/edgesplit
./demo/start_laptop_demo.sh
```

It exposes the router on `http://127.0.0.1:8083`. The script prints copy-paste
requests for both modes and shuts down cleanly on Ctrl-C.

## Scope of what this demonstrates

Do not claim phone-battery power, system energy, a broader device survey, or a
production-ready distributed serving system. The reported GPU metric is laptop
board power only; the results cover one laptop/Phone-2 LAN pair.
