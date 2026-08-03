# Architecture

How a request crosses two machines, and why the pieces are shaped the way they
are.

## Request lifecycle

![EdgeSplit architecture](images/architecture.svg)

1. The caller POSTs a prompt to the FastAPI router (`/v1/generate` or
   `/v2/generate`).
2. The router renders the chat template **once**, on the laptop, through
   llama-server's `/apply-template`, and reads the model's EOS token from
   `/props`.
3. The laptop CUDA llama-server prefills the prompt into a slot with
   `n_predict = 0`.
4. The slot's sequence state crosses to the phone, by one of two paths below.
5. The phone llama-server continues generation from the restored sequence and
   streams tokens back through the router.
6. The router logs one benchmark row and returns the completion.

Both modes are single-request by design. `V1Orchestrator` and `V2Orchestrator`
each hold a `threading.Lock` for the whole request because a slot is not a
concurrent resource.

### Why the template is rendered on the laptop

Prefill and continuation must receive byte-identical text, or the phone restores
a sequence that does not match the tokens it is about to continue. Rendering
through the laptop model's own endpoint keeps the template model-defined instead
of reimplementing Qwen or Llama chat syntax in Python, where it would silently
drift from upstream. See `prepare_chat_generation` in
[`router/v1.py`](../router/v1.py).

## V1: file handoff

The path that proved the pipeline worked end to end, using only stock llama.cpp
endpoints.

```text
laptop: POST /slots/:id?action=save   ->  ~1.26 MB state file on disk
router: POST /state/:filename         ->  phone HTTP receiver writes it
phone:  POST /slots/:id?action=restore -> restore from that file
```

The phone receiver ([`decode/state_receiver.py`](../decode/state_receiver.py)) is
deliberately minimal: it rejects any path that is not a single filename under
`/state/`, enforces a content-length ceiling, and writes to a temporary name
before `os.replace()` so llama-server never sees a partially written state file.

V1 is retained and still works. V2 did not replace it.

## V2: raw sequence-state handoff

```text
laptop: GET  /edgesplit/v2/state/:id_slot  ->  raw sequence envelope
router: one TCP connection, one framed message
phone:  POST /edgesplit/v2/state/:id_slot  ->  in-process restore
```

No file is created on either side.

### The two patched endpoints

The interesting constraint: `llama_state_seq_get_data_ext()` and
`llama_state_seq_set_data_ext()` operate on a slot's sequence and are not safe to
call from an arbitrary thread while the server is running. Rather than reaching
into llama-server's internals from the HTTP handler,
[`patches/llama.cpp/0001-edgesplit-v2-raw-sequence-endpoints.patch`](../patches/llama.cpp/0001-edgesplit-v2-raw-sequence-endpoints.patch)
adds two task types that **enqueue on llama-server's existing task queue**, so
both calls execute on the server thread that already owns the slot.

That is the load-bearing design decision in this project. It means the patch
composes with llama-server's concurrency model instead of fighting it, and it is
why the patch is additive: it does not touch the stock `/slots` save/restore
routes, and both handoff modes remain available from one binary.

```text
GET  /edgesplit/v2/state/:id_slot   ->  ESV2 envelope (header + tokens + state)
POST /edgesplit/v2/state/:id_slot   <-  same envelope, restored in process
```

### The inner envelope is opaque on purpose

The C++ side emits a native little-endian envelope: magic `ESV2`, version, source
slot, token count, state byte count, then the token array, then the opaque
`llama_state_seq_*` bytes.

Python parses only the header, to validate lengths
([`kv_transfer/sender.py`](../kv_transfer/sender.py)). It never interprets the
state payload. The KV cache layout is private to llama.cpp and version-specific;
reimplementing it in Python would be a silent-corruption bug waiting for the next
upstream change. The matching patched server on the other end owns that layout,
and commit parity is enforced by the frame metadata.

### Outer transfer frame

The inner envelope is wrapped for the network by
[`kv_transfer/protocol.py`](../kv_transfer/protocol.py). Everything in the outer
frame is **network byte order**, because the sender is x86_64 and the receiver is
aarch64.

```text
struct FrameHeader {          // >4sHHiIHHQ32s
    char     magic[4];        // "ESKV"
    uint16   version;
    uint16   header_size;
    int32    source_sequence_id;
    uint32   sequence_length;
    uint16   tensor_count;
    uint16   metadata_size;
    uint64   payload_size;
    uint8    sha256[32];      // over metadata + descriptors + payload
}
```

followed by JSON metadata, one or more tensor descriptors, and the payload.

Each descriptor carries a 4-byte name, dtype, rank, byte length, and up to four
`uint64` dimensions. V2 currently sends exactly one: name `SEQ0`, dtype `ESOP`
(`0x45534F50`), rank 1, whose single dimension is its exact byte length.

Limits are explicit: 8 tensors, 64 KiB metadata, 512 MiB payload. `recv_exact()`
fails on short reads rather than accepting truncation. The receiver validates the
digest, then the model ID and llama.cpp commit in the metadata, then the
descriptor shape against the payload length, and only then forwards to the phone
server. A mismatch produces a negative ACK carrying the reason rather than a
dropped connection.

The frame is a versioned, checksummed, explicitly-typed container, not a raw
`send(buffer)`. That is what makes a cross-architecture binary handoff
debuggable when it goes wrong.

## Known cost in the V2 path

The Python receiver on the phone reads the entire frame into memory, then POSTs
the payload to llama-server over localhost HTTP. V2 removes V1's disk write, its
network file upload, and its disk read, but two copies remain. Eliminating them
would mean either moving the listener into the patched C++ server or handing over
a file descriptor. That work was not done.

## Benchmark harness

Every run in every mode writes one row to the same SQLite schema, from the first
single-device baseline onward, so all configurations stay directly comparable.

```sql
benchmark_runs        -- one row per completed run, with a valid/invalid status
benchmark_attempts    -- written BEFORE inference; a killed process stays 'started'
benchmark_power_windows -- per-source power, UNIQUE(run_id, source)
```

`benchmark_power_windows` keeps `laptop_gpu` and `phone_battery` as separate
rows and the legacy scalar `benchmark_runs.power_draw_mw` unset for split runs.
A split request has two power sources measuring different things: GPU board power
and whole-device battery power. Adding them would produce a number with no
physical meaning. See [`BENCHMARKS.md`](BENCHMARKS.md) for why the phone source
was ultimately excluded entirely.

## Deliberately not used: llama.cpp's RPC backend

llama.cpp already ships an RPC backend that distributes work across machines, and
it is the wrong tool here. It shards model layers to pool memory, which solves
"this model does not fit on one device." That is a different problem from phase
disaggregation, and it adds a network round trip per layer. That is pure overhead
once the model already fits on one device, which at 0.6B and 1B it does.

EdgeSplit moves the sequence state exactly once, at the prefill/decode boundary.
