# EdgeSplit V2 transfer framing

The laptop fetches a raw sequence envelope from the additive patched
`llama-server` endpoint, wraps it in this TCP frame, and streams it directly
to the phone listener. The listener validates the model/commit metadata and
descriptor, then POSTs the original sequence envelope to the patched phone
server for in-process `llama_state_seq_set_data_ext()` restoration.

The outer frame is network-byte-order and contains:

- `source_sequence_id` and `sequence_length`;
- JSON metadata including the model ID and llama.cpp commit;
- one or more tensor descriptors: 4-byte name, dtype, rank, byte length, and
  up to four `uint64` dimensions;
- contiguous tensor payload bytes; and
- a SHA-256 over metadata, descriptors, and payload.

Current V2 uses one explicitly typed `SEQ0` tensor (`ESOP`): it is the opaque
sequence-state buffer produced by llama.cpp's public extended state API. Its
one-dimensional shape is its exact byte length. The matching patched server
owns the private KV layout and restores it without touching the V1 slot file
path. This keeps the custom network serialization versioned and verifiable
while avoiding an unsafe reimplementation of llama.cpp internals in Python.
