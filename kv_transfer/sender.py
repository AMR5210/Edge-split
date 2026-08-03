"""Laptop-side sender for EdgeSplit V2 raw TCP sequence transfer."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from protocol import ProtocolError, receive_ack, sequence_frame


INNER_HEADER = struct.Struct("<IIiIQ")
INNER_MAGIC = 0x32565345  # ESV2
INNER_VERSION = 1


class SenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class StateEnvelope:
    source_sequence_id: int
    sequence_length: int
    state_bytes: int
    raw: bytes


def parse_state_envelope(data: bytes) -> StateEnvelope:
    if len(data) < INNER_HEADER.size:
        raise SenderError("patched llama-server returned a truncated V2 state envelope")
    magic, version, source_slot, token_count, state_bytes = INNER_HEADER.unpack_from(data)
    if magic != INNER_MAGIC or version != INNER_VERSION:
        raise SenderError("patched llama-server returned an unsupported V2 state envelope")
    expected = INNER_HEADER.size + token_count * 4 + state_bytes
    if expected != len(data):
        raise SenderError(
            f"invalid V2 state envelope lengths: expected {expected}, received {len(data)}"
        )
    return StateEnvelope(source_slot, token_count, state_bytes, data)


def export_state(laptop_url: str, slot_id: int, timeout_seconds: float) -> StateEnvelope:
    url = f"{laptop_url.rstrip('/')}/edgesplit/v2/state/{slot_id}"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout_seconds) as response:
            if response.status != 200:
                raise SenderError(f"state export returned HTTP {response.status}")
            return parse_state_envelope(response.read())
    except HTTPError as exc:
        raise SenderError(f"state export returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except URLError as exc:
        raise SenderError(f"cannot reach patched laptop server: {exc.reason}") from exc


def send_exported_state(
    *, laptop_url: str, slot_id: int, phone_host: str, phone_port: int,
    model_id: str, llama_commit: str, timeout_seconds: float,
) -> dict[str, object]:
    envelope = export_state(laptop_url, slot_id, timeout_seconds)
    frame = sequence_frame(
        source_sequence_id=envelope.source_sequence_id,
        sequence_length=envelope.sequence_length,
        state_envelope=envelope.raw,
        model_id=model_id,
        llama_commit=llama_commit,
    )
    packed = frame.pack()
    started = time.monotonic()
    try:
        with socket.create_connection((phone_host, phone_port), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            sock.sendall(packed)
            ok, message = receive_ack(sock)
    except OSError as exc:
        raise SenderError(f"raw TCP transfer to {phone_host}:{phone_port} failed: {exc}") from exc
    if not ok:
        raise SenderError(f"phone receiver rejected sequence state: {message}")
    return {
        "source_sequence_id": envelope.source_sequence_id,
        "sequence_length": envelope.sequence_length,
        "inner_state_bytes": envelope.state_bytes,
        "tcp_frame_bytes": len(packed),
        "transfer_seconds": time.monotonic() - started,
        "receiver": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laptop-url", default="http://127.0.0.1:8080")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--phone-host", required=True)
    parser.add_argument("--phone-port", type=int, default=8091)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        result = send_exported_state(
            laptop_url=args.laptop_url, slot_id=args.slot, phone_host=args.phone_host,
            phone_port=args.phone_port, model_id=args.model_id,
            llama_commit=args.llama_commit, timeout_seconds=args.timeout,
        )
    except (SenderError, ProtocolError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
