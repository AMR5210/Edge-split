"""Phone-side EdgeSplit V2 listener: raw TCP frame to patched llama-server."""

from __future__ import annotations

import argparse
import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from protocol import (
    DTYPE_LLAMA_STATE_SEQUENCE,
    ProtocolError,
    TENSOR_NAME_SEQUENCE,
    TransferFrame,
    send_ack,
)


def import_state(phone_server: str, slot_id: int, payload: bytes, timeout: float) -> str:
    url = f"{phone_server.rstrip('/')}/edgesplit/v2/state/{slot_id}"
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProtocolError(f"patched phone server returned HTTP {response.status}")
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise ProtocolError(f"phone import returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except URLError as exc:
        raise ProtocolError(f"cannot reach patched phone server: {exc.reason}") from exc


def validate_frame(frame: TransferFrame, model_id: str, llama_commit: str) -> None:
    if frame.metadata.get("model_id") != model_id:
        raise ProtocolError("model_id does not match this receiver")
    if frame.metadata.get("llama_cpp_commit") != llama_commit:
        raise ProtocolError("llama.cpp commit does not match this receiver")
    if len(frame.tensors) != 1:
        raise ProtocolError("V2 receiver expects exactly one sequence tensor")
    tensor = frame.tensors[0]
    if tensor.name != TENSOR_NAME_SEQUENCE or tensor.dtype != DTYPE_LLAMA_STATE_SEQUENCE:
        raise ProtocolError("unsupported V2 tensor descriptor")
    if tensor.shape != (len(frame.payload),) or tensor.byte_length != len(frame.payload):
        raise ProtocolError("sequence tensor shape does not match its payload")


def serve(
    *, listen_host: str, port: int, phone_server: str, slot_id: int,
    model_id: str, llama_commit: str, timeout: float,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_host, port))
        listener.listen(1)
        print(json.dumps({"status": "listening", "host": listen_host, "port": port}), flush=True)
        while True:
            connection, address = listener.accept()
            with connection:
                connection.settimeout(timeout)
                try:
                    frame = TransferFrame.unpack_from_socket(connection)
                    validate_frame(frame, model_id, llama_commit)
                    result = import_state(phone_server, slot_id, frame.payload, timeout)
                    send_ack(connection, ok=True, message=result)
                    print(json.dumps({"status": "imported", "peer": address[0], "bytes": len(frame.payload)}), flush=True)
                except (OSError, ProtocolError) as exc:
                    try:
                        send_ack(connection, ok=False, message=str(exc))
                    except OSError:
                        pass
                    print(json.dumps({"status": "rejected", "peer": address[0], "error": str(exc)}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--phone-server", default="http://127.0.0.1:8081")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    serve(
        listen_host=args.listen_host, port=args.port, phone_server=args.phone_server,
        slot_id=args.slot, model_id=args.model_id, llama_commit=args.llama_commit,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
