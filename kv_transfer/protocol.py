"""Portable binary framing for EdgeSplit V2 sequence transfer.

The inner llama.cpp endpoint envelope is intentionally opaque here: it is
created and consumed by the matching patched llama-server binaries.  This
module provides the LAN-facing protocol around that envelope.  Its descriptor
contains the tensor shape, dtype, and decoded sequence length explicitly,
while SHA-256 protects the complete metadata, descriptor, and payload.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
from dataclasses import dataclass
from typing import Final


FRAME_MAGIC: Final = b"ESKV"
ACK_MAGIC: Final = b"ESAK"
VERSION: Final = 1
MAX_TENSORS: Final = 8
MAX_METADATA_BYTES: Final = 64 * 1024
MAX_PAYLOAD_BYTES: Final = 512 * 1024 * 1024

# Network byte order makes this outer protocol portable between the x86_64
# laptop and aarch64 phone.  The inner state envelope remains llama.cpp-owned.
FRAME_HEADER: Final = struct.Struct(">4sHHiIHHQ32s")
TENSOR_HEADER: Final = struct.Struct(">4sIHHQQQQQ")
ACK_HEADER: Final = struct.Struct(">4sHHI")

DTYPE_LLAMA_STATE_SEQUENCE: Final = 0x45534F50  # ASCII "ESOP"
TENSOR_NAME_SEQUENCE: Final = b"SEQ0"


class ProtocolError(ValueError):
    """The peer sent an invalid, unsupported, or corrupted V2 frame."""


@dataclass(frozen=True)
class TensorDescriptor:
    name: bytes
    dtype: int
    shape: tuple[int, ...]
    byte_length: int

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 4:
            raise ProtocolError("tensor name must contain one to four bytes")
        if not 1 <= len(self.shape) <= 4:
            raise ProtocolError("tensor rank must be between one and four")
        if any(dimension <= 0 for dimension in self.shape):
            raise ProtocolError("tensor dimensions must be positive")
        if self.byte_length <= 0:
            raise ProtocolError("tensor byte_length must be positive")

    def pack(self) -> bytes:
        padded_name = self.name.ljust(4, b"\0")
        padded_shape = (*self.shape, *(0 for _ in range(4 - len(self.shape))))
        return TENSOR_HEADER.pack(
            padded_name,
            self.dtype,
            len(self.shape),
            0,
            self.byte_length,
            *padded_shape,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TensorDescriptor":
        if len(data) != TENSOR_HEADER.size:
            raise ProtocolError("invalid tensor descriptor length")
        name, dtype, rank, reserved, byte_length, *shape = TENSOR_HEADER.unpack(data)
        if reserved != 0:
            raise ProtocolError("unsupported non-zero tensor descriptor flags")
        if not 1 <= rank <= 4:
            raise ProtocolError(f"unsupported tensor rank: {rank}")
        return cls(
            name=name.rstrip(b"\0"),
            dtype=dtype,
            shape=tuple(shape[:rank]),
            byte_length=byte_length,
        )


@dataclass(frozen=True)
class TransferFrame:
    source_sequence_id: int
    sequence_length: int
    metadata: dict[str, object]
    tensors: tuple[TensorDescriptor, ...]
    payload: bytes

    def __post_init__(self) -> None:
        if self.sequence_length < 0:
            raise ProtocolError("sequence_length cannot be negative")
        if not 1 <= len(self.tensors) <= MAX_TENSORS:
            raise ProtocolError("frame must contain between one and eight tensors")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ProtocolError("payload exceeds protocol limit")
        if sum(tensor.byte_length for tensor in self.tensors) != len(self.payload):
            raise ProtocolError("tensor byte lengths do not match payload")

    def pack(self) -> bytes:
        metadata = json.dumps(
            self.metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise ProtocolError("metadata exceeds protocol limit")
        descriptors = b"".join(tensor.pack() for tensor in self.tensors)
        digest = hashlib.sha256(metadata + descriptors + self.payload).digest()
        header = FRAME_HEADER.pack(
            FRAME_MAGIC,
            VERSION,
            FRAME_HEADER.size,
            self.source_sequence_id,
            self.sequence_length,
            len(self.tensors),
            len(metadata),
            len(self.payload),
            digest,
        )
        return header + metadata + descriptors + self.payload

    @classmethod
    def unpack_from_socket(cls, sock: socket.socket) -> "TransferFrame":
        header_data = recv_exact(sock, FRAME_HEADER.size)
        (
            magic,
            version,
            header_size,
            source_sequence_id,
            sequence_length,
            tensor_count,
            metadata_size,
            payload_size,
            expected_digest,
        ) = FRAME_HEADER.unpack(header_data)
        if magic != FRAME_MAGIC:
            raise ProtocolError("unexpected frame magic")
        if version != VERSION or header_size != FRAME_HEADER.size:
            raise ProtocolError("unsupported protocol version")
        if not 1 <= tensor_count <= MAX_TENSORS:
            raise ProtocolError("invalid tensor count")
        if metadata_size > MAX_METADATA_BYTES or payload_size > MAX_PAYLOAD_BYTES:
            raise ProtocolError("frame exceeds protocol limits")
        metadata_data = recv_exact(sock, metadata_size)
        descriptors_data = recv_exact(sock, tensor_count * TENSOR_HEADER.size)
        payload = recv_exact(sock, payload_size)
        digest = hashlib.sha256(metadata_data + descriptors_data + payload).digest()
        if digest != expected_digest:
            raise ProtocolError("frame SHA-256 mismatch")
        try:
            metadata = json.loads(metadata_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("invalid JSON metadata") from exc
        if not isinstance(metadata, dict):
            raise ProtocolError("metadata must be an object")
        tensors = tuple(
            TensorDescriptor.unpack(
                descriptors_data[index : index + TENSOR_HEADER.size]
            )
            for index in range(0, len(descriptors_data), TENSOR_HEADER.size)
        )
        return cls(
            source_sequence_id=source_sequence_id,
            sequence_length=sequence_length,
            metadata=metadata,
            tensors=tensors,
            payload=payload,
        )


def sequence_frame(
    *, source_sequence_id: int, sequence_length: int, state_envelope: bytes,
    model_id: str, llama_commit: str,
) -> TransferFrame:
    """Build the one-tensor V2 frame around a patched-server state envelope."""
    if not state_envelope:
        raise ProtocolError("cannot transfer an empty sequence envelope")
    return TransferFrame(
        source_sequence_id=source_sequence_id,
        sequence_length=sequence_length,
        metadata={
            "model_id": model_id,
            "llama_cpp_commit": llama_commit,
            "inner_format": "edgesplit-server-state-v1",
        },
        tensors=(TensorDescriptor(
            name=TENSOR_NAME_SEQUENCE,
            dtype=DTYPE_LLAMA_STATE_SEQUENCE,
            shape=(len(state_envelope),),
            byte_length=len(state_envelope),
        ),),
        payload=state_envelope,
    )


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly *size* bytes or fail rather than accepting truncation."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError(f"unexpected EOF with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_ack(sock: socket.socket, *, ok: bool, message: str = "") -> None:
    encoded = message.encode("utf-8")
    sock.sendall(ACK_HEADER.pack(ACK_MAGIC, VERSION, int(ok), len(encoded)) + encoded)


def receive_ack(sock: socket.socket) -> tuple[bool, str]:
    magic, version, status, message_size = ACK_HEADER.unpack(
        recv_exact(sock, ACK_HEADER.size)
    )
    if magic != ACK_MAGIC or version != VERSION or status not in (0, 1):
        raise ProtocolError("invalid acknowledgement")
    message = recv_exact(sock, message_size).decode("utf-8", errors="replace")
    return bool(status), message
