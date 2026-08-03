"""Laptop-only tests for the V2 binary framing."""

from __future__ import annotations

import socket
import threading
import unittest

from protocol import (
    DTYPE_LLAMA_STATE_SEQUENCE,
    ProtocolError,
    TENSOR_NAME_SEQUENCE,
    TransferFrame,
    receive_ack,
    send_ack,
    sequence_frame,
)


class ProtocolTests(unittest.TestCase):
    def test_frame_round_trip_over_socketpair(self) -> None:
        payload = b"raw-llama-state" * 128
        frame = sequence_frame(
            source_sequence_id=3,
            sequence_length=11,
            state_envelope=payload,
            model_id="Qwen3-0.6B",
            llama_commit="e920c523e3b8a0163fe498af5bf90df35ff51d25",
        )
        client, server = socket.socketpair()
        try:
            result: list[TransferFrame] = []

            def receive() -> None:
                result.append(TransferFrame.unpack_from_socket(server))
                send_ack(server, ok=True, message="imported")

            thread = threading.Thread(target=receive)
            thread.start()
            client.sendall(frame.pack())
            self.assertEqual(receive_ack(client), (True, "imported"))
            thread.join()
            received = result[0]
            self.assertEqual(received.payload, payload)
            self.assertEqual(received.sequence_length, 11)
            self.assertEqual(received.tensors[0].name, TENSOR_NAME_SEQUENCE)
            self.assertEqual(received.tensors[0].dtype, DTYPE_LLAMA_STATE_SEQUENCE)
            self.assertEqual(received.tensors[0].shape, (len(payload),))
        finally:
            client.close()
            server.close()

    def test_corruption_is_rejected(self) -> None:
        frame = sequence_frame(
            source_sequence_id=0, sequence_length=1, state_envelope=b"state",
            model_id="model", llama_commit="commit",
        )
        packed = bytearray(frame.pack())
        packed[-1] ^= 0xFF
        client, server = socket.socketpair()
        try:
            client.sendall(packed)
            with self.assertRaisesRegex(ProtocolError, "SHA-256"):
                TransferFrame.unpack_from_socket(server)
        finally:
            client.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
