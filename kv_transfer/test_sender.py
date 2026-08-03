"""Laptop-only validation for the patched-server V2 envelope parser."""

from __future__ import annotations

import struct
import unittest

from sender import INNER_MAGIC, INNER_VERSION, INNER_HEADER, SenderError, parse_state_envelope


class SenderTests(unittest.TestCase):
    def test_parses_patched_server_envelope(self) -> None:
        tokens = (151, 152, 153)
        state = b"KV-state-bytes"
        envelope = (
            INNER_HEADER.pack(INNER_MAGIC, INNER_VERSION, 0, len(tokens), len(state))
            + struct.pack("<3i", *tokens)
            + state
        )
        parsed = parse_state_envelope(envelope)
        self.assertEqual(parsed.source_sequence_id, 0)
        self.assertEqual(parsed.sequence_length, len(tokens))
        self.assertEqual(parsed.state_bytes, len(state))
        self.assertEqual(parsed.raw, envelope)

    def test_rejects_length_mismatch(self) -> None:
        invalid = INNER_HEADER.pack(INNER_MAGIC, INNER_VERSION, 0, 1, 10) + b"short"
        with self.assertRaisesRegex(SenderError, "lengths"):
            parse_state_envelope(invalid)


if __name__ == "__main__":
    unittest.main()
