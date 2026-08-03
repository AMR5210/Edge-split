from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1 import V1Request
from v2 import V2Orchestrator, V2Settings


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get_json(self, url: str) -> dict[str, Any]:
        self.calls.append(("get", url, {}))
        if url.endswith("/props"):
            return {"eos_token": "<|im_end|>"}
        raise AssertionError(url)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("json", url, payload))
        if url.endswith("/apply-template"):
            return {"prompt": "<templated-user-prompt>"}
        return {"timings": {"prompt_n": 9}}

    def stream_json(self, url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.calls.append(("stream", url, payload))
        yield {"content": "hello "}
        yield {"content": "phone"}
        yield {"timings": {"predicted_n": 2, "predicted_per_second": 12.5}}


class V2OrchestratorTest(unittest.TestCase):
    def test_streams_raw_state_and_logs_without_v1_file_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeTransport()
            sent: list[dict[str, object]] = []

            def sender(**kwargs: object) -> dict[str, object]:
                sent.append(kwargs)
                return {
                    "sequence_length": 9,
                    "inner_state_bytes": 1024,
                    "tcp_frame_bytes": 1100,
                    "transfer_seconds": 0.05,
                }

            orchestrator = V2Orchestrator(V2Settings(
                laptop_url="http://laptop:8080",
                phone_url="http://phone:8081",
                phone_host="phone",
                phone_port=8091,
                benchmark_db=root / "bench.sqlite3",
                model_name="Qwen3-0.6B",
                quant="Q4_K_M",
                llama_cpp_commit="e920",
                runtime_label="test",
            ), fake, sender)
            result = orchestrator.run(V1Request("Explain blue sky.", 2))

            self.assertEqual("hello phone", result.content)
            self.assertEqual(1024, result.inner_state_bytes)
            self.assertGreater(result.benchmark_row_id, 0)
            self.assertEqual(1, len(sent))
            self.assertEqual("phone", sent[0]["phone_host"])
            self.assertFalse(any(kind == "file" for kind, _, _ in fake.calls))
            stream_payload = next(payload for kind, _, payload in fake.calls if kind == "stream")
            self.assertTrue(stream_payload["cache_prompt"])
            self.assertTrue(stream_payload["stream"])
            self.assertEqual("<templated-user-prompt>", stream_payload["prompt"])
            self.assertEqual(["<|im_end|>"], stream_payload["stop"])
            prefill_payload = next(
                payload for kind, url, payload in fake.calls
                if kind == "json" and url.endswith("/completion")
            )
            self.assertEqual("<templated-user-prompt>", prefill_payload["prompt"])
            template_payload = next(
                payload for kind, url, payload in fake.calls
                if kind == "json" and url.endswith("/apply-template")
            )
            self.assertEqual(
                {"enable_thinking": False}, template_payload["chat_template_kwargs"]
            )


if __name__ == "__main__":
    unittest.main()
