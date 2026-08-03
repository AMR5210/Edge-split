from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1 import V1Orchestrator, V1Request, V1Settings


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | Path]] = []

    def get_json(self, url: str) -> dict[str, Any]:
        self.calls.append(("get", url, {}))
        if url.endswith("/props"):
            return {"eos_token": "<|im_end|>"}
        raise AssertionError(url)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("json", url, payload))
        if url.endswith("/apply-template"):
            return {"prompt": "<templated-user-prompt>"}
        if url.endswith("/completion"):
            return {"timings": {"prompt_n": 9}}
        if "action=save" in url:
            return {"n_written": 4}
        if "action=restore" in url:
            return {"n_restored": 9}
        raise AssertionError(url)

    def post_file(self, url: str, path: Path) -> dict[str, Any]:
        self.calls.append(("file", url, path))
        return {"filename": path.name, "stored_bytes": path.stat().st_size}

    def stream_json(self, url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.calls.append(("stream", url, payload))
        yield {"content": "hello "}
        yield {"content": "world"}
        yield {
            "stop": True,
            "timings": {"predicted_n": 2, "predicted_per_second": 12.5},
        }


class V1OrchestratorTest(unittest.TestCase):
    def test_saves_uploads_restores_streams_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            state_dir.mkdir()
            # A real server creates this after the save request. This test
            # supplies the expected artifact so it can validate orchestration.
            (state_dir / "edgesplit-v1-test.bin").write_bytes(b"state")
            fake = FakeTransport()
            settings = V1Settings(
                laptop_url="http://laptop:8080",
                phone_url="http://phone:8081",
                phone_upload_url="http://phone:8090",
                laptop_state_dir=state_dir,
                benchmark_db=root / "bench.sqlite3",
                model_name="Qwen3-0.6B",
                quant="Q4_K_M",
                runtime_label="test",
                keep_state_files=True,
            )
            orchestrator = V1Orchestrator(settings, fake)

            # Stabilize the generated name for the test's pre-created artifact.
            import v1
            original_uuid4 = v1.uuid.uuid4
            v1.uuid.uuid4 = lambda: type("U", (), {"hex": "test"})()
            try:
                result = orchestrator.run(V1Request("Explain blue sky.", 2))
            finally:
                v1.uuid.uuid4 = original_uuid4

            self.assertEqual("hello world", result.content)
            self.assertEqual(5, result.state_bytes)
            self.assertEqual(12.5, result.decode_tokens_per_second)
            self.assertGreater(result.benchmark_row_id, 0)
            self.assertTrue(any(kind == "file" for kind, _, _ in fake.calls))
            stream_payload = next(payload for kind, _, payload in fake.calls if kind == "stream")
            self.assertEqual(True, stream_payload["cache_prompt"])
            self.assertEqual(True, stream_payload["stream"])
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
