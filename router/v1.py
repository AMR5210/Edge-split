"""V1 file-based prefill-to-decode orchestration.

The laptop server owns the prefill slot, writes it to its slot-save directory,
and the router uploads those bytes to a small phone-side receiver. The phone
server restores the same filename from its own slot-save directory before
continuing generation from the restored slot.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from edgesplit_bench import log_run


class V1Error(RuntimeError):
    """A remote server or state-transfer operation failed."""


class Transport(Protocol):
    def get_json(self, url: str) -> dict[str, Any]: ...

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def post_file(self, url: str, path: Path) -> dict[str, Any]: ...

    def stream_json(self, url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]: ...


class UrllibTransport:
    """Small HTTP client implemented with the Python standard library."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _decode_response(response: Any, url: str) -> dict[str, Any]:
        raw = response.read()
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V1Error(f"invalid JSON response from {url}: {raw[:200]!r}") from exc
        if not isinstance(decoded, dict):
            raise V1Error(f"unexpected JSON response from {url}: {decoded!r}")
        return decoded

    def _open(self, request: Request, url: str) -> Any:
        try:
            return urlopen(request, timeout=self.timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise V1Error(f"{url} returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise V1Error(f"cannot reach {url}: {exc.reason}") from exc

    def get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, method="GET", headers={"Accept": "application/json"})
        with self._open(request, url) as response:
            return self._decode_response(response, url)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with self._open(request, url) as response:
            return self._decode_response(response, url)

    def post_file(self, url: str, path: Path) -> dict[str, Any]:
        request = Request(
            url, data=path.read_bytes(), method="POST",
            headers={"Content-Type": "application/octet-stream", "Accept": "application/json"},
        )
        with self._open(request, url) as response:
            return self._decode_response(response, url)

    def stream_json(self, url: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with self._open(request, url) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise V1Error(f"invalid SSE data from {url}: {data!r}") from exc
                if isinstance(event, dict):
                    yield event


@dataclass(frozen=True)
class V1Settings:
    laptop_url: str
    phone_url: str
    phone_upload_url: str
    laptop_state_dir: Path
    benchmark_db: Path
    model_name: str
    quant: str
    runtime_label: str
    timeout_seconds: float = 120.0
    keep_state_files: bool = False


@dataclass(frozen=True)
class V1Request:
    prompt: str
    n_predict: int
    slot_id: int = 0
    temperature: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class V1Result:
    content: str
    state_filename: str
    state_bytes: int
    ttft_seconds: float
    decode_tokens_per_second: float
    output_tokens: int
    benchmark_row_id: int


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def prepare_chat_generation(
    transport: Transport, laptop_url: str, user_prompt: str,
) -> tuple[str, list[str]]:
    """Render the model's own chat template once for both split endpoints.

    The two servers restore one shared sequence, so prefill and continuation
    must receive byte-identical text. Rendering through the laptop server
    keeps the template model-defined instead of duplicating Qwen syntax here.
    """
    template = transport.post_json(
        _url(laptop_url, "apply-template"),
        {
            "messages": [{"role": "user", "content": user_prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    rendered_prompt = template.get("prompt")
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise V1Error(f"laptop apply-template returned no prompt: {template!r}")

    props = transport.get_json(_url(laptop_url, "props"))
    eos_token = props.get("eos_token")
    if not isinstance(eos_token, str) or not eos_token:
        raise V1Error(f"laptop props returned no EOS token: {props!r}")
    return rendered_prompt, [eos_token]


class V1Orchestrator:
    """Single-request V1 controller; slots are intentionally not concurrent."""

    def __init__(self, settings: V1Settings, transport: Transport | None = None) -> None:
        self.settings = settings
        self.transport = transport or UrllibTransport(settings.timeout_seconds)
        self._lock = threading.Lock()

    def run(self, request: V1Request) -> V1Result:
        if not request.prompt.strip():
            raise V1Error("prompt must not be empty")
        if request.n_predict <= 0:
            raise V1Error("n_predict must be greater than zero")

        with self._lock:
            return self._run_locked(request)

    def _run_locked(self, request: V1Request) -> V1Result:
        started = time.monotonic()
        filename = f"edgesplit-v1-{uuid.uuid4().hex}.bin"
        state_path = self.settings.laptop_state_dir / filename
        self.settings.laptop_state_dir.mkdir(parents=True, exist_ok=True)
        rendered_prompt, stop = prepare_chat_generation(
            self.transport, self.settings.laptop_url, request.prompt
        )

        prefill = self.transport.post_json(
            _url(self.settings.laptop_url, "completion"),
            {
                "prompt": rendered_prompt,
                "n_predict": 0,
                "id_slot": request.slot_id,
                "cache_prompt": False,
                "temperature": request.temperature,
                "seed": request.seed,
            },
        )
        self.transport.post_json(
            _url(self.settings.laptop_url, f"slots/{request.slot_id}?action=save"),
            {"filename": filename},
        )
        if not state_path.is_file() or state_path.stat().st_size == 0:
            raise V1Error(f"laptop state file was not created: {state_path}")
        state_bytes = state_path.stat().st_size

        self.transport.post_file(
            _url(self.settings.phone_upload_url, f"state/{filename}"), state_path
        )
        self.transport.post_json(
            _url(self.settings.phone_url, f"slots/{request.slot_id}?action=restore"),
            {"filename": filename},
        )

        content: list[str] = []
        first_token_at: float | None = None
        final_event: dict[str, Any] | None = None
        for event in self.transport.stream_json(
            _url(self.settings.phone_url, "completion"),
            {
                "prompt": rendered_prompt,
                "n_predict": request.n_predict,
                "id_slot": request.slot_id,
                "cache_prompt": True,
                "temperature": request.temperature,
                "seed": request.seed,
                "stop": stop,
                "stream": True,
            },
        ):
            token = event.get("content")
            if isinstance(token, str) and token:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                content.append(token)
            if isinstance(event.get("timings"), dict):
                final_event = event

        if first_token_at is None:
            raise V1Error("phone completion returned no generated token")
        if final_event is None:
            raise V1Error("phone completion did not return final timing data")
        timings = final_event["timings"]
        decode_tps = float(timings["predicted_per_second"])
        output_tokens = int(timings["predicted_n"])
        if decode_tps <= 0 or output_tokens <= 0:
            raise V1Error(f"invalid phone timing data: {timings!r}")

        prompt_tokens = int(
            prefill.get("timings", {}).get("prompt_n", len(request.prompt.split()))
        )
        ttft_seconds = first_token_at - started
        row_id = log_run(self.settings.benchmark_db, {
            "device": "wsl-rtx4050+android-phone",
            "config": "split-v1",
            "model": self.settings.model_name,
            "quant": self.settings.quant,
            "prompt_len": str(prompt_tokens),
            "output_len": str(output_tokens),
            "ttft": f"{ttft_seconds:.6f}",
            "tokens_per_sec": f"{decode_tps:.6f}",
            "power_draw_mw": "",
            "llama_cpp_commit": self.settings.runtime_label,
            "notes": (
                "v1=slot-save-http-upload-slot-restore; "
                f"state_bytes={state_bytes}; slot={request.slot_id}"
            ),
        })

        if not self.settings.keep_state_files:
            state_path.unlink(missing_ok=True)
        return V1Result(
            content="".join(content),
            state_filename=filename,
            state_bytes=state_bytes,
            ttft_seconds=ttft_seconds,
            decode_tokens_per_second=decode_tps,
            output_tokens=output_tokens,
            benchmark_row_id=row_id,
        )
