"""V2 raw-TCP prefill-to-decode orchestration.

This leaves V1's save-file/upload/restore pipeline intact. V2 asks the
patched laptop server for its raw sequence buffer, streams the frame over TCP
to phone 2, and lets the patched phone server restore it in-process.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
KV_TRANSFER_DIR = ROOT / "kv_transfer"
for directory in (BENCH_DIR, KV_TRANSFER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from edgesplit_bench import log_run
from sender import SenderError, send_exported_state
from v1 import (
    Transport,
    UrllibTransport,
    V1Error,
    V1Request,
    _url,
    prepare_chat_generation,
)


class V2Error(V1Error):
    """A V2 transfer or phone continuation failed."""


@dataclass(frozen=True)
class V2Settings:
    laptop_url: str
    phone_url: str
    phone_host: str
    phone_port: int
    benchmark_db: Path
    model_name: str
    quant: str
    llama_cpp_commit: str
    runtime_label: str
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class V2Result:
    content: str
    sequence_length: int
    inner_state_bytes: int
    tcp_frame_bytes: int
    transfer_seconds: float
    ttft_seconds: float
    decode_tokens_per_second: float
    output_tokens: int
    benchmark_row_id: int


Sender = Callable[..., dict[str, object]]


class V2Orchestrator:
    """Single-request V2 controller; it never creates a V1 state file."""

    def __init__(
        self, settings: V2Settings, transport: Transport | None = None,
        sender: Sender = send_exported_state,
    ) -> None:
        self.settings = settings
        self.transport = transport or UrllibTransport(settings.timeout_seconds)
        self.sender = sender
        self._lock = threading.Lock()

    def run(self, request: V1Request) -> V2Result:
        if not request.prompt.strip():
            raise V2Error("prompt must not be empty")
        if request.n_predict <= 0:
            raise V2Error("n_predict must be greater than zero")
        with self._lock:
            return self._run_locked(request)

    def _run_locked(self, request: V1Request) -> V2Result:
        started = time.monotonic()
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
        try:
            transfer = self.sender(
                laptop_url=self.settings.laptop_url,
                slot_id=request.slot_id,
                phone_host=self.settings.phone_host,
                phone_port=self.settings.phone_port,
                model_id=self.settings.model_name,
                llama_commit=self.settings.llama_cpp_commit,
                timeout_seconds=self.settings.timeout_seconds,
            )
        except SenderError as exc:
            raise V2Error(str(exc)) from exc

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
            raise V2Error("phone completion returned no generated token")
        if final_event is None:
            raise V2Error("phone completion did not return final timing data")
        timings = final_event["timings"]
        decode_tps = float(timings["predicted_per_second"])
        output_tokens = int(timings["predicted_n"])
        if decode_tps <= 0 or output_tokens <= 0:
            raise V2Error(f"invalid phone timing data: {timings!r}")

        prompt_tokens = int(
            prefill.get("timings", {}).get("prompt_n", len(request.prompt.split()))
        )
        try:
            sequence_length = int(transfer["sequence_length"])
            inner_state_bytes = int(transfer["inner_state_bytes"])
            tcp_frame_bytes = int(transfer["tcp_frame_bytes"])
            transfer_seconds = float(transfer["transfer_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise V2Error(f"invalid sender result: {transfer!r}") from exc
        ttft_seconds = first_token_at - started
        row_id = log_run(self.settings.benchmark_db, {
            "device": "wsl-rtx4050+android-phone2",
            "config": "split-v2",
            "model": self.settings.model_name,
            "quant": self.settings.quant,
            "prompt_len": str(prompt_tokens),
            "output_len": str(output_tokens),
            "ttft": f"{ttft_seconds:.6f}",
            "tokens_per_sec": f"{decode_tps:.6f}",
            "power_draw_mw": "",
            "llama_cpp_commit": self.settings.runtime_label,
            "notes": (
                "v2=state-seq-ext-raw-tcp-in-process-import; "
                f"sequence_length={sequence_length}; inner_state_bytes={inner_state_bytes}; "
                f"tcp_frame_bytes={tcp_frame_bytes}; transfer_seconds={transfer_seconds:.6f}; "
                f"slot={request.slot_id}"
            ),
        })
        return V2Result(
            content="".join(content),
            sequence_length=sequence_length,
            inner_state_bytes=inner_state_bytes,
            tcp_frame_bytes=tcp_frame_bytes,
            transfer_seconds=transfer_seconds,
            ttft_seconds=ttft_seconds,
            decode_tokens_per_second=decode_tps,
            output_tokens=output_tokens,
            benchmark_row_id=row_id,
        )
