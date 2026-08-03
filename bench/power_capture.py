#!/usr/bin/env python3
"""Device-power capture primitives for the paired split benchmark.

The laptop sampler polls ``nvidia-smi --query-gpu=power.draw`` while a router
request is in flight.  The phone sampler is a small Termux HTTP service that
samples battery current and voltage during the matching request window.

Power from the two devices is deliberately retained as two measurements.  It
is not summed: the phone value is whole-device battery power, whereas the
laptop value is GPU-board power reported by NVIDIA.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PowerCaptureError(RuntimeError):
    """A power capture source could not produce a complete window."""


@dataclass(frozen=True)
class PowerWindow:
    average_mw: float
    median_mw: float
    sample_stdev_mw: float
    min_mw: float
    max_mw: float
    sample_count: int
    duration_seconds: float
    metadata: dict[str, object]

    def db_values(self) -> dict[str, object]:
        return asdict(self)


def summarize_samples(
    samples_mw: list[float], duration_seconds: float, metadata: dict[str, object],
) -> PowerWindow:
    if not samples_mw:
        raise PowerCaptureError("power capture produced zero samples")
    if duration_seconds < 0:
        raise PowerCaptureError("power capture duration cannot be negative")
    if any(sample < 0 for sample in samples_mw):
        raise PowerCaptureError("power capture produced a negative magnitude")
    return PowerWindow(
        average_mw=statistics.fmean(samples_mw),
        median_mw=statistics.median(samples_mw),
        sample_stdev_mw=statistics.stdev(samples_mw) if len(samples_mw) > 1 else 0.0,
        min_mw=min(samples_mw),
        max_mw=max(samples_mw),
        sample_count=len(samples_mw),
        duration_seconds=duration_seconds,
        metadata=metadata,
    )


def read_nvidia_power_mw() -> float:
    """Poll NVIDIA's current GPU-board power draw in milliwatts."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10.0,
        )
        first_line = output.strip().splitlines()[0]
        return float(first_line) * 1000.0
    except (FileNotFoundError, IndexError, OSError, ValueError, subprocess.SubprocessError) as exc:
        raise PowerCaptureError(f"nvidia-smi power query failed: {exc}") from exc


class NvidiaPowerSampler:
    """Poll NVIDIA power in a background thread for one request window."""

    def __init__(
        self, interval_seconds: float = 0.1,
        reader: Callable[[], float] = read_nvidia_power_mw,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.reader = reader
        self._samples_mw: list[float] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._error: Exception | None = None

    def _read_once(self) -> None:
        try:
            value = float(self.reader())
            if value < 0:
                raise ValueError(f"negative GPU power value: {value}")
        except Exception as exc:  # reader implementations are intentionally pluggable
            self._error = exc
            self._stop_event.set()
            return
        with self._samples_lock:
            self._samples_mw.append(value)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._read_once()
            if self._error is not None:
                return

    def start(self) -> None:
        if self._thread is not None:
            raise PowerCaptureError("GPU sampler is already running")
        self._samples_mw = []
        self._error = None
        self._stop_event.clear()
        self._started_at = time.monotonic()
        # Capture an immediate sample so even short requests have a window.
        self._read_once()
        if self._error is not None:
            raise PowerCaptureError(f"GPU sampler could not start: {self._error}") from self._error
        self._thread = threading.Thread(target=self._run, name="edgesplit-gpu-power", daemon=True)
        self._thread.start()

    def stop(self) -> PowerWindow:
        if self._thread is None or self._started_at is None:
            raise PowerCaptureError("GPU sampler was not started")
        self._stop_event.set()
        self._thread.join(timeout=12.0)
        if self._thread.is_alive():
            raise PowerCaptureError("GPU sampler did not stop within 12 seconds")
        duration = time.monotonic() - self._started_at
        if self._error is not None:
            raise PowerCaptureError(f"GPU sampler failed: {self._error}") from self._error
        with self._samples_lock:
            samples = list(self._samples_mw)
        return summarize_samples(
            samples,
            duration,
            {"source": "nvidia-smi", "interval_seconds": self.interval_seconds},
        )


class PhonePowerClient:
    """Client for the manually started Termux phone battery sampler."""

    def __init__(self, base_url: str, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, body: dict[str, object]) -> dict[str, Any]:
        request = Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise PowerCaptureError(
                f"phone power sampler {path} returned HTTP {exc.code}: {exc.read()[:300]!r}"
            ) from exc
        except URLError as exc:
            raise PowerCaptureError(f"cannot reach phone power sampler: {exc.reason}") from exc
        if not isinstance(decoded, dict):
            raise PowerCaptureError(f"unexpected phone power response: {decoded!r}")
        return decoded

    def health(self) -> dict[str, Any]:
        request = Request(self.base_url + "/healthz", headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, ValueError) as exc:
            raise PowerCaptureError(f"phone power health check failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise PowerCaptureError(f"unexpected phone power health: {decoded!r}")
        return decoded

    def start_window(self) -> str:
        response = self._post("/v1/window/start", {})
        window_id = response.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise PowerCaptureError(f"phone power start returned no window id: {response!r}")
        return window_id

    def stop_window(self, window_id: str) -> PowerWindow:
        response = self._post("/v1/window/stop", {"window_id": window_id})
        try:
            return PowerWindow(
                average_mw=float(response["average_mw"]),
                median_mw=float(response["median_mw"]),
                sample_stdev_mw=float(response["sample_stdev_mw"]),
                min_mw=float(response["min_mw"]),
                max_mw=float(response["max_mw"]),
                sample_count=int(response["sample_count"]),
                duration_seconds=float(response["duration_seconds"]),
                metadata=dict(response.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PowerCaptureError(f"invalid phone power window: {response!r}") from exc
