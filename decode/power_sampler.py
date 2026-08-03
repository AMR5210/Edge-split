#!/data/data/com.termux/files/usr/bin/python
"""Termux HTTP service for per-request Phone 2 battery-power windows.

This file is intentionally written for manual execution on Android; it is not
run or tested from the laptop.  It invokes ``termux-battery-status`` for each
sample rather than reading vendor fuel-gauge sysfs nodes.  Termux:API exposes
Android's battery current-now value in microamps and battery voltage in
millivolts; their product is reported as an absolute battery-power magnitude in
milliwatts, not CPU/package power.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class SamplerError(RuntimeError):
    """The Termux:API sampler cannot produce a trustworthy window."""


@dataclass(frozen=True)
class BatteryReading:
    current_ua: int
    voltage_mv: int
    power_mw: float
    status: str | None
    plugged: str | None


def _json_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplerError(f"termux-battery-status lacks numeric {name!r}: {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise SamplerError(f"termux-battery-status {name!r} is not integral: {value!r}")
    return int(value)


def parse_battery_status(output: str) -> BatteryReading:
    """Parse the documented Termux:API current (uA) and voltage (mV) fields."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise SamplerError(f"termux-battery-status returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SamplerError("termux-battery-status JSON must be an object")
    current_ua = _json_integer(payload, "current")
    voltage_mv = _json_integer(payload, "voltage")
    status = payload.get("status")
    plugged = payload.get("plugged")
    if status is not None and not isinstance(status, str):
        raise SamplerError("termux-battery-status status must be a string when present")
    if plugged is not None and not isinstance(plugged, str):
        raise SamplerError("termux-battery-status plugged must be a string when present")
    # uA * mV / 1,000,000 = mW.
    return BatteryReading(
        current_ua=current_ua,
        voltage_mv=voltage_mv,
        power_mw=abs(current_ua * voltage_mv) / 1_000_000.0,
        status=status,
        plugged=plugged,
    )


def read_battery_status(timeout_seconds: float) -> tuple[BatteryReading, float]:
    """Run the official Termux client once and retain its elapsed time."""
    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            ["termux-battery-status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SamplerError(f"termux-battery-status failed: {exc}") from exc
    elapsed = time.monotonic() - started_at
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SamplerError(
            f"termux-battery-status exited {completed.returncode}: {detail[:300]}"
        )
    return parse_battery_status(completed.stdout), elapsed


@dataclass(frozen=True)
class WindowSummary:
    average_mw: float
    median_mw: float
    sample_stdev_mw: float
    min_mw: float
    max_mw: float
    sample_count: int
    duration_seconds: float
    metadata: dict[str, object]


class BatteryWindow:
    def __init__(
        self,
        interval_seconds: float,
        command_timeout_seconds: float,
        min_samples: int,
        max_current_amps: float,
        min_voltage_volts: float,
        max_voltage_volts: float,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.min_samples = min_samples
        self.max_current_amps = max_current_amps
        self.min_voltage_volts = min_voltage_volts
        self.max_voltage_volts = max_voltage_volts
        self.samples_mw: list[float] = []
        self.current_samples_ua: list[int] = []
        self.voltage_samples_mv: list[int] = []
        self.read_durations_seconds: list[float] = []
        self.statuses: list[str] = []
        self.plugged_states: list[str] = []
        self.zero_samples_dropped = 0
        self.implausible_samples_dropped = 0
        self.interval_overruns = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._error: Exception | None = None

    def _read(self) -> None:
        try:
            reading, elapsed = read_battery_status(self.command_timeout_seconds)
            if reading.current_ua == 0 or reading.voltage_mv == 0:
                with self._lock:
                    self.zero_samples_dropped += 1
                    self.read_durations_seconds.append(elapsed)
                return
            current_amps = abs(reading.current_ua) / 1_000_000.0
            voltage_volts = abs(reading.voltage_mv) / 1000.0
            if (
                current_amps > self.max_current_amps
                or voltage_volts < self.min_voltage_volts
                or voltage_volts > self.max_voltage_volts
            ):
                with self._lock:
                    self.implausible_samples_dropped += 1
                    self.read_durations_seconds.append(elapsed)
                return
        except SamplerError as exc:
            self._error = exc
            self._stop.set()
            return
        with self._lock:
            self.samples_mw.append(reading.power_mw)
            self.current_samples_ua.append(reading.current_ua)
            self.voltage_samples_mv.append(reading.voltage_mv)
            self.read_durations_seconds.append(elapsed)
            if reading.status:
                self.statuses.append(reading.status)
            if reading.plugged:
                self.plugged_states.append(reading.plugged)
            if elapsed > self.interval_seconds:
                self.interval_overruns += 1

    def _run(self) -> None:
        deadline = time.monotonic() + self.interval_seconds
        while not self._stop.is_set():
            if self._stop.wait(max(0.0, deadline - time.monotonic())):
                return
            self._read()
            if self._error is not None:
                return
            deadline += self.interval_seconds

    def start(self) -> None:
        if self._thread is not None:
            raise SamplerError("battery window is already running")
        self._started_at = time.monotonic()
        self._read()
        if self._error is not None:
            raise SamplerError(f"initial battery read failed: {self._error}") from self._error
        self._thread = threading.Thread(
            target=self._run, name="edgesplit-battery-power", daemon=True
        )
        self._thread.start()

    def stop(self) -> WindowSummary:
        if self._thread is None or self._started_at is None:
            raise SamplerError("battery window was not started")
        self._stop.set()
        self._thread.join(timeout=self.command_timeout_seconds + 2.0)
        if self._thread.is_alive():
            raise SamplerError("battery sampler did not stop within timeout")
        if self._error is not None:
            raise SamplerError(f"battery sampler failed: {self._error}") from self._error
        duration = time.monotonic() - self._started_at
        with self._lock:
            power = list(self.samples_mw)
            current = list(self.current_samples_ua)
            voltage = list(self.voltage_samples_mv)
            reads = list(self.read_durations_seconds)
            statuses = sorted(set(self.statuses))
            plugged_states = sorted(set(self.plugged_states))
        if len(power) < self.min_samples:
            raise SamplerError(
                f"battery window has {len(power)} valid samples; need at least {self.min_samples}"
            )
        return WindowSummary(
            average_mw=statistics.fmean(power),
            median_mw=statistics.median(power),
            sample_stdev_mw=statistics.stdev(power) if len(power) > 1 else 0.0,
            min_mw=min(power),
            max_mw=max(power),
            sample_count=len(power),
            duration_seconds=duration,
            metadata={
                "source": "termux-battery-status-current-now-times-voltage",
                "command": "termux-battery-status",
                "current_unit": "uA",
                "voltage_unit": "mV",
                "power_unit": "mW",
                "current_sign_preserved_in_mean_raw": statistics.fmean(current),
                "mean_voltage_raw": statistics.fmean(voltage),
                "sample_interval_seconds": self.interval_seconds,
                "minimum_valid_samples": self.min_samples,
                "valid_sample_rate_hz": len(power) / duration if duration else 0.0,
                "command_read_mean_seconds": statistics.fmean(reads) if reads else 0.0,
                "command_read_max_seconds": max(reads) if reads else 0.0,
                "interval_overruns": self.interval_overruns,
                "dropped_zero_samples": self.zero_samples_dropped,
                "dropped_implausible_samples": self.implausible_samples_dropped,
                "max_current_amps": self.max_current_amps,
                "min_voltage_volts": self.min_voltage_volts,
                "max_voltage_volts": self.max_voltage_volts,
                "battery_statuses": statuses,
                "plugged_states": plugged_states,
            },
        )


class SamplerService:
    def __init__(
        self,
        interval_seconds: float,
        command_timeout_seconds: float,
        min_samples: int,
        max_current_amps: float,
        min_voltage_volts: float,
        max_voltage_volts: float,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.min_samples = min_samples
        self.max_current_amps = max_current_amps
        self.min_voltage_volts = min_voltage_volts
        self.max_voltage_volts = max_voltage_volts
        self._lock = threading.Lock()
        self._active_id: str | None = None
        self._active: BatteryWindow | None = None

    def health(self) -> dict[str, object]:
        try:
            reading, elapsed = read_battery_status(self.command_timeout_seconds)
        except SamplerError as exc:
            return {"status": "error", "source": "termux-battery-status", "error": str(exc)}
        return {
            "status": "ok",
            "source": "termux-battery-status-current-now-times-voltage",
            "command": "termux-battery-status",
            "current_raw": reading.current_ua,
            "voltage_raw": reading.voltage_mv,
            "current_unit": "uA",
            "voltage_unit": "mV",
            "power_magnitude_mw": reading.power_mw,
            "battery_status": reading.status,
            "plugged": reading.plugged,
            "interval_seconds": self.interval_seconds,
            "command_read_seconds": elapsed,
            "minimum_valid_samples": self.min_samples,
            "max_current_amps": self.max_current_amps,
            "min_voltage_volts": self.min_voltage_volts,
            "max_voltage_volts": self.max_voltage_volts,
        }

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._active is not None:
                raise SamplerError("a battery window is already active")
            window = BatteryWindow(
                self.interval_seconds,
                self.command_timeout_seconds,
                self.min_samples,
                self.max_current_amps,
                self.min_voltage_volts,
                self.max_voltage_volts,
            )
            window.start()
            window_id = uuid.uuid4().hex
            self._active_id = window_id
            self._active = window
            return {"status": "sampling", "window_id": window_id}

    def stop(self, window_id: str) -> dict[str, object]:
        with self._lock:
            if self._active is None or self._active_id != window_id:
                raise SamplerError("unknown or inactive battery window")
            window = self._active
            self._active = None
            self._active_id = None
        return {"status": "ok", "window_id": window_id, **asdict(window.stop())}


class Handler(BaseHTTPRequestHandler):
    server: "SamplerHttpServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("JSON body must be an object")
        return decoded

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            health = self.server.service.health()
            status = HTTPStatus.OK if health.get("status") == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
            self._json(status, health)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/v1/window/start":
                self._json(HTTPStatus.OK, self.server.service.start())
                return
            if self.path == "/v1/window/stop":
                window_id = body.get("window_id")
                if not isinstance(window_id, str) or not window_id:
                    raise ValueError("window_id is required")
                self._json(HTTPStatus.OK, self.server.service.stop(window_id))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (SamplerError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


class SamplerHttpServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: SamplerService) -> None:
        super().__init__(address, Handler)
        self.service = service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("EDGESPLIT_POWER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EDGESPLIT_POWER_PORT", "8092")))
    parser.add_argument(
        "--interval-ms", type=float,
        default=float(os.environ.get("EDGESPLIT_POWER_INTERVAL_MS", "750")),
        help="Target cadence. The subprocess duration is recorded in each window.",
    )
    parser.add_argument(
        "--command-timeout-seconds", type=float,
        default=float(os.environ.get("EDGESPLIT_POWER_COMMAND_TIMEOUT_SECONDS", "3")),
    )
    parser.add_argument(
        "--min-samples", type=int,
        default=int(os.environ.get("EDGESPLIT_POWER_MIN_SAMPLES", "3")),
    )
    parser.add_argument(
        "--max-current-amps", type=float,
        default=float(os.environ.get("EDGESPLIT_POWER_MAX_CURRENT_AMPS", "10")),
    )
    parser.add_argument(
        "--min-voltage-volts", type=float,
        default=float(os.environ.get("EDGESPLIT_POWER_MIN_VOLTAGE_VOLTS", "2.5")),
    )
    parser.add_argument(
        "--max-voltage-volts", type=float,
        default=float(os.environ.get("EDGESPLIT_POWER_MAX_VOLTAGE_VOLTS", "5.0")),
    )
    args = parser.parse_args()
    if args.interval_ms <= 0 or args.command_timeout_seconds <= 0:
        parser.error("--interval-ms and --command-timeout-seconds must be positive")
    if args.min_samples < 1 or args.max_current_amps <= 0:
        parser.error("--min-samples must be at least one and --max-current-amps positive")
    if args.min_voltage_volts <= 0 or args.min_voltage_volts >= args.max_voltage_volts:
        parser.error("voltage bounds must be positive and ordered")
    service = SamplerService(
        args.interval_ms / 1000.0,
        args.command_timeout_seconds,
        args.min_samples,
        args.max_current_amps,
        args.min_voltage_volts,
        args.max_voltage_volts,
    )
    server = SamplerHttpServer((args.host, args.port), service)
    print(json.dumps({"status": "listening", "host": args.host, "port": args.port}, sort_keys=True), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
