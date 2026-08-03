#!/usr/bin/env python3
"""Loopback-only BFF and static server for the EdgeSplit demo dashboard.

This service is intentionally presentation-only. It never starts, stops, or
installs inference processes; it proxies explicit generation requests to the
existing local router and reads the tracked result artifacts.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = (Path(__file__).resolve().parent / "static").resolve()
RESULTS_ROOT = ROOT / "results"
DEFAULT_PROMPT = "Explain why the sky is blue in one sentence."
HISTORY_FILES = (
    ("Qwen3-0.6B", "Q4_K_M", "qwen_q4_k_m_v1_v2_laptop_gpu_power_repetitions.json"),
    ("Qwen3-0.6B", "Q4_0", "qwen_q4_0_v1_v2_laptop_gpu_power_repetitions.json"),
    ("Qwen3-0.6B", "Q8_0", "qwen_q8_0_v1_v2_laptop_gpu_power_repetitions.json"),
    ("Llama-3.2-1B-Instruct", "Q4_K_M", "llama32_q4_k_m_v1_v2_laptop_gpu_power_repetitions.json"),
    ("Llama-3.2-1B-Instruct", "Q4_0", "llama32_q4_0_v1_v2_laptop_gpu_power_repetitions.json"),
    ("Llama-3.2-1B-Instruct", "Q8_0", "llama32_q8_0_v1_v2_laptop_gpu_power_repetitions.json"),
)


class DashboardError(RuntimeError):
    """A proxied EdgeSplit request failed."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ActivityLog:
    """Small in-memory activity log; this is not a shell or terminal capture."""

    def __init__(self) -> None:
        self._items: deque[dict[str, object]] = deque(maxlen=240)
        self._lock = threading.Lock()
        self._next_id = 1

    def add(self, source: str, message: str) -> dict[str, object]:
        with self._lock:
            item = {
                "id": self._next_id,
                "timestamp": utc_now(),
                "source": source,
                "message": message,
            }
            self._next_id += 1
            self._items.append(item)
            return item

    def after(self, event_id: int) -> list[dict[str, object]]:
        with self._lock:
            return [item for item in self._items if int(item["id"]) > event_id]

    def clear(self, source: str) -> int:
        if source not in {"laptop", "phone"}:
            raise DashboardError("activity source must be laptop or phone")
        with self._lock:
            before = len(self._items)
            self._items = deque(
                (item for item in self._items if item["source"] != source),
                maxlen=self._items.maxlen,
            )
            return before - len(self._items)


def join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def request_json(url: str, *, payload: dict[str, object] | None = None, timeout: float = 8.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method="POST" if payload is not None else "GET",
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if body else {})},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise DashboardError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise DashboardError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise DashboardError(f"expected JSON object from {url}")
    return decoded


def probe(name: str, url: str) -> dict[str, object]:
    try:
        payload = request_json(url, timeout=3.0)
    except DashboardError as exc:
        return {"name": name, "state": "offline", "detail": str(exc)[:160]}
    return {"name": name, "state": "ready", "detail": payload.get("status", "ok")}


def metric(summary: dict[str, Any], key: str) -> dict[str, float | int]:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {
        field: value[field]
        for field in ("mean", "median", "sample_stdev", "n")
        if field in value
    }


def history() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model, quant, filename in HISTORY_FILES:
        path = RESULTS_ROOT / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            v1 = summary["split-v1"]
            v2 = summary["split-v2"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        records.append({
            "model": model,
            "quant": quant,
            "v1": {
                "ttft": metric(v1, "ttft_seconds"),
                "decode": metric(v1, "decode_tokens_per_second"),
                "gpu_mw": metric(v1, "laptop_gpu_average_mw"),
            },
            "v2": {
                "ttft": metric(v2, "ttft_seconds"),
                "decode": metric(v2, "decode_tokens_per_second"),
                "gpu_mw": metric(v2, "laptop_gpu_average_mw"),
            },
        })
    return records


class DashboardService:
    def __init__(self) -> None:
        phone_host = os.environ.get("EDGESPLIT_PHONE_HOST", "YOUR_PHONE_IP")
        self.router_url = os.environ.get("EDGESPLIT_ROUTER_URL", "http://127.0.0.1:8083")
        self.laptop_url = os.environ.get("EDGESPLIT_LAPTOP_URL", "http://127.0.0.1:8080")
        self.phone_url = os.environ.get("EDGESPLIT_PHONE_URL", f"http://{phone_host}:8081")
        self.phone_v1_url = os.environ.get("EDGESPLIT_PHONE_V1_URL", f"http://{phone_host}:8090")
        self.model = os.environ.get("EDGESPLIT_MODEL_NAME", "Qwen3-0.6B")
        self.quant = os.environ.get("EDGESPLIT_QUANT", "Q4_K_M")
        self.events = ActivityLog()
        self.events.add("laptop", "dashboard BFF ready; proxy-only mode enabled")

    def config(self) -> dict[str, object]:
        return {
            "model": self.model,
            "quant": self.quant,
            "default_prompt": DEFAULT_PROMPT,
            "router_url": self.router_url,
            "phone_control": "manual",
            "process_control": "disabled",
            "terminal_mode": "read-only activity mirrors",
        }

    def status(self) -> dict[str, object]:
        return {
            "services": [
                probe("Router", join_url(self.router_url, "healthz")),
                probe("Laptop prefill", join_url(self.laptop_url, "health")),
                probe("Phone decode", join_url(self.phone_url, "health")),
                probe("Phone V1 receiver", join_url(self.phone_v1_url, "healthz")),
                {
                    "name": "Phone V2 listener",
                    "state": "manual",
                    "detail": "raw TCP listener; no HTTP health probe is sent",
                },
            ],
            "checked_at": utc_now(),
        }

    def generate(self, payload: dict[str, object]) -> dict[str, object]:
        mode = payload.get("mode")
        if mode not in {"v1", "v2"}:
            raise DashboardError("mode must be v1 or v2")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DashboardError("prompt is required")
        if len(prompt) > 8000:
            raise DashboardError("prompt exceeds dashboard limit")
        request_body = {
            "prompt": prompt.strip(),
            "n_predict": int(payload.get("n_predict", 16)),
            "slot_id": int(payload.get("slot_id", 0)),
            "temperature": float(payload.get("temperature", 0)),
            "seed": int(payload.get("seed", 42)),
        }
        if not 1 <= int(request_body["n_predict"]) <= 512:
            raise DashboardError("n_predict must be between 1 and 512")
        self.events.add("laptop", f"proxy POST /{mode}/generate; prefill requested")
        self.events.add("phone", f"awaiting {mode.upper()} handoff and phone decode")
        response = request_json(
            join_url(self.router_url, f"{mode}/generate"),
            payload=request_body,
            timeout=240.0,
        )
        ttft = response.get("ttft_seconds", "?")
        decode = response.get("decode_tokens_per_second", "?")
        self.events.add("laptop", f"{mode.upper()} completed; TTFT={ttft}s, decode={decode} tok/s")
        self.events.add("phone", "decode response received by router proxy")
        return {"mode": mode, "request": request_body, "result": response}

    def clear_events(self, payload: dict[str, object]) -> dict[str, object]:
        source = payload.get("source")
        if not isinstance(source, str):
            raise DashboardError("activity source is required")
        return {"source": source, "cleared": self.events.clear(source)}


class Handler(BaseHTTPRequestHandler):
    server: "DashboardServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 32 * 1024:
            raise DashboardError("request body must be between 1 and 32768 bytes")
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise DashboardError("JSON body must be an object")
        return decoded

    def _static(self, relative: str) -> None:
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in {"/", "/index.html"}:
                self._static("index.html")
            elif self.path.startswith("/assets/"):
                self._static(self.path.removeprefix("/assets/"))
            elif self.path == "/api/config":
                self._json(HTTPStatus.OK, self.server.service.config())
            elif self.path == "/api/status":
                self._json(HTTPStatus.OK, self.server.service.status())
            elif self.path.startswith("/api/events"):
                after = 0
                if "?" in self.path:
                    query = self.path.split("?", 1)[1]
                    for part in query.split("&"):
                        if part.startswith("after="):
                            after = max(0, int(part.removeprefix("after=")))
                self._json(HTTPStatus.OK, {"events": self.server.service.events.after(after)})
            elif self.path == "/api/history":
                self._json(HTTPStatus.OK, {"records": history()})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (DashboardError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/generate":
                self._json(HTTPStatus.OK, self.server.service.generate(self._body()))
            elif self.path == "/api/events/clear":
                self._json(HTTPStatus.OK, self.server.service.clear_events(self._body()))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except DashboardError as exc:
            self.server.service.events.add("laptop", f"request failed: {exc}")
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, Handler)
        self.service = DashboardService()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("dashboard must remain loopback-only")
    server = DashboardServer((args.host, args.port))
    print(f"EdgeSplit dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
