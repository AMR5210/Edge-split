from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from edgesplit_bench import (
    log_power_window,
    log_run,
    power_windows,
    termux_battery_status_power_mw,
)
from power_capture import NvidiaPowerSampler, PhonePowerClient, summarize_samples
from run_power_split_repetitions import PowerMeasurement, summarize
from run_split_repetitions import Measurement


class _PhoneHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _write(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._write({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.endswith("/start"):
            self._write({"status": "sampling", "window_id": "test-window"})
            return
        self._write({
            "status": "ok", "window_id": "test-window", "average_mw": 4000.0,
            "median_mw": 4000.0, "sample_stdev_mw": 0.0, "min_mw": 4000.0,
            "max_mw": 4000.0, "sample_count": 3, "duration_seconds": 0.3,
            "metadata": {"current_unit": "uA"},
        })


class PowerCaptureTests(unittest.TestCase):
    def test_termux_battery_status_power_uses_documented_units(self) -> None:
        payload = json.dumps({"current": -139000, "voltage": 3755})
        self.assertAlmostEqual(521.945, termux_battery_status_power_mw(payload))
        with self.assertRaises(ValueError):
            termux_battery_status_power_mw(json.dumps({"current": -139000}))

    def test_laptop_only_summary_omits_phone_metric(self) -> None:
        laptop = summarize_samples([2000.0, 3000.0], 1.0, {"source": "test"})
        result = summarize([
            PowerMeasurement(Measurement(1, "split-v1", "Q4_0", 1.0, 10.0), laptop, None),
            PowerMeasurement(Measurement(2, "split-v1", "Q4_0", 2.0, 12.0), laptop, None),
        ])
        self.assertEqual(2500.0, result["laptop_gpu_average_mw"]["mean"])
        self.assertNotIn("phone_battery_average_mw", result)

    def test_summary_and_gpu_sampler(self) -> None:
        summary = summarize_samples([1000.0, 2000.0, 3000.0], 0.3, {"source": "test"})
        self.assertEqual(2000.0, summary.average_mw)
        self.assertEqual(1000.0, summary.sample_stdev_mw)

        values = iter([1200.0, 1300.0, 1400.0])
        sampler = NvidiaPowerSampler(interval_seconds=0.001, reader=lambda: next(values))
        sampler.start()
        window = sampler.stop()
        self.assertGreaterEqual(window.sample_count, 1)
        self.assertEqual(1200.0, window.average_mw)

    def test_phone_client_and_power_persistence(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _PhoneHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = PhonePowerClient(f"http://127.0.0.1:{server.server_port}")
            self.assertEqual("ok", client.health()["status"])
            window_id = client.start_window()
            phone = client.stop_window(window_id)
            self.assertEqual(4000.0, phone.average_mw)

            with tempfile.TemporaryDirectory() as directory:
                db = Path(directory) / "bench.sqlite3"
                run_id = log_run(db, {
                    "device": "test", "config": "split-v1", "model": "test",
                    "quant": "Q4_0", "prompt_len": "1", "output_len": "1",
                    "ttft": "1", "tokens_per_sec": "1", "power_draw_mw": "",
                    "llama_cpp_commit": "test", "notes": "",
                })
                log_power_window(db, run_id, "phone_battery", phone.db_values())
                stored = power_windows(db, run_id)
                self.assertEqual(4000.0, stored["phone_battery"]["average_mw"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
