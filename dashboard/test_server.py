from __future__ import annotations

import json
import unittest
from threading import Thread
from urllib.request import Request, urlopen

import server


class DashboardServiceTests(unittest.TestCase):
    def test_history_reads_all_tracked_power_artifacts(self) -> None:
        records = server.history()
        self.assertEqual(6, len(records))
        self.assertEqual({"Qwen3-0.6B", "Llama-3.2-1B-Instruct"}, {
            record["model"] for record in records
        })
        self.assertTrue(all(record["v1"]["ttft"] for record in records))
        self.assertTrue(all(record["v2"]["gpu_mw"] for record in records))

    def test_generate_rejects_unknown_mode_without_contacting_router(self) -> None:
        service = server.DashboardService()
        with self.assertRaises(server.DashboardError):
            service.generate({"mode": "invalid", "prompt": "test"})

    def test_static_assets_are_served_from_the_assets_route(self) -> None:
        httpd = server.DashboardServer(("127.0.0.1", 0))
        worker = Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            port = httpd.server_address[1]
            with urlopen(f"http://127.0.0.1:{port}/assets/app.js", timeout=2) as response:
                self.assertEqual(200, response.status)
                self.assertIn(b"pollStatus", response.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=2)

    def test_clear_activity_endpoint_removes_only_requested_source(self) -> None:
        httpd = server.DashboardServer(("127.0.0.1", 0))
        httpd.service.events.add("phone", "phone event")
        httpd.service.events.add("laptop", "laptop event")
        worker = Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            request = Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/api/events/clear",
                data=json.dumps({"source": "laptop"}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=2) as response:
                payload = json.loads(response.read())
            self.assertGreaterEqual(payload["cleared"], 2)
            self.assertEqual(["phone"], [item["source"] for item in httpd.service.events.after(0)])
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
