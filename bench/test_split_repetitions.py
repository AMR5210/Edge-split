from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edgesplit_bench import invalidate_run, log_run
from run_split_repetitions import Measurement, metric_summary, read_measurement, summarize


class RepetitionStatisticsTests(unittest.TestCase):
    def test_statistics_use_mean_median_and_sample_stdev(self) -> None:
        stats = metric_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(5, stats["n"])
        self.assertEqual(3.0, stats["mean"])
        self.assertEqual(3.0, stats["median"])
        self.assertAlmostEqual(1.5811388300841898, float(stats["sample_stdev"]))

        summary = summarize([
            Measurement(1, "split-v1", "Q4_0", 1.0, 10.0),
            Measurement(2, "split-v1", "Q4_0", 2.0, 12.0),
        ])
        self.assertEqual(1.5, summary["ttft_seconds"]["mean"])
        self.assertEqual(11.0, summary["decode_tokens_per_second"]["mean"])

    def test_read_measurement_rejects_excluded_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "bench.sqlite3"
            row_id = log_run(db, {
                "device": "test", "config": "split-v1", "model": "Qwen3-0.6B",
                "quant": "Q4_0", "prompt_len": "11", "output_len": "16",
                "ttft": "1", "tokens_per_sec": "10", "power_draw_mw": "",
                "llama_cpp_commit": "test", "notes": "",
            })
            self.assertEqual(row_id, read_measurement(db, row_id, "split-v1", "Q4_0").row_id)
            self.assertTrue(invalidate_run(db, row_id, "warmup"))
            with self.assertRaisesRegex(Exception, "does not match"):
                read_measurement(db, row_id, "split-v1", "Q4_0")


if __name__ == "__main__":
    unittest.main()
