#!/usr/bin/env python3
"""Run warmed paired V1/V2 requests and export exact-row statistics.

The router logs every request in its SQLite database. This laptop-side helper
calls the two router modes in paired order, marks warm-up rows invalid, and
writes a JSON evidence file containing the exact retained row IDs plus mean,
median, and sample standard deviation for TTFT and decode throughput.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from edgesplit_bench import invalidate_run


class RepetitionError(RuntimeError):
    """A router request or its benchmark row was invalid."""


@dataclass(frozen=True)
class Measurement:
    row_id: int
    config: str
    quant: str
    ttft_seconds: float
    decode_tokens_per_second: float


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RepetitionError(f"{url} returned HTTP {exc.code}: {exc.read()[:300]!r}") from exc
    except URLError as exc:
        raise RepetitionError(f"cannot reach {url}: {exc.reason}") from exc
    if not isinstance(data, dict):
        raise RepetitionError(f"unexpected response from {url}: {data!r}")
    return data


def read_measurement(db_path: Path, row_id: int, expected_config: str, expected_quant: str) -> Measurement:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT config, quant, ttft, tokens_per_sec, status FROM benchmark_runs WHERE id = ?",
            (row_id,),
        ).fetchone()
    if row is None:
        raise RepetitionError(f"router returned row {row_id}, but it is absent from {db_path}")
    config, quant, ttft, tps, status = row
    if config != expected_config or quant != expected_quant or status != "valid":
        raise RepetitionError(
            f"row {row_id} does not match expected {expected_config}/{expected_quant}: {row!r}"
        )
    return Measurement(row_id, config, quant, float(ttft), float(tps))


def request_pair(
    *, router_url: str, payload: dict[str, object], db_path: Path, quant: str,
    timeout: float,
) -> tuple[Measurement, Measurement]:
    results: list[Measurement] = []
    for endpoint, config in (("v1", "split-v1"), ("v2", "split-v2")):
        response = post_json(f"{router_url.rstrip('/')}/{endpoint}/generate", payload, timeout)
        try:
            row_id = int(response["benchmark_row_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RepetitionError(f"{endpoint} response has no benchmark row: {response!r}") from exc
        results.append(read_measurement(db_path, row_id, config, quant))
    return results[0], results[1]


def metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise RepetitionError("cannot summarize zero measurements")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(measurements: list[Measurement]) -> dict[str, dict[str, float | int]]:
    return {
        "ttft_seconds": metric_summary([item.ttft_seconds for item in measurements]),
        "decode_tokens_per_second": metric_summary(
            [item.decode_tokens_per_second for item in measurements]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default="http://127.0.0.1:8083")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1 or args.output_tokens < 1:
        parser.error("warmups must be non-negative; runs and output-tokens must be positive")

    payload: dict[str, object] = {
        "prompt": args.prompt,
        "n_predict": args.output_tokens,
        "slot_id": 0,
        "temperature": 0,
        "seed": args.seed,
    }
    warmups: list[Measurement] = []
    recorded_v1: list[Measurement] = []
    recorded_v2: list[Measurement] = []

    for _ in range(args.warmups):
        v1, v2 = request_pair(
            router_url=args.router_url, payload=payload, db_path=args.db,
            quant=args.quant, timeout=args.timeout,
        )
        for measurement in (v1, v2):
            if not invalidate_run(args.db, measurement.row_id, "excluded warm-up repetition"):
                raise RepetitionError(f"could not invalidate warm-up row {measurement.row_id}")
            warmups.append(measurement)
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    for _ in range(args.runs):
        v1, v2 = request_pair(
            router_url=args.router_url, payload=payload, db_path=args.db,
            quant=args.quant, timeout=args.timeout,
        )
        recorded_v1.append(v1)
        recorded_v2.append(v2)
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    evidence = {
        "method": {
            "prompt": args.prompt,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "slot_id": 0,
            "warmups_excluded_per_config": args.warmups,
            "recorded_pairs": args.runs,
            "statistics": "mean, median, and sample standard deviation",
        },
        "quant": args.quant,
        "excluded_warmup_rows": [asdict(item) for item in warmups],
        "recorded_rows": {
            "split-v1": [asdict(item) for item in recorded_v1],
            "split-v2": [asdict(item) for item in recorded_v2],
        },
        "summary": {
            "split-v1": summarize(recorded_v1),
            "split-v2": summarize(recorded_v2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
