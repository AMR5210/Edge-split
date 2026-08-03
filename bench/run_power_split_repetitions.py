#!/usr/bin/env python3
"""Run warmed V1/V2 pairs while sampling laptop GPU and Phone 2 battery power.

The existing router still owns inference and timing.  This additive runner
opens a Phone 2 battery window and a laptop GPU window around every individual
router request, then attaches both summaries to the resulting benchmark row.
One pair is excluded as warm-up by default; five paired samples are retained.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from edgesplit_bench import invalidate_run, log_power_window
from power_capture import NvidiaPowerSampler, PhonePowerClient, PowerCaptureError, PowerWindow
from run_split_repetitions import Measurement, RepetitionError, metric_summary, post_json, read_measurement


@dataclass(frozen=True)
class PowerMeasurement:
    measurement: Measurement
    laptop_gpu: PowerWindow
    phone_battery: PowerWindow | None


def power_summary(items: list[PowerMeasurement], source: str) -> dict[str, float | int]:
    if source == "laptop_gpu":
        values = [item.laptop_gpu.average_mw for item in items]
    else:
        if any(item.phone_battery is None for item in items):
            raise RepetitionError("phone power is unavailable for this run")
        values = [item.phone_battery.average_mw for item in items if item.phone_battery]
    return metric_summary(values)


def summarize(items: list[PowerMeasurement]) -> dict[str, dict[str, float | int]]:
    result = {
        "ttft_seconds": metric_summary([item.measurement.ttft_seconds for item in items]),
        "decode_tokens_per_second": metric_summary(
            [item.measurement.decode_tokens_per_second for item in items]
        ),
        "laptop_gpu_average_mw": power_summary(items, "laptop_gpu"),
    }
    if all(item.phone_battery is not None for item in items):
        result["phone_battery_average_mw"] = power_summary(items, "phone_battery")
    return result


def evidence_item(item: PowerMeasurement) -> dict[str, object]:
    result: dict[str, object] = {
        **asdict(item.measurement),
        "laptop_gpu": asdict(item.laptop_gpu),
    }
    if item.phone_battery is not None:
        result["phone_battery"] = asdict(item.phone_battery)
    return result


def request_measured(
    *,
    router_url: str,
    endpoint: str,
    config: str,
    payload: dict[str, object],
    db_path: Path,
    quant: str,
    timeout: float,
    phone_power: PhonePowerClient | None,
    laptop_sampler_factory: Callable[[], NvidiaPowerSampler],
) -> PowerMeasurement:
    """Measure one completed router request, invalidating it if power fails."""
    window_id = phone_power.start_window() if phone_power else None
    sampler = laptop_sampler_factory()
    response: dict[str, Any] | None = None
    request_error: Exception | None = None
    laptop_window: PowerWindow | None = None
    phone_window: PowerWindow | None = None
    try:
        sampler.start()
        response = post_json(f"{router_url.rstrip('/')}/{endpoint}/generate", payload, timeout)
    except Exception as exc:
        request_error = exc
    finally:
        try:
            laptop_window = sampler.stop()
        except Exception as exc:
            if request_error is None:
                request_error = exc
        if phone_power and window_id:
            try:
                phone_window = phone_power.stop_window(window_id)
            except Exception as exc:
                if request_error is None:
                    request_error = exc

    row_id: int | None = None
    if response is not None:
        try:
            row_id = int(response["benchmark_row_id"])
        except (KeyError, TypeError, ValueError):
            request_error = request_error or RepetitionError(
                f"{endpoint} response has no benchmark row: {response!r}"
            )
    if request_error is not None:
        if row_id is not None:
            invalidate_run(db_path, row_id, f"power capture incomplete: {request_error}")
        raise RepetitionError(f"{endpoint} request/power capture failed: {request_error}") from request_error
    if row_id is None or laptop_window is None or (phone_power and phone_window is None):
        raise RepetitionError(f"{endpoint} did not return a complete power measurement")

    measurement = read_measurement(db_path, row_id, config, quant)
    try:
        log_power_window(db_path, row_id, "laptop_gpu", laptop_window.db_values())
        if phone_window is not None:
            log_power_window(db_path, row_id, "phone_battery", phone_window.db_values())
    except Exception as exc:
        invalidate_run(db_path, row_id, f"power persistence failed: {exc}")
        raise RepetitionError(f"could not persist power for row {row_id}: {exc}") from exc
    return PowerMeasurement(measurement, laptop_window, phone_window)


def request_pair(**kwargs: object) -> tuple[PowerMeasurement, PowerMeasurement]:
    v1 = request_measured(endpoint="v1", config="split-v1", **kwargs)
    v2 = request_measured(endpoint="v2", config="split-v2", **kwargs)
    return v1, v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default="http://127.0.0.1:8083")
    parser.add_argument(
        "--phone-power-url",
        help="Optional Phone 2 power sampler URL; omit when phone power is unreliable.",
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=0.1)
    parser.add_argument("--phone-power-timeout", type=float, default=15.0)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1 or args.output_tokens < 1:
        parser.error("warmups must be non-negative; runs and output-tokens must be positive")
    if args.gpu_poll_seconds <= 0 or args.phone_power_timeout <= 0:
        parser.error("power polling interval and phone timeout must be positive")

    phone_power: PhonePowerClient | None = None
    if args.phone_power_url:
        phone_power = PhonePowerClient(args.phone_power_url, args.phone_power_timeout)
        health = phone_power.health()
        if health.get("status") != "ok":
            raise RepetitionError(f"phone power sampler is not healthy: {health!r}")
    payload: dict[str, object] = {
        "prompt": args.prompt,
        "n_predict": args.output_tokens,
        "slot_id": 0,
        "temperature": 0,
        "seed": args.seed,
    }
    common: dict[str, object] = {
        "router_url": args.router_url,
        "payload": payload,
        "db_path": args.db,
        "quant": args.quant,
        "timeout": args.timeout,
        "phone_power": phone_power,
        "laptop_sampler_factory": lambda: NvidiaPowerSampler(args.gpu_poll_seconds),
    }
    warmups: list[PowerMeasurement] = []
    recorded_v1: list[PowerMeasurement] = []
    recorded_v2: list[PowerMeasurement] = []

    for _ in range(args.warmups):
        v1, v2 = request_pair(**common)
        for item in (v1, v2):
            if not invalidate_run(args.db, item.measurement.row_id, "excluded warm-up repetition"):
                raise RepetitionError(f"could not invalidate warm-up row {item.measurement.row_id}")
            warmups.append(item)
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    for _ in range(args.runs):
        v1, v2 = request_pair(**common)
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
            "laptop_power": "nvidia-smi --query-gpu=power.draw polling",
            "phone_power": (
                "vendor fuel-gauge current_avg * voltage_now, absolute magnitude"
                if phone_power else "unavailable: Phone 2 fuel-gauge reliability insufficient"
            ),
            "gpu_poll_seconds": args.gpu_poll_seconds,
        },
        "quant": args.quant,
        "excluded_warmup_rows": [evidence_item(item) for item in warmups],
        "recorded_rows": {
            "split-v1": [evidence_item(item) for item in recorded_v1],
            "split-v2": [evidence_item(item) for item in recorded_v2],
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
    try:
        raise SystemExit(main())
    except (PowerCaptureError, RepetitionError) as exc:
        raise SystemExit(f"power repetition failed: {exc}")
