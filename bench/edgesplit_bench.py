#!/usr/bin/env python3
"""EdgeSplit benchmark database and best-effort power sampling utilities.

Uses only the Python standard library so the same file runs in WSL2 and Termux.
Completed benchmark rows remain clean. A separate attempt record is written
before inference so an Android process kill is visible as an unfinished attempt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    device TEXT NOT NULL,
    config TEXT NOT NULL,
    model TEXT NOT NULL,
    quant TEXT NOT NULL,
    prompt_len INTEGER NOT NULL CHECK (prompt_len >= 0),
    output_len INTEGER NOT NULL CHECK (output_len >= 0),
    ttft REAL NOT NULL CHECK (ttft >= 0),
    tokens_per_sec REAL NOT NULL CHECK (tokens_per_sec >= 0),
    power_draw_mw REAL,
    llama_cpp_commit TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'valid' CHECK (status IN ('valid', 'invalid'))
);
CREATE INDEX IF NOT EXISTS benchmark_runs_config_timestamp
    ON benchmark_runs (config, timestamp);
CREATE TABLE IF NOT EXISTS benchmark_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
    device TEXT NOT NULL,
    config TEXT NOT NULL,
    model TEXT NOT NULL,
    quant TEXT NOT NULL,
    prompt_len INTEGER NOT NULL CHECK (prompt_len >= 0),
    output_len INTEGER NOT NULL CHECK (output_len >= 0),
    llama_cpp_commit TEXT NOT NULL,
    command TEXT NOT NULL,
    benchmark_run_id INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS benchmark_attempts_status_started
    ON benchmark_attempts (status, started_at);
CREATE TABLE IF NOT EXISTS benchmark_power_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_run_id INTEGER NOT NULL REFERENCES benchmark_runs(id),
    source TEXT NOT NULL CHECK (source IN ('laptop_gpu', 'phone_battery')),
    average_mw REAL NOT NULL CHECK (average_mw >= 0),
    median_mw REAL NOT NULL CHECK (median_mw >= 0),
    sample_stdev_mw REAL NOT NULL CHECK (sample_stdev_mw >= 0),
    min_mw REAL NOT NULL CHECK (min_mw >= 0),
    max_mw REAL NOT NULL CHECK (max_mw >= 0),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (benchmark_run_id, source)
);
CREATE INDEX IF NOT EXISTS benchmark_power_windows_run_source
    ON benchmark_power_windows (benchmark_run_id, source);
"""

FIELDS = (
    "timestamp", "device", "config", "model", "quant", "prompt_len",
    "output_len", "ttft", "tokens_per_sec", "power_draw_mw",
    "llama_cpp_commit", "notes",
)
EXPORT_FIELDS = (
    "id",
) + FIELDS + (
    "laptop_gpu_power_mw",
    "phone_battery_power_mw",
    "status",
)
REQUIRED = set(FIELDS) - {"timestamp", "power_draw_mw", "notes"}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(benchmark_runs)")
    }
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE benchmark_runs ADD COLUMN status TEXT NOT NULL "
            "DEFAULT 'valid'"
        )
    return conn


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def as_row(values: dict[str, str]) -> dict[str, object]:
    missing = sorted(REQUIRED - values.keys())
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing))
    return {
        "timestamp": values.get("timestamp") or utc_now(),
        "device": values["device"],
        "config": values["config"],
        "model": values["model"],
        "quant": values["quant"],
        "prompt_len": int(values["prompt_len"]),
        "output_len": int(values["output_len"]),
        "ttft": float(values["ttft"]),
        "tokens_per_sec": float(values["tokens_per_sec"]),
        "power_draw_mw": (float(values["power_draw_mw"])
                          if values.get("power_draw_mw") not in (None, "") else None),
        "llama_cpp_commit": values["llama_cpp_commit"],
        "notes": values.get("notes") or "",
    }


def log_run(db_path: Path, values: dict[str, str]) -> int:
    row = as_row(values)
    with connect(db_path) as conn:
        cursor = conn.execute(
            f"INSERT INTO benchmark_runs ({', '.join(FIELDS)}) "
            f"VALUES ({', '.join('?' for _ in FIELDS)})",
            tuple(row[field] for field in FIELDS),
        )
        return int(cursor.lastrowid)


def start_attempt(
    db_path: Path, values: dict[str, str], command: str,
) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO benchmark_attempts (
                started_at, status, device, config, model, quant, prompt_len,
                output_len, llama_cpp_commit, command
            ) VALUES (?, 'started', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(), values["device"], values["config"], values["model"],
                values["quant"], int(values["prompt_len"]),
                int(values["output_len"]), values["llama_cpp_commit"], command,
            ),
        )
        return int(cursor.lastrowid)


def finish_attempt(
    db_path: Path, attempt_id: int, status: str, *,
    benchmark_run_id: int | None = None, error: str | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError(f"invalid terminal attempt status: {status}")
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE benchmark_attempts
            SET finished_at = ?, status = ?, benchmark_run_id = ?, error = ?
            WHERE id = ?
            """,
            (utc_now(), status, benchmark_run_id, error, attempt_id),
        )


def invalidate_run(db_path: Path, run_id: int, reason: str) -> bool:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT notes FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return False
        notes = row[0]
        suffix = f"INVALID: {reason}"
        updated_notes = f"{notes}\n{suffix}" if notes else suffix
        conn.execute(
            "UPDATE benchmark_runs SET status = 'invalid', notes = ? WHERE id = ?",
            (updated_notes, run_id),
        )
        return True


def log_power_window(
    db_path: Path, run_id: int, source: str, values: dict[str, object],
) -> None:
    """Attach one fully sampled device-power window to a benchmark run.

    `benchmark_runs.power_draw_mw` predates cross-device measurements and is
    intentionally left unset here: a split request has two distinct sources.
    The explicit source rows avoid silently combining laptop GPU and phone
    battery power into a value with ambiguous meaning.
    """
    if source not in {"laptop_gpu", "phone_battery"}:
        raise ValueError(f"unsupported power source: {source}")
    required = (
        "average_mw", "median_mw", "sample_stdev_mw", "min_mw", "max_mw",
        "sample_count", "duration_seconds",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError("missing power fields: " + ", ".join(missing))
    sample_count = int(values["sample_count"])
    if sample_count < 1:
        raise ValueError("power sample_count must be positive")
    numeric = {key: float(values[key]) for key in required if key != "sample_count"}
    if any(value < 0 for value in numeric.values()):
        raise ValueError("power values must be non-negative")
    metadata = values.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("power metadata must be a dictionary")
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"benchmark run does not exist: {run_id}")
        conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_power_windows (
                benchmark_run_id, source, average_mw, median_mw,
                sample_stdev_mw, min_mw, max_mw, sample_count,
                duration_seconds, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, source, numeric["average_mw"], numeric["median_mw"],
                numeric["sample_stdev_mw"], numeric["min_mw"], numeric["max_mw"],
                sample_count, numeric["duration_seconds"],
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            ),
        )


def power_windows(db_path: Path, run_id: int) -> dict[str, sqlite3.Row]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows_for_run = conn.execute(
            "SELECT * FROM benchmark_power_windows WHERE benchmark_run_id = ?",
            (run_id,),
        )
        return {row["source"]: row for row in rows_for_run}


def rows(db_path: Path, include_invalid: bool = False) -> Iterable[sqlite3.Row]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM benchmark_runs"
        if not include_invalid:
            query += " WHERE status = 'valid'"
        query += " ORDER BY id"
        yield from conn.execute(query)


def export_csv(
    db_path: Path, output_path: Path, include_invalid: bool = False,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EXPORT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows(db_path, include_invalid):
            exported = dict(row)
            by_source = power_windows(db_path, int(row["id"]))
            exported["laptop_gpu_power_mw"] = (
                by_source["laptop_gpu"]["average_mw"]
                if "laptop_gpu" in by_source else ""
            )
            exported["phone_battery_power_mw"] = (
                by_source["phone_battery"]["average_mw"]
                if "phone_battery" in by_source else ""
            )
            writer.writerow(exported)
            count += 1
    return count


def print_runs(db_path: Path, include_invalid: bool) -> int:
    for row in rows(db_path, include_invalid):
        print(
            f"id={row['id']} status={row['status']} device={row['device']} "
            f"config={row['config']} quant={row['quant']} "
            f"tps={row['tokens_per_sec']:.3f} notes={row['notes']!r}"
        )
    return 0


def print_attempts(db_path: Path, include_completed: bool) -> int:
    with connect(db_path) as conn:
        query = "SELECT * FROM benchmark_attempts"
        if not include_completed:
            query += " WHERE status != 'completed'"
        query += " ORDER BY id"
        for row in conn.execute(query):
            print(
                f"id={row[0]} status={row[3]} started={row[1]} "
                f"device={row[4]} config={row[5]} quant={row[7]} "
                f"run_id={row[12]} error={row[13]!r}"
            )
    return 0


def termux_battery_status_power_mw(output: str) -> float:
    """Return battery-power magnitude from Termux:API current (uA) and voltage (mV)."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"termux-battery-status returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("termux-battery-status JSON must be an object")
    current = payload.get("current")
    voltage = payload.get("voltage")
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"termux-battery-status lacks numeric current: {current!r}")
    if isinstance(voltage, bool) or not isinstance(voltage, (int, float)):
        raise ValueError(f"termux-battery-status lacks numeric voltage: {voltage!r}")
    if isinstance(current, float) and not current.is_integer():
        raise ValueError(f"termux-battery-status current is not integral: {current!r}")
    if isinstance(voltage, float) and not voltage.is_integer():
        raise ValueError(f"termux-battery-status voltage is not integral: {voltage!r}")
    # uA * mV / 1,000,000 = mW.
    return abs(int(current) * int(voltage)) / 1_000_000.0


def read_phone_power_mw() -> float:
    output = subprocess.check_output(["termux-battery-status"], text=True)
    return termux_battery_status_power_mw(output)


def read_nvidia_power_mw() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(output) * 1000.0


def sample_power(source: str, interval: float, samples: int | None) -> int:
    reader = read_nvidia_power_mw if source == "nvidia" else read_phone_power_mw
    emitted = 0
    while samples is None or emitted < samples:
        stamp = utc_now()
        try:
            print(f"{stamp},{source},{reader():.3f}", flush=True)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError, ValueError) as exc:
            print(f"power sampling failed ({source}): {exc}", file=sys.stderr)
            return 2
        emitted += 1
        if samples is None or emitted < samples:
            time.sleep(interval)
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--db", type=Path, default=Path("bench/edgesplit.sqlite3"))
    sub = command.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create or migrate the SQLite database")
    log = sub.add_parser("log", help="insert one completed benchmark row")
    for field in FIELDS:
        flag = "--" + field.replace("_", "-")
        log.add_argument(flag, dest=field, required=field in REQUIRED)
    export = sub.add_parser("export-csv", help="write benchmark rows as CSV")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-invalid", action="store_true")
    listed = sub.add_parser("list", help="display benchmark rows")
    listed.add_argument("--include-invalid", action="store_true")
    attempts = sub.add_parser("attempts", help="display failed or unfinished attempts")
    attempts.add_argument("--include-completed", action="store_true")
    invalidate = sub.add_parser("invalidate", help="exclude a completed row from clean exports")
    invalidate.add_argument("--id", type=int, required=True)
    invalidate.add_argument("--reason", required=True)
    power = sub.add_parser("power-sample", help="print timestamp, source, milliwatts")
    power.add_argument("--source", choices=("nvidia", "phone"), required=True)
    power.add_argument("--interval", type=float, default=0.5)
    power.add_argument("--samples", type=int)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.command == "init":
        connect(args.db).close()
        print(args.db)
        return 0
    if args.command == "log":
        row_id = log_run(args.db, vars(args))
        print(row_id)
        return 0
    if args.command == "export-csv":
        print(export_csv(args.db, args.output, args.include_invalid))
        return 0
    if args.command == "list":
        return print_runs(args.db, args.include_invalid)
    if args.command == "attempts":
        return print_attempts(args.db, args.include_completed)
    if args.command == "invalidate":
        if not invalidate_run(args.db, args.id, args.reason):
            print(f"benchmark row does not exist: {args.id}", file=sys.stderr)
            return 2
        print(args.id)
        return 0
    return sample_power(args.source, args.interval, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
