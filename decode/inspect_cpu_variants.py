#!/data/data/com.termux/files/usr/bin/python3
"""Inspect Android CPU backend eligibility and dotprod instructions."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import shutil
import subprocess
from pathlib import Path


def cpu_features() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("features"):
            return line
    return "Features line not found"


def backend_score(path: Path) -> int | str:
    try:
        library = ctypes.CDLL(str(path))
        score = library.ggml_backend_score
        score.argtypes = []
        score.restype = ctypes.c_int
        return int(score())
    except OSError as exc:
        return f"load error: {exc}"
    except AttributeError as exc:
        return f"score symbol error: {exc}"


def sdot_count(path: Path) -> int | str:
    objdump = shutil.which("llvm-objdump") or shutil.which("objdump")
    if objdump is None:
        return "objdump unavailable"
    result = subprocess.run(
        [objdump, "--mattr=+dotprod", "-d", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        return f"objdump failed: {result.stderr.strip()}"
    return len(re.findall(r"\bsdot\b", result.stdout, flags=re.IGNORECASE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.bin_dir.is_dir():
        parser.error(f"not a directory: {args.bin_dir}")

    variants = []
    for path in sorted(args.bin_dir.glob("libggml-cpu-android_*.so")):
        variants.append({
            "library": path.name,
            "score": backend_score(path),
            "sdot_instruction_count": sdot_count(path),
        })
    if not variants:
        parser.error(f"no Android CPU variants in {args.bin_dir}")

    print(json.dumps({
        "cpu_features": cpu_features(),
        "variants": variants,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
