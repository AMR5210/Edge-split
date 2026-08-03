#!/data/data/com.termux/files/usr/bin/python
"""Receive V1 state files into llama-server's slot-save directory."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def state_name(path: str) -> str | None:
    path = unquote(path)
    if not path.startswith("/state/"):
        return None
    name = path.removeprefix("/state/")
    if not name or Path(name).name != name or name in {".", ".."}:
        return None
    return name


def handler(state_dir: Path, max_bytes: int) -> type[BaseHTTPRequestHandler]:
    class Receiver(BaseHTTPRequestHandler):
        def reply(self, status: HTTPStatus, body: dict[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/healthz":
                self.reply(HTTPStatus.OK, {"status": "ok"})
            else:
                self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            name = state_name(urlparse(self.path).path)
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                size = -1
            if name is None:
                self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif size < 0 or size > max_bytes:
                self.reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid content length"})
            else:
                target = state_dir / name
                temporary = state_dir / f".{name}.uploading"
                try:
                    with temporary.open("wb") as output:
                        remaining = size
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ConnectionError("upload ended early")
                            output.write(chunk)
                            remaining -= len(chunk)
                    os.replace(temporary, target)
                except (OSError, ConnectionError) as exc:
                    temporary.unlink(missing_ok=True)
                    self.reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self.reply(HTTPStatus.CREATED, {"filename": name, "stored_bytes": size})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Receiver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.bind, args.port), handler(args.state_dir, args.max_bytes))
    print(f"state receiver listening on http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
