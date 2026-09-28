"""Local file server for the step viewer (stdlib only).

The viewer reads the captured tensors straight out of the .safetensors files with HTTP Range
requests (header first, then only the byte span of the values it shows), so nothing is copied
or packaged.  Only a few directories of this repository are served.

It listens on 127.0.0.1 only.  The capture holds camera frames and values derived from the
PhysicalAI-AV dataset, whose license does not allow sharing them, so do not bind it to another
address or put it behind a tunnel or proxy.  A request must also name 127.0.0.1 or localhost in
its Host header, which keeps DNS-rebinding pages out.

    python3 viewer/serve.py              # http://127.0.0.1:8765/viewer/index.html
    python3 viewer/serve.py --port 9000  # or WALK_PORT=9000
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import posixpath
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent            # repository root
ALLOWED = ("viewer/", "walk/out/raw/", "walk/out/derived/", "walk/out/result/")
LOOPBACK = frozenset(("127.0.0.1", "localhost", "[::1]"))
CHUNK = 1 << 20
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
TYPES = {".safetensors": "application/octet-stream", ".json": "application/json; charset=utf-8",
         ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg"}


def served(rel: str) -> bool:
    return any(rel.startswith(a) or rel + "/" == a for a in ALLOWED)


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "step-viewer"
    sys_version = ""

    def log_message(self, fmt, *args):             # quiet: one line per non-range request
        if not self.headers.get("Range"):
            sys.stderr.write("%s %s %s\n" % (self.client_address[0], self.command, self.path))

    def _host_ok(self) -> bool:
        h = (self.headers.get("Host") or "").strip().lower()
        h = h[: h.find("]") + 1] if h.startswith("[") else h.split(":", 1)[0]   # drop the port
        return h in LOOPBACK

    def _resolve(self) -> Path | None:
        rel = posixpath.normpath(unquote(urlsplit(self.path).path)).lstrip("/")
        if rel in ("", "."):
            return None
        if not served(rel) or ".." in rel.split("/"):
            return None
        p = (ROOT / rel).resolve()
        try:
            real = p.relative_to(ROOT).as_posix()
        except ValueError:
            return None
        return p if served(real) else None           # a symlink may not lead out of the served folders either

    def _send_head(self):
        if not self._host_ok():
            self.send_error(HTTPStatus.MISDIRECTED_REQUEST, "unknown Host",
                            f"Open the viewer at http://127.0.0.1:{self.server.server_address[1]}/viewer/index.html")
            return None
        if urlsplit(self.path).path in ("/", "/viewer", "/viewer/"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/viewer/index.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        p = self._resolve()
        if p is None or not p.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        size = p.stat().st_size
        start, end = 0, size - 1
        partial = False
        rng = self.headers.get("Range")
        if rng:
            m = RANGE_RE.match(rng.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            if m.group(1) == "":                         # suffix range: last N bytes
                n = int(m.group(2))
                start, end = max(0, size - n), size - 1
            else:
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            partial = True
        ctype = TYPES.get(p.suffix) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return p, start, end

    def do_HEAD(self):
        self._send_head()

    def do_GET(self):
        r = self._send_head()
        if r is None:
            return
        p, start, end = r
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    buf = f.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("WALK_PORT", 8765)))
    args = ap.parse_args()
    if not (ROOT / "walk" / "out" / "raw" / "meta.json").exists():
        sys.exit(f"no capture under {ROOT / 'walk/out/raw'} (run walk/run.py first)")
    if not (ROOT / "walk" / "out" / "derived" / "manifest.json").exists():
        sys.exit(f"no derived data under {ROOT / 'walk/out/derived'} (run walk/analyze.py first)")
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on 127.0.0.1:{args.port} ({e}); pick another one with --port")
    print(f"serving {ROOT} (only {', '.join(ALLOWED)})")
    print(f"open  http://127.0.0.1:{args.port}/viewer/index.html   (this machine only)")
    print("(Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
