"""
prefetch/specialist_proxy.py — single-endpoint HTTP forwarder for disjoint
specialist GPU pools (chemgraph_screen_pool workload).

The ChemGraph worker LLM client is built ONCE with a fixed base_url; with
disjoint pools each specialist vLLM server lives on its own port, so the
client cannot reach "whichever specialist is current" directly.  This proxy
binds the client-facing port and forwards every request to the currently
selected upstream port.  The adapter flips the target at task boundaries via
set_target(), always AFTER wait_until_ready() — requests never race a booting
engine.

Forwarding is store-and-forward (the full upstream response is read before it
is relayed).  ChemGraph agent calls are non-streaming ChatOpenAI invokes, so
buffering costs nothing next to LLM latency.  Upstream connections are opened
per request; client-side keep-alive is preserved.
"""
from __future__ import annotations

import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# LLM generations on a busy engine can take minutes.
_UPSTREAM_TIMEOUT_S = 1800.0

_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te",
                "proxy-connection", "upgrade", "host", "content-length"}


class _ForwardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silence per-request stderr logging (one line per LLM call is noise).
    def log_message(self, fmt, *args):
        pass

    def _forward(self) -> None:
        target = self.server.proxy.target_port  # type: ignore[attr-defined]
        if target is None:
            self.send_error(503, "no specialist engine selected yet")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(
            "127.0.0.1", target, timeout=_UPSTREAM_TIMEOUT_S)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in _HOP_HEADERS}
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in _HOP_HEADERS:
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            try:
                self.send_error(502, f"specialist proxy upstream error: {exc}")
            except Exception:
                pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _forward


class SpecialistProxy:
    """Threaded HTTP forwarder with a switchable upstream port."""

    def __init__(self, listen_port: int, initial_target: int | None = None):
        self.listen_port = listen_port
        self._target = initial_target
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", listen_port), _ForwardHandler)
        self._server.proxy = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="specialist_proxy",
            daemon=True)

    @property
    def target_port(self) -> int | None:
        return self._target

    def set_target(self, port: int) -> None:
        self._target = port

    def start(self) -> None:
        self._thread.start()
        print(f"[specialist_proxy] listening on :{self.listen_port} "
              f"(target={self._target})", flush=True)

    def stop(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
