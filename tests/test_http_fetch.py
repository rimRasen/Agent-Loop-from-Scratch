"""http_fetch against a local server on 127.0.0.1: no external network."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_loop.tools import ToolError, ToolOutput
from agent_loop.tools.http_fetch import http_fetch_tool


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # keep test output quiet
        pass

    def send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/text":
            self.send(200, "text/plain; charset=utf-8", "héllo".encode())
        elif self.path == "/latin1":
            self.send(200, "text/html; charset=latin-1", "café".encode("latin-1"))
        elif self.path == "/json":
            self.send(200, "application/json", b'{"a": 1}')
        elif self.path == "/big":
            self.send(200, "text/plain", b"x" * 2_000)
        elif self.path == "/png":
            self.send(200, "image/png", b"\x89PNG")
        elif self.path == "/missing":
            self.send(404, "text/plain", b"not here")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/text")
            self.end_headers()
        elif self.path == "/redirect-file":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/slow":  # trickles bytes so no single read times out
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "100")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass  # the client gave up, as it should


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")


def fetch(url: str, **kw: float) -> ToolOutput:
    out = http_fetch_tool(**kw).run({"url": url})  # type: ignore[arg-type]
    assert isinstance(out, ToolOutput)
    return out


def test_text_body_with_status_and_type(base: str) -> None:
    out = fetch(base + "/text")
    assert out.text == f"HTTP 200 text/plain {base}/text\n\nhéllo"
    assert out.attrs == {"http_status": 200, "content_type": "text/plain", "bytes": 6}


def test_charset_from_header_and_json_is_text(base: str) -> None:
    assert fetch(base + "/latin1").text.endswith("café")
    assert fetch(base + "/json").text.endswith('{"a": 1}')


def test_size_cap(base: str) -> None:
    with pytest.raises(ToolError, match="1000-byte limit"):
        fetch(base + "/big", max_bytes=1_000)
    assert fetch(base + "/big", max_bytes=2_000).attrs["bytes"] == 2_000


def test_overall_deadline_catches_a_trickling_server(base: str) -> None:
    start = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        fetch(base + "/slow", timeout_s=1.0)
    assert time.monotonic() - start < 3  # 100 bytes at 0.1s each would take 10s


def test_non_text_is_refused(base: str) -> None:
    with pytest.raises(ToolError, match="image/png is not text"):
        fetch(base + "/png")


def test_http_error_status_is_a_tool_error_with_the_body(base: str) -> None:
    with pytest.raises(ToolError, match="HTTP 404") as exc:
        fetch(base + "/missing")
    assert "not here" in str(exc.value)


def test_redirects_are_followed_and_recorded(base: str) -> None:
    out = fetch(base + "/redirect")
    assert out.text.endswith("héllo") and out.attrs["final_url"] == base + "/text"


def test_redirect_to_another_scheme_is_refused(base: str) -> None:
    # urllib refuses file:// itself before our handler sees it; ours covers the rest.
    with pytest.raises(ToolError, match="not allowed|only http"):
        fetch(base + "/redirect-file")


def test_redirect_loop_is_an_error(base: str) -> None:
    with pytest.raises(ToolError, match="HTTP 302"):
        fetch(base + "/loop")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "http://", "nope"])
def test_only_http_urls(url: str) -> None:
    with pytest.raises(ToolError, match="only http"):
        fetch(url)


def test_connection_refused_is_a_tool_error() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # bound but not listening: connections are refused
        with pytest.raises(ToolError, match="request failed"):
            fetch(f"http://127.0.0.1:{port}/", timeout_s=2.0)
