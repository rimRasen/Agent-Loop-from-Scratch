"""Fetch one http(s) URL with the standard library: a time limit, a size cap, text only.

`urllib`'s timeout applies to each socket operation, not to the whole request, so a server
that trickles one byte per second would never trip it. The read loop therefore also checks an
overall deadline between chunks. One blocked read can still take up to `timeout_s`, so the
worst case is about twice `timeout_s`.

Not done (DECISIONS.md, "http_fetch uses the standard library"): private and loopback
addresses are not blocked.
"""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
from http.client import HTTPMessage
from typing import IO, Any, Protocol
from urllib.parse import urlsplit

from agent_loop.tools import Tool, ToolError, ToolOutput, require_str

CHUNK = 64 * 1024
MAX_REDIRECTS = 5
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/javascript",
              "application/xhtml+xml", "application/ld+json")


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ToolError(f"only http:// and https:// URLs are allowed, got {url!r}")


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow at most MAX_REDIRECTS redirects, and only to http(s) URLs."""

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int,
                         msg: str, headers: HTTPMessage, newurl: str
                         ) -> urllib.request.Request | None:
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Readable(Protocol):
    def read1(self, n: int = ..., /) -> bytes: ...


def _read_capped(resp: _Readable, max_bytes: int, deadline: float, timeout_s: float) -> bytes:
    """Read until EOF, the size cap, or the deadline.

    `read1`, not `read`: `read(n)` keeps reading until it has n bytes, so a server trickling
    one byte at a time would hold it far past the deadline. `read1` returns what has arrived.
    """
    buf = bytearray()
    while True:
        if time.monotonic() > deadline:
            raise ToolError(f"timed out after {timeout_s:g}s while reading the body")
        chunk = resp.read1(CHUNK)
        if not chunk:
            return bytes(buf)
        buf += chunk
        if len(buf) > max_bytes:
            raise ToolError(f"response is larger than the {max_bytes}-byte limit")


def http_fetch_tool(timeout_s: float = 10.0, max_bytes: int = 1_000_000) -> Tool:
    opener = urllib.request.build_opener(_Redirects)

    def run(inp: dict[str, Any]) -> ToolOutput:
        url = require_str(inp, "url")
        _check_url(url)
        deadline = time.monotonic() + timeout_s
        req = urllib.request.Request(url, headers={"User-Agent": "agent-loop/0.1"})
        try:
            with opener.open(req, timeout=timeout_s) as resp:
                status = resp.status
                ctype = resp.headers.get_content_type()
                charset = resp.headers.get_content_charset() or "utf-8"
                final_url = resp.geturl()
                if not ctype.startswith(TEXT_TYPES):
                    raise ToolError(f"content type {ctype} is not text; not fetched")
                body = _read_capped(resp, max_bytes, deadline, timeout_s)
        except urllib.error.HTTPError as exc:
            detail = _read_capped(exc, 2_000, deadline, timeout_s).decode("utf-8", "replace")
            raise ToolError(f"HTTP {exc.code} {exc.reason}\n{detail}") from None
        except urllib.error.URLError as exc:  # DNS failure, refused connection, TLS error
            raise ToolError(f"request failed: {exc.reason}") from None
        except TimeoutError:
            raise ToolError(f"timed out after {timeout_s:g}s") from None
        except (OSError, http.client.HTTPException) as exc:  # reset, truncated body, bad TLS
            raise ToolError(f"request failed: {type(exc).__name__}: {exc}") from None
        try:
            text = body.decode(charset, errors="replace")
        except LookupError:  # unknown charset name in the header
            text = body.decode("utf-8", errors="replace")
        attrs: dict[str, object] = {"http_status": status, "content_type": ctype,
                                    "bytes": len(body)}
        if final_url != url:
            attrs["final_url"] = final_url
        return ToolOutput(f"HTTP {status} {ctype} {final_url}\n\n{text}", attrs)

    return Tool({
        "name": "http_fetch",
        "description": (f"GET an http(s) URL and return the text body. Text content only; "
                        f"bodies over {max_bytes} bytes are rejected; times out after "
                        f"{timeout_s:g}s."),
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}},
                         "required": ["url"], "additionalProperties": False},
    }, run)
