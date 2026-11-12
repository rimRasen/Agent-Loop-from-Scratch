"""Span writer (JSONL) and reader.

One JSON object per line, one line per span, written and flushed when the span ends
(see DECISIONS.md, "Spans are written when they end"). Lines are in end order, so a child
appears before its parent; rebuild the tree from `parent_id`, never from line order.

This format is reused in later phases. Changing a field needs a DECISIONS.md entry.

Kinds in use: "run" (one per agent run; its attrs hold the stop reason), "model", "tool".
`attrs` holds small kind-specific facts, e.g. {"stop_reason": "max_steps"} on a run span or
{"truncated": true, "original_chars": 48213} on a tool span. Model spans carry
requested_model, served_model, fallback and (on a hand-over) fallback_hops; see
DECISIONS.md. Keys are added, never repurposed.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Literal

Status = Literal["ok", "error"]

# The order here is the key order on disk.
FIELDS: tuple[str, ...] = (
    "run_id",
    "span_id",
    "parent_id",
    "kind",
    "name",
    "start_ms",
    "end_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "status",
    "error",
    "attrs",
)


@dataclass(frozen=True)
class Span:
    """A finished span, exactly as it is stored on disk.

    Token and cost fields are None when they don't apply (a tool call has no tokens) or are
    unknown (a model missing from the price table has no cost). None is never written as 0.
    """

    run_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    start_ms: int
    end_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    status: Status
    error: str | None
    attrs: dict[str, object]


assert tuple(f.name for f in fields(Span)) == FIELDS


def new_run_id() -> str:
    """Sortable by start time, unique enough for a local runs/ directory."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def default_clock() -> Callable[[], int]:
    """Milliseconds since the epoch that never go backwards within one process.

    Anchored to wall-clock time once, then advanced by the monotonic clock, so start_ms/end_ms
    are readable timestamps but a system clock change can't produce a negative duration.
    """
    wall0 = time.time_ns() // 1_000_000
    mono0 = time.monotonic_ns()
    return lambda: wall0 + (time.monotonic_ns() - mono0) // 1_000_000


class OpenSpan:
    """A span that has started but not ended. The tracer hands one to the code being traced."""

    def __init__(self, run_id: str, span_id: str, parent_id: str | None,
                 kind: str, name: str, start_ms: int) -> None:
        self.run_id = run_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.start_ms = start_ms
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cost_usd: float | None = None
        self.error: str | None = None
        self.attrs: dict[str, object] = {}

    def record_usage(self, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd

    def set(self, key: str, value: object) -> None:
        """Attach a JSON-serialisable fact to the span (stored under `attrs`).

        Checked here, not at write time: a bad value found while the span is being closed would
        raise inside `finally`, mask whatever exception was in flight, and lose the span.
        """
        json.dumps(value)  # raises TypeError at the call site if not serialisable
        self.attrs[key] = value

    def fail(self, error: str) -> None:
        """Mark the span as failed without raising, e.g. a tool error returned to the model."""
        self.error = error

    def finish(self, end_ms: int) -> Span:
        return Span(
            run_id=self.run_id, span_id=self.span_id, parent_id=self.parent_id,
            kind=self.kind, name=self.name, start_ms=self.start_ms, end_ms=end_ms,
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            status="error" if self.error is not None else "ok",
            error=self.error,
            attrs=dict(self.attrs),
        )


class Tracer:
    """Creates spans for one run and appends each to a JSONL file when it ends.

    Parent tracking uses a stack: a span opened inside another span's `with` block becomes its
    child. That is correct only because the loop is synchronous (DECISIONS.md).
    """

    def __init__(self, path: Path, run_id: str | None = None,
                 clock: Callable[[], int] | None = None) -> None:
        self.run_id = run_id or new_run_id()
        self.path = path
        self._clock = clock or default_clock()
        self._stack: list[OpenSpan] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = path.open("a", encoding="utf-8", newline="\n")

    @contextmanager
    def span(self, kind: str, name: str) -> Iterator[OpenSpan]:
        """Open a span; it is written when the block exits, however it exits.

        If the block raises, the span is recorded with status "error" and the exception's type
        and message, then the exception propagates. The tracer never swallows it.
        """
        parent_id = self._stack[-1].span_id if self._stack else None
        open_span = OpenSpan(self.run_id, _new_span_id(), parent_id, kind, name, self._clock())
        self._stack.append(open_span)
        try:
            yield open_span
        except BaseException as exc:  # recorded, then re-raised: nothing is hidden
            if open_span.error is None:
                open_span.fail(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._stack.pop()
            self._write(open_span.finish(self._clock()))

    def _write(self, span: Span) -> None:
        self._file.write(json.dumps(asdict(span), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()


class TraceFormatError(ValueError):
    pass


def read_spans(path: Path) -> list[Span]:
    """Read every span in a trace file, in file (end) order.

    A final line with no trailing newline is a write torn by a hard kill; it is skipped.
    Any other bad line raises TraceFormatError naming the line, because a trace that silently
    drops spans would misreport where time and tokens went.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    torn_tail = lines.pop()  # "" if the file ends with a newline, else a torn write
    del torn_tail
    spans: list[Span] = []
    for lineno, line in enumerate(lines, 1):
        if not line:
            raise TraceFormatError(f"{path}:{lineno}: empty line")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceFormatError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict) or set(obj) != set(FIELDS):
            raise TraceFormatError(f"{path}:{lineno}: fields do not match the span format")
        if not isinstance(obj["attrs"], dict):
            raise TraceFormatError(f"{path}:{lineno}: attrs must be an object")
        if obj["status"] not in ("ok", "error"):
            raise TraceFormatError(f"{path}:{lineno}: bad status {obj['status']!r}")
        spans.append(Span(**obj))
    return spans
