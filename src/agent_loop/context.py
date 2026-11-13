"""Context management. Milestone 2 has only tool-output truncation; summarisation and the
context-size budget arrive in milestone 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Truncated:
    text: str
    truncated: bool
    original_chars: int


def truncate(text: str, max_chars: int) -> Truncated:
    """Keep the first and last halves of `text` with a marker between them.

    Head and tail, not just head: the end of a command's output is often the part that matters
    (the error, the test summary). The marker adds about 60 characters on top of `max_chars`.
    """
    n = len(text)
    if n <= max_chars:
        return Truncated(text, False, n)
    head = max_chars // 2
    tail = max_chars - head
    marker = f"\n[... truncated: {n - max_chars} of {n} characters omitted ...]\n"
    return Truncated(text[:head] + marker + (text[-tail:] if tail else ""), True, n)
