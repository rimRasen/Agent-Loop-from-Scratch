"""What a tool is, and how a tool says "that didn't work" (DECISIONS.md, "Tool interface").

A tool is its API spec plus a function from the model's input to a string. Expected failures
(a missing file, a timeout, a command not on the allow-list) raise `ToolError`; the loop sends the
message back to the model as an error result. Any other exception is a bug and ends the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_loop.model import ToolSpec


class ToolError(Exception):
    """An expected failure. The message goes back to the model, so make it actionable."""


@dataclass(frozen=True)
class ToolOutput:
    """A tool's result text plus facts for its trace span (merged into the span's `attrs`)."""

    text: str
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    run: Callable[[dict[str, Any]], str | ToolOutput]

    @property
    def name(self) -> str:
        return str(self.spec["name"])


# --- Input helpers: the model's input is untrusted JSON, so check every field. -------------

def require_str(inp: dict[str, Any], key: str) -> str:
    v = inp.get(key)
    if not isinstance(v, str):
        raise ToolError(f"input field {key!r} must be a string")
    return v


def optional_str(inp: dict[str, Any], key: str, default: str) -> str:
    if key not in inp:
        return default
    return require_str(inp, key)


def require_str_list(inp: dict[str, Any], key: str) -> list[str]:
    v = inp.get(key)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ToolError(f"input field {key!r} must be a list of strings")
    return list(v)


def inside(workdir: Path, candidate: str) -> Path:
    """Resolve `candidate` against `workdir`; raise ToolError if the result leaves `workdir`.

    `resolve()` follows symlinks and junctions that exist, so a link inside the working
    directory that points outside it is caught. Absolute paths, `..`, drive-relative paths
    (`C:foo`) and UNC paths (`\\\\server\\share`) all resolve outside and are rejected.
    `workdir` must already be resolved.
    """
    resolved = (workdir / candidate).resolve()
    if resolved != workdir and not resolved.is_relative_to(workdir):
        raise ToolError(f"path {candidate!r} is outside the working directory")
    return resolved
