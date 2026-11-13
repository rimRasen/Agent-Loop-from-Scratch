"""File tools restricted to one working directory: read_file, write_file, list_dir.

Every path the model gives is resolved against the working directory and rejected if it ends up
outside it (see `inside`). Paths inside a `.git` directory are also rejected: a model that could
write `.git/config` could make `git status` (an allowed shell command) run any program it likes,
through settings like `core.fsmonitor` or filter drivers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loop.tools import Tool, ToolError, inside, optional_str, require_str

MAX_READ_BYTES = 1_000_000
MAX_LIST_ENTRIES = 1_000


def _checked(workdir: Path, candidate: str) -> Path:
    path = inside(workdir, candidate)
    rel = path.relative_to(workdir)
    if any(part.lower() == ".git" for part in rel.parts):
        raise ToolError(f"path {candidate!r} is inside a .git directory, which is off limits")
    return path


def file_tools(workdir: Path) -> list[Tool]:
    root = workdir.resolve()
    if not root.is_dir():
        raise ValueError(f"working directory {workdir} does not exist")

    def read_file(inp: dict[str, Any]) -> str:
        path = _checked(root, require_str(inp, "path"))
        if not path.is_file():
            raise ToolError(f"no such file: {inp['path']}")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(f"file is {size} bytes; the limit is {MAX_READ_BYTES}")
        return path.read_bytes().decode("utf-8", errors="replace")

    def write_file(inp: dict[str, Any]) -> str:
        path = _checked(root, require_str(inp, "path"))
        content = require_str(inp, "content")
        if path == root or path.is_dir():
            raise ToolError(f"{inp['path']} is a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" writes the text exactly as given (no \n -> \r\n on Windows).
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"wrote {len(content)} characters to {path.relative_to(root).as_posix()}"

    def list_dir(inp: dict[str, Any]) -> str:
        path = _checked(root, optional_str(inp, "path", "."))
        if not path.is_dir():
            raise ToolError(f"no such directory: {inp.get('path', '.')}")
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
        lines = [p.name + ("/" if p.is_dir() else "") for p in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... and {len(entries) - MAX_LIST_ENTRIES} more")
        return "\n".join(lines) if lines else "(empty directory)"

    path_prop = {"type": "string", "description": "Path relative to the working directory."}
    return [
        Tool({"name": "read_file",
              "description": "Read a UTF-8 text file in the working directory.",
              "input_schema": {"type": "object", "properties": {"path": path_prop},
                               "required": ["path"], "additionalProperties": False}},
             read_file),
        Tool({"name": "write_file",
              "description": "Create or overwrite a text file in the working directory. "
                             "Parent directories are created as needed.",
              "input_schema": {"type": "object",
                               "properties": {"path": path_prop,
                                              "content": {"type": "string"}},
                               "required": ["path", "content"],
                               "additionalProperties": False}},
             write_file),
        Tool({"name": "list_dir",
              "description": "List a directory in the working directory. Directories end "
                             "with '/'. Defaults to the working directory itself.",
              "input_schema": {"type": "object", "properties": {"path": path_prop},
                               "additionalProperties": False}},
             list_dir),
    ]
