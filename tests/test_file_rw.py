from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_loop.tools import Tool, ToolError, file_rw

# A drive-relative path on another drive ("D:x") points outside. One on the same drive as
# the working directory is joined to it by pathlib, so it stays inside, which is safe.
OTHER_DRIVE = ("D:" if Path.cwd().drive.upper() != "D:" else "C:") if os.name == "nt" else "/"


@pytest.fixture
def tools(tmp_path: Path) -> dict[str, Tool]:
    work = tmp_path / "work"
    work.mkdir()
    return {t.name: t for t in file_rw.file_tools(work)}


def call(tools: dict[str, Tool], name: str, **inp: Any) -> str:
    out = tools[name].run(inp)
    assert isinstance(out, str)
    return out


def test_write_then_read_round_trip_with_new_dirs(tools: dict[str, Tool],
                                                  tmp_path: Path) -> None:
    assert "wrote 5 characters to a/b/c.txt" in call(tools, "write_file", path="a/b/c.txt",
                                                    content="x\ny\nz")
    assert call(tools, "read_file", path="a/b/c.txt") == "x\ny\nz"
    assert (tmp_path / "work/a/b/c.txt").read_bytes() == b"x\ny\nz"  # no CRLF on Windows


@pytest.mark.parametrize("path", [
    "../outside.txt", "a/../../outside.txt", str(Path("/").resolve() / "outside.txt"),
    "\\\\server\\share\\x", OTHER_DRIVE + "outside.txt",
])
def test_paths_outside_the_working_directory_are_rejected(tools: dict[str, Tool],
                                                         path: str) -> None:
    for name, extra in (("read_file", {}), ("write_file", {"content": "x"}), ("list_dir", {})):
        with pytest.raises(ToolError, match="outside the working directory"):
            tools[name].run({"path": path, **extra})


@pytest.mark.parametrize("path", [".git/config", "sub/.GIT/hooks/pre-commit", ".git"])
def test_git_directories_are_off_limits(tools: dict[str, Tool], path: str) -> None:
    with pytest.raises(ToolError, match=r"\.git directory"):
        tools["write_file"].run({"path": path, "content": "x"})


def test_symlink_pointing_outside_is_rejected(tools: dict[str, Tool], tmp_path: Path) -> None:
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "s.txt").write_text("secret")
    try:
        (tmp_path / "work/link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs extra privileges on this system")
    with pytest.raises(ToolError, match="outside"):
        tools["read_file"].run({"path": "link/s.txt"})


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_junction_pointing_outside_is_rejected(tools: dict[str, Tool], tmp_path: Path) -> None:
    # A junction escapes like a symlink but needs no privilege, so this test always runs.
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "s.txt").write_text("secret")
    make_junction(tmp_path / "work/jn", outside)
    for name, extra in (("read_file", {"path": "jn/s.txt"}),
                        ("write_file", {"path": "jn/new.txt", "content": "x"}),
                        ("list_dir", {"path": "jn"})):
        with pytest.raises(ToolError, match="outside"):
            tools[name].run(extra)
    assert not (outside / "new.txt").exists()


def make_junction(link: Path, target: Path) -> None:
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True,
                   capture_output=True)


def test_read_errors(tools: dict[str, Tool], tmp_path: Path,
                     monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ToolError, match="no such file"):
        tools["read_file"].run({"path": "missing.txt"})
    (tmp_path / "work/big.txt").write_text("x" * 20)
    monkeypatch.setattr(file_rw, "MAX_READ_BYTES", 10)
    with pytest.raises(ToolError, match="limit is 10"):
        tools["read_file"].run({"path": "big.txt"})


def test_invalid_utf8_is_replaced_not_fatal(tools: dict[str, Tool], tmp_path: Path) -> None:
    (tmp_path / "work/bin").write_bytes(b"ok\xff")
    assert call(tools, "read_file", path="bin") == "ok�"


def test_bad_input_types_are_tool_errors(tools: dict[str, Tool]) -> None:
    with pytest.raises(ToolError, match="'path' must be a string"):
        tools["read_file"].run({"path": 3})
    with pytest.raises(ToolError, match="'content'"):
        tools["write_file"].run({"path": "a.txt"})


def test_write_to_a_directory_is_an_error(tools: dict[str, Tool], tmp_path: Path) -> None:
    (tmp_path / "work/d").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        tools["write_file"].run({"path": "d", "content": "x"})


def test_list_dir(tools: dict[str, Tool], tmp_path: Path) -> None:
    assert call(tools, "list_dir") == "(empty directory)"
    (tmp_path / "work/sub").mkdir()
    (tmp_path / "work/B.txt").write_text("")
    (tmp_path / "work/a.txt").write_text("")
    assert call(tools, "list_dir", path=".") == "a.txt\nB.txt\nsub/"
    with pytest.raises(ToolError, match="no such directory"):
        tools["list_dir"].run({"path": "nope"})


def test_missing_workdir_is_a_startup_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        file_rw.file_tools(tmp_path / "nope")
