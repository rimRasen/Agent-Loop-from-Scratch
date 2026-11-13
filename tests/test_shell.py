from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_loop.tools import Tool, ToolError, ToolOutput
from agent_loop.tools.shell import _build_git, _check_git, _check_rg, shell_tool

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="rg not installed")
PY = Path(sys.executable).stem  # "python": an unvetted program, allowed explicitly in tests


def out(tool: Tool, *argv: str) -> ToolOutput:
    result = tool.run({"argv": list(argv)})
    assert isinstance(result, ToolOutput)
    return result


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    if shutil.which("git"):
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        (work / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "a.txt"], cwd=work, check=True)
    return work


@pytest.fixture
def python_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure `python` resolves to the interpreter running the tests."""
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep
                       + os.environ.get("PATH", ""))


# --- allow-list and program resolution ----------------------------------------------------

def test_program_not_on_the_allow_list(repo: Path) -> None:
    with pytest.raises(ToolError, match="not on the allow-list"):
        shell_tool(repo).run({"argv": ["curl", "x"]})


@pytest.mark.parametrize("name", ["C:\\Windows\\git.exe", "./git", "bin/git"])
def test_program_must_be_a_bare_name(repo: Path, name: str) -> None:
    with pytest.raises(ToolError, match="bare name"):
        shell_tool(repo).run({"argv": [name, "status"]})


def test_empty_or_bad_argv(repo: Path) -> None:
    with pytest.raises(ToolError, match="empty"):
        shell_tool(repo).run({"argv": []})
    with pytest.raises(ToolError, match="list of strings"):
        shell_tool(repo).run({"argv": "git status"})


def test_allowed_but_not_installed(repo: Path) -> None:
    with pytest.raises(ToolError, match="not installed"):
        shell_tool(repo, allow=["definitely-not-a-program-xyz"]).run(
            {"argv": ["definitely-not-a-program-xyz"]})


def test_program_inside_the_working_directory_is_refused(
        repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = repo / "bin"
    bindir.mkdir()
    planted = bindir / ("evil.exe" if os.name == "nt" else "evil")
    planted.write_bytes(b"")
    planted.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    with pytest.raises(ToolError, match="inside the working directory"):
        shell_tool(repo, allow=["evil"]).run({"argv": ["evil"]})


@pytest.mark.skipif(os.name != "nt", reason="batch files are a Windows concern")
def test_batch_files_are_refused(repo: Path, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tool.bat").write_text("@echo hi\n")
    monkeypatch.setenv("PATH", str(bindir))
    with pytest.raises(ToolError, match="batch file"):
        shell_tool(repo, allow=["tool"]).run({"argv": ["tool"]})


# --- the path check -----------------------------------------------------------------------

@needs_git
@pytest.mark.parametrize("argv", [
    ["git", "log", "--", "../.."],
    ["git", "diff", str(Path("/").resolve())],
    ["git", "diff", "--relative=../x"],
])
def test_arguments_that_leave_the_working_directory(repo: Path, argv: list[str]) -> None:
    with pytest.raises(ToolError, match="outside the working directory"):
        shell_tool(repo).run({"argv": argv})


@needs_git
def test_pattern_values_skip_the_path_check(repo: Path) -> None:
    # "\bhello" looks like a drive-rooted path on Windows; as a -e value it is a pattern.
    result = out(shell_tool(repo), "git", "grep", "-e", "\\bhello")
    assert result.attrs["exit_code"] == 0 and "a.txt:hello" in result.text


# --- git rules ----------------------------------------------------------------------------

@pytest.mark.parametrize("args", [
    [], ["config", "core.pager", "x"], ["-c", "x=y", "status"], ["-C", "..", "status"],
    ["--git-dir=x", "status"], ["commit", "-m", "x"], ["push"],
    ["diff", "--output=x.txt"], ["diff", "--output", "x.txt"], ["diff", "--ext-diff"],
    ["log", "--textconv"], ["log", "--show-signature"], ["grep", "-Ovim", "x"],
    ["grep", "--open-files-in-pager=vim", "x"], ["log", "--format=%G?"],
    ["diff", "--no-index", "a", "b"],
])
def test_git_rules_reject(args: list[str]) -> None:
    with pytest.raises(ToolError):
        _check_git(args)


def test_git_options_after_double_dash_are_paths_not_options() -> None:
    _check_git(["log", "--", "--output"])


def test_git_gets_safety_settings_injected() -> None:
    argv = _build_git("git", ["diff", "HEAD"])
    assert argv[:2] == ["git", "--no-pager"]
    assert "core.fsmonitor=false" in argv and "core.pager=cat" in argv
    assert argv[argv.index("diff"):] == ["diff", "--no-ext-diff", "--no-textconv", "HEAD"]
    assert "--no-ext-diff" not in _build_git("git", ["status"])


@needs_git
def test_hostile_repo_config_cannot_run_a_program_through_git_status(repo: Path) -> None:
    # Without our override, `git status` runs this command (checked by hand on Git for
    # Windows). With it, the marker file must not appear.
    subprocess.run(["git", "config", "core.fsmonitor", "echo pwned > marker.txt"], cwd=repo,
                   check=True)
    result = out(shell_tool(repo), "git", "status")
    assert result.attrs["exit_code"] == 0
    assert not (repo / "marker.txt").exists()


@needs_git
def test_git_status_runs_and_reports(repo: Path) -> None:
    result = out(shell_tool(repo), "git", "status", "--short")
    assert result.text.startswith("exit code 0\n") and "a.txt" in result.text
    assert result.attrs == {"program": "git", "exit_code": 0, "program_policy": "rules"}


# --- rg rules (rg need not be installed) --------------------------------------------------

@pytest.mark.parametrize("args", [["--pre", "cat", "x"], ["--pre=cat", "x"],
                                  ["--pre-glob", "*"], ["-z", "x"], ["--search-zip", "x"],
                                  ["--hostname-bin=x", "y"], ["-iz", "x"]])
def test_rg_rules_reject(args: list[str]) -> None:
    with pytest.raises(ToolError):
        _check_rg(args)


@needs_rg
def test_rg_runs_for_real(repo: Path) -> None:
    (repo / "b.txt").write_text("say hello\nbye\n")
    result = out(shell_tool(repo), "rg", "-n", "-e", "\\bhello", "b.txt")
    assert result.attrs == {"program": "rg", "exit_code": 0, "program_policy": "rules"}
    assert "1:say hello" in result.text


@needs_rg
def test_rg_pre_is_refused_before_anything_runs(repo: Path) -> None:
    with pytest.raises(ToolError, match="runs another program"):
        shell_tool(repo).run({"argv": ["rg", "--pre=cmd", "-e", "x", "."]})


@needs_rg
@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_rg_cannot_search_through_a_junction_outside(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "s.txt").write_text("password=hunter2")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(repo / "jn"), str(outside)], check=True,
                   capture_output=True)
    with pytest.raises(ToolError, match="outside the working directory"):
        shell_tool(repo).run({"argv": ["rg", "-e", "password", "jn"]})


def test_rg_ordinary_use_passes() -> None:
    _check_rg(["-i", "-n", "--glob=*.py", "-e", "TODO", "src", "--", "-z"])


# --- unvetted programs, environment, exit codes, timeouts ---------------------------------

@pytest.mark.usefixtures("python_on_path")
def test_unvetted_program_is_marked_and_nonzero_exit_is_not_an_error(repo: Path) -> None:
    result = out(shell_tool(repo, allow=[PY]), PY, "-c", "import sys; sys.exit(3)")
    assert result.text.startswith("exit code 3")
    assert result.attrs["program_policy"] == "unvetted"


@pytest.mark.usefixtures("python_on_path")
def test_child_cannot_see_the_api_key(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    code = "import os; print(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('GIT_DIR'))"
    result = out(shell_tool(repo, allow=[PY]), PY, "-c", code)
    assert "sk-test-secret" not in result.text and "None None" in result.text


@pytest.mark.usefixtures("python_on_path")
def test_runs_in_the_working_directory(repo: Path) -> None:
    result = out(shell_tool(repo, allow=[PY]), PY, "-c", "import os; print(os.getcwd())")
    assert str(repo.resolve()) in result.text


def _alive(pid: int) -> bool:
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, check=False).stdout
        return str(pid) in listing
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.usefixtures("python_on_path")
def test_timeout_kills_the_whole_process_tree(repo: Path) -> None:
    code = ("import subprocess, sys, time; "
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "open('child.pid', 'w').write(str(c.pid)); time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(ToolError, match="timed out after 2s"):
        shell_tool(repo, allow=[PY], timeout_s=2).run({"argv": [PY, "-c", code]})
    assert time.monotonic() - start < 20
    child = int((repo / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child), "grandchild process survived the timeout"
