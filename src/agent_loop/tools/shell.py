"""Run one allow-listed program with an argument list: no shell, a timeout, a sandboxed cwd.

Checks, in order (each failure is a ToolError returned to the model):
1. `argv[0]` is a bare program name on the allow-list (no paths, no `.bat`/`.cmd`).
2. It resolves on PATH to a file outside the working directory, so the model can't plant its
   own `git.exe` next to the files it writes.
3. Every argument that could be a path stays inside the working directory.
4. Programs with a `Policy` (the defaults, git and rg) get their own argument rules.
   Programs the user adds without one get only checks 1-3 and are marked "unvetted" in the
   tool span: we don't know which of their options run other programs.

The child gets a cleaned environment: no ANTHROPIC_* variables (so the API key can't be
printed by a command), no GIT_* or RIPGREP_CONFIG_PATH (which could redirect git or rg).
On timeout the whole process tree is killed (taskkill /T on Windows, the process group
elsewhere).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_loop.tools import Tool, ToolError, ToolOutput, inside, require_str_list

INTERPRETERS = frozenset({
    "python", "python3", "py", "uv", "uvx", "node", "deno", "bun", "ruby", "perl", "php",
    "bash", "sh", "zsh", "cmd", "powershell", "pwsh", "wsl", "java", "dotnet", "npx", "npm",
})


@dataclass(frozen=True)
class Policy:
    """Argument rules for one program.

    `build` turns the model's arguments into the real argv after the rules pass (git uses it
    to inject safety settings). `exempt_values` are options whose *next* argument (or
    `=value`) is a pattern, not a path, so the path check skips it.
    """

    check: Callable[[list[str]], None]
    exempt_values: frozenset[str] = frozenset()
    build: Callable[[str, list[str]], list[str]] = field(
        default=lambda exe, args: [exe, *args])


# --- git: read-only subcommands, nothing that runs another program -------------------------

GIT_SUBCOMMANDS = frozenset({"status", "diff", "log", "show", "ls-files", "grep"})
GIT_BLOCKED = (
    "--output",          # writes a file
    "--ext-diff",        # runs an external diff program
    "--textconv",        # runs a textconv filter program
    "--show-signature",  # runs gpg
    "-O", "--open-files-in-pager",  # git grep: runs a pager program
    "--no-index",        # git diff of arbitrary files, not the repo
)
# Settings that override the repo's own config, so a hostile .git/config can't use them to run
# a program. `-c` also reaches child git processes (submodules) via GIT_CONFIG_PARAMETERS.
GIT_SAFETY = [
    "--no-pager",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "log.showSignature=false",
]


def _check_git(args: list[str]) -> None:
    if not args or args[0] not in GIT_SUBCOMMANDS:
        raise ToolError("git: the first argument must be one of "
                        f"{', '.join(sorted(GIT_SUBCOMMANDS))} (no global options)")
    for a in args[1:]:
        if a == "--":
            break
        for flag in GIT_BLOCKED:
            if a == flag or a.startswith(flag + "=") or (
                    len(flag) == 2 and a.startswith(flag)):  # -Ofoo
                raise ToolError(f"git: option {a!r} is not allowed")
        if "%G" in a:  # --format=%G? etc. make git run gpg to check signatures
            raise ToolError(f"git: signature placeholders (%G...) are not allowed: {a!r}")


def _build_git(exe: str, args: list[str]) -> list[str]:
    sub, rest = args[0], args[1:]
    extra = ["--no-ext-diff", "--no-textconv"] if sub in ("diff", "log", "show") else []
    return [exe, *GIT_SAFETY, sub, *extra, *rest]


# --- rg: no options that run other programs ------------------------------------------------

RG_BLOCKED = ("--pre", "--pre-glob", "--hostname-bin", "--search-zip", "-z")


def _check_rg(args: list[str]) -> None:
    for a in args:
        if a == "--":
            break
        if a.startswith("-") and not a.startswith("--") and len(a) > 2:
            # rg accepts bundles like -iz; parsing them correctly is fragile, so refuse them.
            raise ToolError(f"rg: combined short options like {a!r} are not allowed; "
                            "pass each option separately (use -e PATTERN for patterns)")
        for flag in RG_BLOCKED:
            if a == flag or a.startswith(flag + "="):
                raise ToolError(f"rg: option {a!r} is not allowed (it runs another program)")


DEFAULT_POLICIES: dict[str, Policy] = {
    "git": Policy(check=_check_git, build=_build_git,
                  # only options that always take a value: an exempt option that doesn't would
                  # let the argument after it skip the path check.
                  exempt_values=frozenset({"-e", "--grep", "--author", "--committer", "-S",
                                           "-G"})),
    "rg": Policy(check=_check_rg, exempt_values=frozenset({"-e", "--regexp", "-r",
                                                           "--replace"})),
}
DEFAULT_ALLOW = tuple(DEFAULT_POLICIES)


def _path_candidates(args: list[str], exempt: frozenset[str]) -> Iterable[str]:
    """Every argument and `--opt=value` value that could name a path.

    Without knowing a program's grammar we can't tell paths from other text, so this is
    deliberately broad: anything that resolves outside the working directory is rejected.
    Relative text that isn't a path ("HEAD~1", "foo|bar") resolves inside and passes.
    """
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in exempt:
            skip_next = True
            continue
        if a.startswith("-") and "=" in a:
            opt, value = a.split("=", 1)
            if opt not in exempt:
                yield value
            continue
        if a.startswith("-"):
            continue
        yield a


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("ANTHROPIC_", "GIT_")) and k.upper() !=
           "RIPGREP_CONFIG_PATH"}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _program_key(name: str) -> str:
    key = name.lower() if os.name == "nt" else name
    return key[:-4] if key.endswith(".exe") else key


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def shell_tool(workdir: Path, allow: Iterable[str] = DEFAULT_ALLOW, timeout_s: float = 30.0,
               policies: dict[str, Policy] | None = None) -> Tool:
    root = workdir.resolve()
    pols = DEFAULT_POLICIES if policies is None else policies
    allowed = sorted({_program_key(p) for p in allow})

    def run(inp: dict[str, Any]) -> ToolOutput:
        argv = require_str_list(inp, "argv")
        if not argv:
            raise ToolError("argv must not be empty")
        name, args = argv[0], argv[1:]
        if any(c in name for c in "/\\:"):
            raise ToolError(f"give the program as a bare name, not a path: {name!r}")
        key = _program_key(name)
        if key not in allowed:
            raise ToolError(f"{name!r} is not on the allow-list ({', '.join(allowed)})")
        exe = shutil.which(key)
        if exe is None:
            raise ToolError(f"{name!r} is allowed but not installed (not found on PATH)")
        exe_path = Path(exe).resolve()
        if exe_path.is_relative_to(root):
            raise ToolError(f"{name!r} resolves to a file inside the working directory")
        if exe_path.suffix.lower() in (".bat", ".cmd"):
            # Batch files run through cmd.exe, which re-parses arguments: argument injection.
            raise ToolError(f"{name!r} is a batch file ({exe_path.name}); refusing to run it")

        policy = pols.get(key)
        for cand in _path_candidates(args, policy.exempt_values if policy else frozenset()):
            inside(root, cand)
        if policy:
            policy.check(args)
            real_argv = policy.build(str(exe_path), args)
        else:
            real_argv = [str(exe_path), *args]

        proc = subprocess.Popen(real_argv, cwd=root, env=_child_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            raise ToolError(f"timed out after {timeout_s:g}s and was killed") from None
        text = out.decode("utf-8", errors="replace")
        attrs: dict[str, object] = {"program": key, "exit_code": proc.returncode,
                                    "program_policy": "rules" if policy else "unvetted"}
        # A non-zero exit is information for the model (failing tests, no grep match), not a
        # tool failure, so it doesn't count toward the tool error limit.
        return ToolOutput(f"exit code {proc.returncode}\n{text}", attrs)

    return Tool({
        "name": "shell",
        "description": (
            "Run one program in the working directory, without a shell: no pipes, "
            "redirection, globbing or `&&`. argv[0] must be one of: "
            f"{', '.join(allowed)}. Paths must stay inside the working directory. "
            "git allows only status/diff/log/show/ls-files/grep. "
            f"Times out after {timeout_s:g}s. Returns the exit code and combined output."),
        "input_schema": {"type": "object",
                         "properties": {"argv": {"type": "array", "items": {"type": "string"},
                                                 "description": "Program and arguments."}},
                         "required": ["argv"], "additionalProperties": False},
    }, run)
