"""Command-line entry point: `agent run "<task>"`. `agent trace` arrives in milestone 5."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from agent_loop.agents import WORKER
from agent_loop.loop import Approver, ExtensionRequest, Limits, decline, run_agent
from agent_loop.model import AnthropicModel, ModelConfigError
from agent_loop.tools import Tool
from agent_loop.tools.file_rw import file_tools
from agent_loop.tools.http_fetch import http_fetch_tool
from agent_loop.tools.shell import DEFAULT_ALLOW, DEFAULT_POLICIES, INTERPRETERS, shell_tool
from agent_loop.trace import Tracer, new_run_id

LIMIT_HELP = {
    "max_steps": "model calls per run, retries included",
    "token_budget": "billed input+output tokens per run; you are asked before going over",
    "max_tool_errors": "tool failures in a row before the run stops",
    "tool_output_chars": "characters of each tool result kept in context",
    "max_tokens": "output tokens per model reply",
    "max_tokens_ceiling": "highest a cut-off reply's limit can be raised to (asks first)",
    "shell_timeout_s": "seconds before a shell command is killed",
    "http_timeout_s": "seconds for an http_fetch request",
    "http_max_bytes": "largest http_fetch body accepted",
}


def _fmt_cost(cost: float | None) -> str:
    return "unknown" if cost is None else f"${cost:.4f}"


def ask_on_terminal(read: Callable[[str], str] = input) -> Approver:
    """Asks y/N on the terminal. EOF or anything but y/yes means no."""

    def approve(req: ExtensionRequest) -> bool:
        what = ("Token budget reached" if req.limit == "token_budget"
                else "The model's reply was cut off at its output limit")
        calls = f"{req.steps} model call" + ("" if req.steps == 1 else "s")
        print(f"\n{what}. So far: {calls}, {req.tokens_used:,} tokens, "
              f"cost {_fmt_cost(req.cost_usd)}.", file=sys.stderr)
        prompt = (f"Raise the {req.limit} from {req.current:,} to {req.proposed:,} "
                  "and continue? [y/N] ")
        try:
            answer = read(prompt)
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    return approve


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("run", help="run the agent on one task",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("task")
    p.add_argument("--workdir", type=Path, default=Path("."),
                   help="the only directory the file and shell tools can touch")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--allow", action="append", default=[], metavar="PROGRAM",
                   help="add a program to the shell allow-list (repeatable)")
    p.add_argument("--no-default-allow", action="store_true",
                   help=f"start from an empty allow-list instead of {', '.join(DEFAULT_ALLOW)}")
    p.add_argument("--no-ask", action="store_true",
                   help="never ask to extend a limit; stop instead")
    defaults = Limits()
    for f in fields(Limits):
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                       type=type(getattr(defaults, f.name)),
                       default=getattr(defaults, f.name), help=LIMIT_HELP[f.name])


def _cmd_run(args: argparse.Namespace) -> int:
    limits = Limits(**{f.name: getattr(args, f.name) for f in fields(Limits)})
    allow = ([] if args.no_default_allow else list(DEFAULT_ALLOW)) + args.allow
    for prog in sorted(set(args.allow) - set(DEFAULT_POLICIES)):
        note = (" It is an interpreter: allowing it lets the model run any code."
                if prog.lower() in INTERPRETERS else "")
        print(f"warning: {prog!r} has no argument rules; only the path check and timeout "
              f"apply.{note}", file=sys.stderr)
    try:
        model = AnthropicModel.from_env()
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    workdir = args.workdir.resolve()
    tools: list[Tool] = [*file_tools(workdir),
                         shell_tool(workdir, allow, timeout_s=limits.shell_timeout_s),
                         http_fetch_tool(limits.http_timeout_s, limits.http_max_bytes)]
    approve = decline if args.no_ask or not sys.stdin.isatty() else ask_on_terminal()
    run_id = new_run_id()
    with Tracer(args.runs_dir / f"{run_id}.jsonl", run_id=run_id) as tracer:
        result = run_agent(model=model, tracer=tracer, system=WORKER.system, task=args.task,
                           tools=tools, limits=limits, approve=approve, name=WORKER.name)
    print(result.text)
    print(f"\n[{result.status}] stop_reason={result.stop_reason} steps={result.steps} "
          f"tokens={result.input_tokens + result.output_tokens:,} "
          f"cost={_fmt_cost(result.cost_usd)} trace={tracer.path}", file=sys.stderr)
    return 0 if result.status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 0
