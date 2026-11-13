"""Try the "ask before spending more" prompts by hand, offline, at no cost.

    uv run python -m tests.manual_approver

A scripted FakeModel (no network) first returns a cut-off reply, so you are asked to raise
max_tokens; then two large tool turns push the run over its token budget, so you are asked to
raise the budget. Answer y or n and watch how the run ends. The trace goes to runs/.
Not collected by pytest (the file name doesn't start with test_).
"""

from __future__ import annotations

from pathlib import Path

from agent_loop.cli import ask_on_terminal
from agent_loop.loop import Limits, run_agent
from agent_loop.tools import Tool
from agent_loop.trace import Tracer, new_run_id
from tests.fakes import FakeModel, reply, text, text_reply, tool_reply, tool_use


def main() -> None:
    priced = {"model": "claude-opus-5-5"}  # so the prompt can show a cost
    model = FakeModel([
        reply(text("This reply was cut o"), stop_reason="max_tokens", input_tokens=2_000,
              output_tokens=8_000, **priced),
        tool_reply(tool_use("noop", {}), input_tokens=120_000, output_tokens=500, **priced),
        tool_reply(tool_use("noop", {}), input_tokens=130_000, output_tokens=500, **priced),
        text_reply("Finished after the budget was raised.", input_tokens=140_000,
                   output_tokens=300, **priced),
    ])
    noop = Tool({"name": "noop", "description": "does nothing",
                 "input_schema": {"type": "object", "properties": {}}}, lambda _: "ok")
    run_id = "manual-" + new_run_id()
    with Tracer(Path("runs") / f"{run_id}.jsonl", run_id=run_id) as tracer:
        result = run_agent(model=model, tracer=tracer, system="(manual test)",
                           task="(manual test)", tools=[noop], limits=Limits(),
                           approve=ask_on_terminal())
    print(f"\nstatus={result.status} stop_reason={result.stop_reason} steps={result.steps} "
          f"tokens={result.input_tokens + result.output_tokens:,}")
    print(f"text: {result.text!r}")
    print(f"trace: {tracer.path}")


if __name__ == "__main__":
    main()
