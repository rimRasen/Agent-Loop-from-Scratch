"""The agent loop: call the model, run the tools it asks for, feed the results back, repeat
until a stop condition. No prompts live here; the caller passes the system prompt in
(prompts belong in agents.py, next to the agent they belong to).

Stop conditions, checked in this order (DECISIONS.md, "Run limits" and "Run outcome"):

    before each model call   max_steps     model calls used up
                             token_budget  billed tokens >= budget, extension declined
    after each model call    refusal       every model in the fallback chain declined
                             max_tokens    reply cut off, a bigger limit declined or impossible
                             final_answer  end_turn / stop_sequence: the only `ok` outcome
                             unexpected_stop_reason  anything we don't handle (pause_turn...)
    after the tool results   tool_errors   max_tool_errors failures in a row
    any time                 exception     a bug or API error; recorded, then re-raised

Two limits can be extended mid-run if `approve` says yes (it is asked with the tokens and cost
so far). Without an approver every request is declined, so tests and CI never block.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from agent_loop.context import truncate
from agent_loop.model import (
    DEFAULT_MAX_TOKENS,
    MAX_NON_STREAMING_TOKENS,
    Block,
    Message,
    Model,
    ModelResponse,
    ToolCall,
)
from agent_loop.tools import Tool, ToolError, ToolOutput
from agent_loop.trace import Status, Tracer

StopReason = Literal["final_answer", "max_steps", "token_budget", "tool_errors", "refusal",
                     "max_tokens", "unexpected_stop_reason", "exception"]


@dataclass(frozen=True)
class Limits:
    """Every bound on one run. Each field is a flag on `agent run` (see README)."""

    max_steps: int = 20                 # model calls per run, including retries
    token_budget: int = 200_000         # billed input + output tokens over the run
    max_tool_errors: int = 3            # tool failures in a row; a success resets the count
    tool_output_chars: int = 10_000     # tool result size before it enters context
    max_tokens: int = DEFAULT_MAX_TOKENS             # output tokens per reply
    max_tokens_ceiling: int = MAX_NON_STREAMING_TOKENS  # highest a reply limit can be raised to
    shell_timeout_s: float = 30.0
    http_timeout_s: float = 10.0
    http_max_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_tokens > self.max_tokens_ceiling:
            raise ValueError("max_tokens must not exceed max_tokens_ceiling")


@dataclass(frozen=True)
class ExtensionRequest:
    """The loop asking whether to spend more. `proposed` replaces `current` if approved."""

    limit: Literal["token_budget", "max_tokens"]
    current: int
    proposed: int
    tokens_used: int
    cost_usd: float | None
    steps: int


Approver = Callable[[ExtensionRequest], bool]


def decline(_request: ExtensionRequest) -> bool:
    return False


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: Status
    stop_reason: StopReason
    text: str                 # the final answer, or the last partial text on any other stop
    messages: list[Message]   # full history, valid to send back to the API
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None    # None if any call's cost was unknown


class _Run:
    """State of one run. Split out of `run_agent` so each step reads as one method."""

    def __init__(self, model: Model, tracer: Tracer, system: str, task: str,
                 tools: Sequence[Tool], limits: Limits, approve: Approver) -> None:
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        self.model, self.tracer, self.system = model, tracer, system
        self.tools = {t.name: t for t in tools}
        self.specs = [t.spec for t in tools]
        self.limits, self.approve = limits, approve
        self.messages: list[Message] = [{"role": "user", "content": task}]
        self.steps = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost: float | None = 0.0
        self.budget = limits.token_budget
        self.max_tokens = limits.max_tokens
        self.errors_in_a_row = 0
        self.last_text = ""
        self.extensions: list[dict[str, object]] = []

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens

    # --- the loop -----------------------------------------------------------------------

    def loop(self) -> StopReason:
        while True:
            if self.steps >= self.limits.max_steps:
                return "max_steps"
            if self.tokens_used >= self.budget:
                if not self.ask("token_budget", self.budget,
                                self.budget + self.limits.token_budget):
                    return "token_budget"
            resp = self.call_model()
            self.last_text = resp.text()

            if resp.stop_reason == "refusal":
                return "refusal"
            if resp.stop_reason == "max_tokens":
                # The cut-off reply is not added to history: current models reject a partial
                # assistant turn ("prefill"), and a cut-off tool_use can't be run safely.
                # So the only way on is to resend the same request with a bigger limit.
                bigger = min(self.max_tokens * 2, self.limits.max_tokens_ceiling)
                if bigger > self.max_tokens and self.ask("max_tokens", self.max_tokens, bigger):
                    self.max_tokens = bigger  # stays raised for the rest of the run
                    continue
                return "max_tokens"
            if resp.stop_reason in ("end_turn", "stop_sequence"):
                self.messages.append(resp.as_message())
                return "final_answer"
            if resp.stop_reason == "tool_use" and resp.tool_calls():
                self.messages.append(resp.as_message())
                # Every result from one turn goes back in a single user message.
                results = [self.run_tool(call) for call in resp.tool_calls()]
                self.messages.append({"role": "user", "content": results})
                if self.errors_in_a_row >= self.limits.max_tool_errors:
                    return "tool_errors"
                continue
            return "unexpected_stop_reason"

    def ask(self, limit: Literal["token_budget", "max_tokens"], current: int,
            proposed: int) -> bool:
        approved = self.approve(ExtensionRequest(limit, current, proposed, self.tokens_used,
                                                 self.cost, self.steps))
        self.extensions.append({"limit": limit, "from": current, "to": proposed,
                                "approved": approved, "after_step": self.steps})
        if approved and limit == "token_budget":
            self.budget = proposed
        return approved

    def call_model(self) -> ModelResponse:
        self.steps += 1
        with self.tracer.span("model", self.model.model_id) as span:
            span.set("step", self.steps)
            span.set("max_tokens", self.max_tokens)
            resp = self.model.complete(system=self.system, messages=self.messages,
                                       tools=self.specs, max_tokens=self.max_tokens)
            cost = resp.cost_usd()
            span.record_usage(resp.input_tokens, resp.output_tokens, cost)
            for key, value in resp.trace_attrs().items():
                span.set(key, value)
            span.set("stop_reason", resp.stop_reason)
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cost = None if self.cost is None or cost is None else self.cost + cost
        return resp

    def run_tool(self, call: ToolCall) -> Block:
        with self.tracer.span("tool", call.name) as span:
            span.set("tool_use_id", call.id)
            tool = self.tools.get(call.name)
            try:
                if tool is None:
                    raise ToolError(f"unknown tool {call.name!r}; available: "
                                    f"{', '.join(self.tools) or 'none'}")
                out = tool.run(call.input)
            except ToolError as exc:  # expected failure: back to the model, run continues
                text, is_error = str(exc) or type(exc).__name__, True
                span.fail(text)
                self.errors_in_a_row += 1
            else:  # any other exception propagates: the span records it, the run ends
                if isinstance(out, str):
                    out = ToolOutput(out)
                for key, value in out.attrs.items():
                    span.set(key, value)
                text, is_error = out.text or "(no output)", False
                self.errors_in_a_row = 0
            cut = truncate(text, self.limits.tool_output_chars)
            span.set("truncated", cut.truncated)
            span.set("original_chars", cut.original_chars)
        block: Block = {"type": "tool_result", "tool_use_id": call.id, "content": cut.text}
        if is_error:
            block["is_error"] = True
        return block


def run_agent(*, model: Model, tracer: Tracer, system: str, task: str,
              tools: Sequence[Tool] = (), limits: Limits | None = None,
              approve: Approver = decline, name: str = "agent") -> RunResult:
    """Run one agent to a stop condition. Every exit is recorded in the `run` span."""
    limits = limits or Limits()
    run = _Run(model, tracer, system, task, tools, limits, approve)
    with tracer.span("run", name) as span:
        span.set("limits", asdict(limits))

        def record(reason: StopReason) -> None:
            span.set("stop_reason", reason)
            span.set("steps", run.steps)
            span.set("extensions", run.extensions)
            span.record_usage(run.input_tokens, run.output_tokens, run.cost)

        try:
            reason = run.loop()
        except BaseException:  # recorded, then re-raised: nothing is hidden
            record("exception")
            raise
        record(reason)
        if reason != "final_answer":
            span.fail(f"stopped: {reason}")
    return RunResult(run_id=tracer.run_id, status="ok" if reason == "final_answer" else "error",
                     stop_reason=reason, text=run.last_text, messages=run.messages,
                     steps=run.steps, input_tokens=run.input_tokens,
                     output_tokens=run.output_tokens, cost_usd=run.cost)
