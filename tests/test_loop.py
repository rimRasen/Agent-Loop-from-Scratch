"""The loop against FakeModel: every stop condition, tool dispatch, extensions, and the trace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_loop.loop import ExtensionRequest, Limits, RunResult, run_agent
from agent_loop.model import ModelResponse
from agent_loop.tools import Tool, ToolError, ToolOutput
from agent_loop.trace import Span, Tracer, read_spans
from tests.fakes import FakeModel, reply, text, text_reply, tool_reply, tool_use
from tests.test_trace import fake_clock


def make_tool(name: str, fn: Callable[[dict[str, Any]], str | ToolOutput]) -> Tool:
    return Tool({"name": name, "description": name,
                 "input_schema": {"type": "object", "properties": {}}}, fn)


def echo(inp: dict[str, Any]) -> str:
    return f"echo:{inp.get('x', '')}"


def failing(_inp: dict[str, Any]) -> str:
    raise ToolError("nope")


def run(tmp_path: Path, script: list[ModelResponse | BaseException], *,
        tools: list[Tool] | None = None, limits: Limits | None = None,
        approve: Callable[[ExtensionRequest], bool] | None = None,
        ) -> tuple[RunResult, FakeModel, list[Span]]:
    model = FakeModel(script)
    with Tracer(tmp_path / "t.jsonl", run_id="r1", clock=fake_clock()) as tr:
        kw: dict[str, Any] = {}
        if approve is not None:
            kw["approve"] = approve
        result = run_agent(model=model, tracer=tr, system="sys", task="do it",
                           tools=tools if tools is not None else [make_tool("echo", echo)],
                           limits=limits, **kw)
    return result, model, read_spans(tmp_path / "t.jsonl")


def run_span(spans: list[Span]) -> Span:
    [s] = [s for s in spans if s.kind == "run"]
    return s


# --- final answer and the shape of the trace ----------------------------------------------

def test_final_answer_is_the_only_ok_outcome(tmp_path: Path) -> None:
    result, model, spans = run(tmp_path, [text_reply("done", input_tokens=7, output_tokens=3)])
    assert (result.status, result.stop_reason, result.text) == ("ok", "final_answer", "done")
    assert result.steps == 1 and (result.input_tokens, result.output_tokens) == (7, 3)
    assert model.calls[0].system == "sys"
    assert model.calls[0].messages == [{"role": "user", "content": "do it"}]
    assert result.messages[-1] == {"role": "assistant", "content": [text("done")]}

    rs = run_span(spans)
    assert rs.status == "ok" and rs.error is None and rs.parent_id is None
    assert rs.attrs["stop_reason"] == "final_answer" and rs.attrs["steps"] == 1
    assert rs.attrs["limits"] == {**Limits().__dict__}
    assert (rs.input_tokens, rs.output_tokens) == (7, 3)
    [ms] = [s for s in spans if s.kind == "model"]
    assert ms.parent_id == rs.span_id and ms.name == "fake-model"
    assert ms.attrs["stop_reason"] == "end_turn" and ms.attrs["fallback"] is False
    assert ms.attrs["max_tokens"] == 8_000 and ms.attrs["step"] == 1


def test_stop_sequence_is_a_final_answer(tmp_path: Path) -> None:
    result, _, _ = run(tmp_path, [reply(text("x"), stop_reason="stop_sequence")])
    assert result.stop_reason == "final_answer"


def test_cost_is_summed_and_none_if_any_call_is_unpriced(tmp_path: Path) -> None:
    priced = dict(model="claude-opus-5-5", input_tokens=50_000, output_tokens=0)
    result, _, spans = run(tmp_path, [tool_reply(tool_use("echo", {}), **priced),
                                      text_reply("ok", **priced)])
    assert result.cost_usd == pytest.approx(0.4)  # 2 x 50k input at $4/M
    assert run_span(spans).cost_usd == pytest.approx(0.4)

    result, _, _ = run(tmp_path / "b", [tool_reply(tool_use("echo", {}), **priced),
                                        text_reply("ok")])  # fake-model: not in the table
    assert result.cost_usd is None


# --- tool dispatch ------------------------------------------------------------------------

def test_tool_result_goes_back_with_matching_id(tmp_path: Path) -> None:
    result, model, spans = run(tmp_path, [tool_reply(tool_use("echo", {"x": 1}, id="t9")),
                                          text_reply("done")])
    assert result.stop_reason == "final_answer"
    second = model.calls[1].messages
    assert second[1]["role"] == "assistant"
    assert second[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t9", "content": "echo:1"}]}
    [ts] = [s for s in spans if s.kind == "tool"]
    assert ts.name == "echo" and ts.status == "ok"
    assert ts.parent_id == run_span(spans).span_id
    assert ts.input_tokens is None and ts.cost_usd is None
    assert ts.attrs == {"tool_use_id": "t9", "truncated": False, "original_chars": 6}


def test_parallel_calls_return_in_one_user_message_in_order(tmp_path: Path) -> None:
    _, model, _ = run(tmp_path, [tool_reply(tool_use("echo", {"x": "a"}, id="1"),
                                            tool_use("echo", {"x": "b"}, id="2")),
                                 text_reply("done")])
    msgs = model.calls[1].messages
    assert len(msgs) == 3
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["1", "2"]
    assert [b["content"] for b in msgs[2]["content"]] == ["echo:a", "echo:b"]


def test_tool_error_goes_back_to_the_model_and_the_run_continues(tmp_path: Path) -> None:
    result, model, spans = run(tmp_path, [tool_reply(tool_use("bad", {})), text_reply("ok")],
                               tools=[make_tool("bad", failing)])
    assert result.status == "ok"
    [block] = model.calls[1].messages[2]["content"]
    assert block["is_error"] is True and block["content"] == "nope"
    [ts] = [s for s in spans if s.kind == "tool"]
    assert ts.status == "error" and ts.error == "nope"


def test_unknown_tool_is_a_tool_error(tmp_path: Path) -> None:
    _, model, _ = run(tmp_path, [tool_reply(tool_use("nosuch", {})), text_reply("ok")])
    [block] = model.calls[1].messages[2]["content"]
    assert block["is_error"] is True and "unknown tool 'nosuch'" in block["content"]


def test_tool_output_is_truncated_and_recorded(tmp_path: Path) -> None:
    long = make_tool("long", lambda _: "x" * 50)
    _, model, spans = run(tmp_path, [tool_reply(tool_use("long", {})), text_reply("ok")],
                          tools=[long], limits=Limits(tool_output_chars=10))
    content = model.calls[1].messages[2]["content"][0]["content"]
    assert content.startswith("x" * 5) and content.endswith("x" * 5)
    assert "40 of 50 characters omitted" in content
    [ts] = [s for s in spans if s.kind == "tool"]
    assert ts.attrs["truncated"] is True and ts.attrs["original_chars"] == 50


def test_tool_output_attrs_reach_the_span(tmp_path: Path) -> None:
    t = make_tool("t", lambda _: ToolOutput("", {"exit_code": 3}))
    _, model, spans = run(tmp_path, [tool_reply(tool_use("t", {})), text_reply("ok")],
                          tools=[t])
    assert model.calls[1].messages[2]["content"][0]["content"] == "(no output)"
    [ts] = [s for s in spans if s.kind == "tool"]
    assert ts.attrs["exit_code"] == 3


def test_a_bug_in_a_tool_ends_the_run_and_is_recorded(tmp_path: Path) -> None:
    def buggy(_: dict[str, Any]) -> str:
        raise KeyError("oops")

    with pytest.raises(KeyError):
        run(tmp_path, [tool_reply(tool_use("b", {}))], tools=[make_tool("b", buggy)])
    spans = read_spans(tmp_path / "t.jsonl")
    [ts] = [s for s in spans if s.kind == "tool"]
    assert ts.status == "error" and "KeyError" in (ts.error or "")
    rs = run_span(spans)
    assert rs.status == "error" and rs.attrs["stop_reason"] == "exception"


def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, [], tools=[make_tool("a", echo), make_tool("a", echo)])


# --- stop conditions ----------------------------------------------------------------------

def test_tool_error_limit_counts_failures_in_a_row(tmp_path: Path) -> None:
    script: list[ModelResponse | BaseException] = [
        tool_reply(tool_use("bad", {})) for _ in range(3)]
    result, model, spans = run(tmp_path, script, tools=[make_tool("bad", failing)])
    assert (result.status, result.stop_reason) == ("error", "tool_errors")
    assert len(model.calls) == 3
    # history ends with the error results, so it stays valid to send again
    assert result.messages[-1]["role"] == "user"
    assert run_span(spans).error == "stopped: tool_errors"


def test_a_success_resets_the_tool_error_count(tmp_path: Path) -> None:
    tools = [make_tool("bad", failing), make_tool("echo", echo)]
    script: list[ModelResponse | BaseException] = [
        tool_reply(tool_use("bad", {})), tool_reply(tool_use("bad", {})),
        tool_reply(tool_use("echo", {})),
        tool_reply(tool_use("bad", {})), tool_reply(tool_use("bad", {})),
        text_reply("done")]
    result, _, _ = run(tmp_path, script, tools=tools)
    assert result.stop_reason == "final_answer"


def test_errors_within_one_turn_count_toward_the_limit(tmp_path: Path) -> None:
    calls = [tool_use("bad", {}, id=str(i)) for i in range(3)]
    result, model, _ = run(tmp_path, [tool_reply(*calls)], tools=[make_tool("bad", failing)])
    assert result.stop_reason == "tool_errors" and len(model.calls) == 1


def test_max_steps_stops_before_the_next_call(tmp_path: Path) -> None:
    script: list[ModelResponse | BaseException] = [
        tool_reply(tool_use("echo", {})), tool_reply(tool_use("echo", {}))]
    result, model, spans = run(tmp_path, script, limits=Limits(max_steps=2))
    assert (result.status, result.stop_reason, result.steps) == ("error", "max_steps", 2)
    assert len(model.calls) == 2  # a third call would exhaust the script and fail the test
    assert run_span(spans).attrs["stop_reason"] == "max_steps"


def test_token_budget_asks_then_stops_when_declined(tmp_path: Path) -> None:
    asked: list[ExtensionRequest] = []

    def no(req: ExtensionRequest) -> bool:
        asked.append(req)
        return False

    script: list[ModelResponse | BaseException] = [
        tool_reply(tool_use("echo", {})), tool_reply(tool_use("echo", {}))]
    result, model, spans = run(tmp_path, script, limits=Limits(token_budget=30), approve=no)
    assert result.stop_reason == "token_budget" and len(model.calls) == 2
    assert asked == [ExtensionRequest("token_budget", current=30, proposed=60,
                                      tokens_used=30, cost_usd=None, steps=2)]
    assert run_span(spans).attrs["extensions"] == [
        {"limit": "token_budget", "from": 30, "to": 60, "approved": False, "after_step": 2}]


def test_token_budget_extension_continues_the_run(tmp_path: Path) -> None:
    script: list[ModelResponse | BaseException] = [
        tool_reply(tool_use("echo", {})), tool_reply(tool_use("echo", {})), text_reply("ok")]
    result, model, _ = run(tmp_path, script, limits=Limits(token_budget=30),
                           approve=lambda _: True)
    assert result.stop_reason == "final_answer" and len(model.calls) == 3


def test_without_an_approver_extensions_are_declined(tmp_path: Path) -> None:
    result, _, _ = run(tmp_path, [tool_reply(tool_use("echo", {}))],
                       limits=Limits(token_budget=1))
    assert result.stop_reason == "token_budget"


def test_refusal_stops_immediately_and_keeps_the_text(tmp_path: Path) -> None:
    result, model, spans = run(tmp_path, [reply(text("I can't"), stop_reason="refusal")])
    assert (result.status, result.stop_reason, result.text) == ("error", "refusal", "I can't")
    assert len(model.calls) == 1
    assert result.messages == [{"role": "user", "content": "do it"}]
    assert run_span(spans).error == "stopped: refusal"


def test_max_tokens_declined_stops_without_adding_the_cut_off_reply(tmp_path: Path) -> None:
    cut = reply(text("partial"), tool_use("echo", {}), stop_reason="max_tokens")
    result, _, _ = run(tmp_path, [cut])
    assert (result.stop_reason, result.text) == ("max_tokens", "partial")
    assert result.messages == [{"role": "user", "content": "do it"}]


def test_max_tokens_approved_resends_the_same_request_with_a_bigger_limit(
        tmp_path: Path) -> None:
    asked: list[ExtensionRequest] = []

    def yes(req: ExtensionRequest) -> bool:
        asked.append(req)
        return True

    cut = reply(text("part"), stop_reason="max_tokens")
    script: list[ModelResponse | BaseException] = [
        cut, tool_reply(tool_use("echo", {})), text_reply("done")]
    result, model, spans = run(tmp_path, script, approve=yes)
    assert result.stop_reason == "final_answer" and result.steps == 3
    assert [c.max_tokens for c in model.calls] == [8_000, 16_000, 16_000]  # stays raised
    assert model.calls[0].messages == model.calls[1].messages  # the cut-off reply was dropped
    assert [(r.limit, r.current, r.proposed) for r in asked] == [("max_tokens", 8_000, 16_000)]
    # the wasted call is still billed and traced
    assert result.input_tokens == 30
    assert [s.attrs["max_tokens"] for s in spans if s.kind == "model"] == [8_000, 16_000, 16_000]


def test_max_tokens_at_the_ceiling_stops_without_asking(tmp_path: Path) -> None:
    def must_not_ask(_: ExtensionRequest) -> bool:
        raise AssertionError("asked")

    result, _, _ = run(tmp_path, [reply(text("p"), stop_reason="max_tokens")],
                       limits=Limits(max_tokens=16_000), approve=must_not_ask)
    assert result.stop_reason == "max_tokens"


@pytest.mark.parametrize("stop", ["pause_turn", "something_new", None])
def test_unhandled_stop_reasons_end_the_run(tmp_path: Path, stop: str | None) -> None:
    resp = reply(text("x"))
    resp = ModelResponse(content=resp.content, stop_reason=stop, model=resp.model,
                         requested_model=resp.requested_model, attempts=resp.attempts)
    result, _, _ = run(tmp_path, [resp])
    assert (result.status, result.stop_reason) == ("error", "unexpected_stop_reason")


def test_tool_use_stop_without_tool_calls_is_unexpected(tmp_path: Path) -> None:
    result, _, _ = run(tmp_path, [reply(text("hm"), stop_reason="tool_use")])
    assert result.stop_reason == "unexpected_stop_reason"


def test_model_exception_is_recorded_on_both_spans_and_re_raised(tmp_path: Path) -> None:
    with pytest.raises(ConnectionError):
        run(tmp_path, [ConnectionError("down")])
    spans = read_spans(tmp_path / "t.jsonl")
    [ms] = [s for s in spans if s.kind == "model"]
    assert ms.status == "error" and "down" in (ms.error or "")
    rs = run_span(spans)
    assert rs.attrs["stop_reason"] == "exception" and rs.attrs["steps"] == 1


# --- limits -------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"max_steps": 0}, {"token_budget": -1},
                                 {"max_tokens": 20_000}, {"shell_timeout_s": 0.0}])
def test_invalid_limits_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Limits(**bad)
