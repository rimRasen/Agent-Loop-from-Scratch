import copy
from typing import Any

import pytest

from agent_loop.model import (
    FALLBACK_BETA,
    AnthropicModel,
    Attempt,
    FallbackHop,
    MalformedResponseError,
    Model,
    ModelConfigError,
    ToolCall,
    parse_response,
)
from tests.fakes import FakeModel, text_reply, tool_reply, tool_use

OPUS = "claude-opus-5-5"
THINKING = {"type": "thinking", "thinking": "", "signature": "EqQBCkgIARABGAIi...opaque=="}
FUTURE_BLOCK = {"type": "some_block_from_2027", "payload": {"nested": [1, 2, 3]}, "x": None}


def body(*blocks: dict[str, Any], stop_reason: str = "tool_use", model: str = OPUS,
         usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": list(blocks), "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 1200, "output_tokens": 80,
                           "cache_read_input_tokens": 0},
    }


def parse(b: object) -> Any:
    return parse_response(b, requested_model=OPUS, request_id="req_1")


# --- parse_response: content -----------------------------------------------------------------

def test_every_block_is_kept_unchanged_including_unknown_types() -> None:
    blocks = [THINKING, {"type": "text", "text": "Let me look."}, FUTURE_BLOCK,
              {"type": "tool_use", "id": "toolu_9", "name": "shell", "input": {"cmd": "ls"}}]
    original = copy.deepcopy(blocks)
    resp = parse(body(*blocks))
    assert resp.content == original  # same blocks, same order, same fields (signature included)
    assert resp.as_message() == {"role": "assistant", "content": original}
    assert (resp.input_tokens, resp.output_tokens) == (1200, 80)
    assert (resp.stop_reason, resp.model, resp.request_id) == ("tool_use", OPUS, "req_1")


def test_text_and_tool_calls_are_read_from_the_blocks() -> None:
    resp = parse(body(
        {"type": "text", "text": "a"}, THINKING, {"type": "text", "text": "b"},
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"}},
        {"type": "tool_use", "id": "t2", "name": "shell", "input": {}},
    ))
    assert resp.text() == "ab"
    assert resp.tool_calls() == [ToolCall("t1", "read", {"path": "x"}),
                                 ToolCall("t2", "shell", {})]


OK_USAGE = {"input_tokens": 1, "output_tokens": 1}


@pytest.mark.parametrize("bad", [
    [],                                                                  # not an object
    {"model": OPUS, "content": "hi", "usage": OK_USAGE},                 # content not a list
    {"model": OPUS, "content": [{"text": "no type"}], "usage": OK_USAGE},
    {"model": OPUS, "content": [], "usage": {"input_tokens": 1}},        # missing output_tokens
    {"model": OPUS, "content": [], "usage": {"input_tokens": True, "output_tokens": 1}},
    {"content": [], "usage": OK_USAGE},                                  # missing model
    {"model": OPUS, "content": [{"type": "tool_use", "id": "t", "name": "x"}], "usage": OK_USAGE},
    {"model": OPUS, "content": [{"type": "fallback", "from": {}, "to": {"model": "m"}}],
     "usage": OK_USAGE},
    {"model": OPUS, "content": [], "usage": {**OK_USAGE, "iterations": []}},
    {"model": OPUS, "content": [], "usage": {**OK_USAGE, "iterations": [{"type": "message"}]}},
])
def test_malformed_bodies_raise_instead_of_guessing(bad: object) -> None:
    with pytest.raises(MalformedResponseError):
        parse(bad)


# --- parse_response: fallback and cost --------------------------------------------------------

def test_plain_call_has_one_attempt_and_no_fallback() -> None:
    resp = parse(body(stop_reason="end_turn"))
    assert resp.attempts == (Attempt("message", OPUS, 1200, 80),)
    assert not resp.fell_back and resp.fallback_hops() == []
    assert resp.cost_usd() == pytest.approx((1200 * 4 + 80 * 20) / 1e6)
    assert resp.trace_attrs() == {"requested_model": OPUS, "served_model": OPUS,
                                  "fallback": False}


FALLBACK_BLOCK = {"type": "fallback", "from": {"model": OPUS}, "to": {"model": "claude-opus-5"},
                  "trigger": {"type": "refusal", "category": "cyber"}}


def fallback_body() -> dict[str, Any]:
    # Opus 5.5 declined after 30 output tokens; Opus 5 answered. Top-level usage covers only
    # the answering attempt, so it must NOT be what we report.
    return body(
        {"type": "text", "text": "partial from opus 5.5"}, FALLBACK_BLOCK,
        {"type": "text", "text": "answer from opus 5"},
        stop_reason="end_turn", model="claude-opus-5",
        usage={"input_tokens": 1000, "output_tokens": 200, "iterations": [
            {"type": "message", "model": OPUS, "input_tokens": 1000, "output_tokens": 30},
            {"type": "fallback_message", "model": "claude-opus-5",
             "input_tokens": 1000, "output_tokens": 200},
        ]},
    )


def test_fallback_is_visible_and_every_attempt_is_counted_and_priced_at_its_own_rate() -> None:
    resp = parse(fallback_body())
    assert resp.fell_back
    assert (resp.requested_model, resp.model) == (OPUS, "claude-opus-5")
    assert resp.fallback_hops() == [FallbackHop(OPUS, "claude-opus-5", "cyber")]
    assert (resp.input_tokens, resp.output_tokens) == (2000, 230)  # both attempts, not top-level
    assert resp.cost_usd() == pytest.approx(
        (1000 * 4 + 30 * 20) / 1e6       # declined attempt at Opus 5.5 rates
        + (1000 * 5 + 200 * 25) / 1e6    # answering attempt at Opus 5 rates
    )
    assert resp.trace_attrs() == {
        "requested_model": OPUS, "served_model": "claude-opus-5", "fallback": True,
        "fallback_hops": [{"from": OPUS, "to": "claude-opus-5", "category": "cyber"}],
    }
    assert FALLBACK_BLOCK in resp.as_message()["content"]  # marker kept in history too


def test_sticky_fallback_without_a_marker_block_is_still_detected() -> None:
    # After a conversation falls back, later calls go straight to the fallback model:
    # no fallback block in content, only the fallback_message iteration.
    resp = parse(body(
        {"type": "text", "text": "hi"}, stop_reason="end_turn", model="claude-opus-5",
        usage={"input_tokens": 50, "output_tokens": 5, "iterations": [
            {"type": "fallback_message", "model": "claude-opus-5",
             "input_tokens": 50, "output_tokens": 5}]},
    ))
    assert resp.fell_back and resp.fallback_hops() == []
    assert resp.trace_attrs()["served_model"] == "claude-opus-5"
    assert resp.trace_attrs()["fallback"] is True


def test_attempt_without_a_model_is_priced_only_when_there_was_no_fallback() -> None:
    no_fallback = parse(body(usage={"input_tokens": 10, "output_tokens": 1, "iterations": [
        {"type": "message", "input_tokens": 10, "output_tokens": 1}]}))
    assert no_fallback.cost_usd() == pytest.approx((10 * 4 + 1 * 20) / 1e6)

    ambiguous = parse(body(FALLBACK_BLOCK, model="claude-opus-5", usage={
        "input_tokens": 10, "output_tokens": 1, "iterations": [
            {"type": "message", "input_tokens": 10, "output_tokens": 1},
            {"type": "fallback_message", "model": "claude-opus-5",
             "input_tokens": 10, "output_tokens": 1}]}))
    assert ambiguous.cost_usd() is None


def test_cost_is_none_if_any_attempt_model_is_unpriced() -> None:
    resp = parse(body(model="claude-opus-9", usage={"input_tokens": 1, "output_tokens": 1,
        "iterations": [
            {"type": "message", "model": OPUS, "input_tokens": 1, "output_tokens": 1},
            {"type": "fallback_message", "model": "claude-opus-9",
             "input_tokens": 1, "output_tokens": 1}]}))
    assert resp.cost_usd() is None


# --- AnthropicModel request building (stub client, no network) ---------------------------

class _Raw:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.request_id = "req_stub"

    def json(self) -> object:
        return self._payload


class _StubClient:
    """Mimics client.beta.messages.with_raw_response.create and records its kwargs."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.kwargs: dict[str, Any] = {}
        outer = self

        class _WithRaw:
            def create(self, **kwargs: Any) -> _Raw:
                outer.kwargs = kwargs
                return _Raw(payload)

        class _Messages:
            with_raw_response = _WithRaw()

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def make_model(payload: dict[str, Any], **kw: Any) -> tuple[AnthropicModel, _StubClient]:
    stub = _StubClient(payload)
    return AnthropicModel(OPUS, client=stub, **kw), stub  # type: ignore[arg-type]


def test_request_turns_on_default_fallback_and_omits_empty_system_and_tools() -> None:
    model, stub = make_model(body({"type": "text", "text": "hi"}, stop_reason="end_turn"))
    msgs = [{"role": "user", "content": "hello"}]
    resp = model.complete(system="", messages=msgs, tools=[], max_tokens=123)
    assert stub.kwargs == {"model": OPUS, "max_tokens": 123, "messages": msgs,
                           "betas": [FALLBACK_BETA], "fallbacks": "default"}
    assert resp.text() == "hi" and resp.request_id == "req_stub"
    assert resp.requested_model == OPUS


def test_fallback_can_be_switched_off() -> None:
    model, stub = make_model(body(stop_reason="end_turn"), fallback=False)
    model.complete(system="", messages=[{"role": "user", "content": "x"}], tools=[])
    assert "fallbacks" not in stub.kwargs and "betas" not in stub.kwargs


def test_request_includes_system_and_tools_when_given() -> None:
    model, stub = make_model(body(stop_reason="end_turn"))
    tool = {"name": "shell", "description": "run", "input_schema": {"type": "object"}}
    model.complete(system="be brief", messages=[{"role": "user", "content": "x"}], tools=[tool])
    assert stub.kwargs["system"] == "be brief"
    assert stub.kwargs["tools"] == [tool]


def test_history_with_unknown_blocks_is_sent_back_unchanged() -> None:
    model, stub = make_model(body(stop_reason="end_turn"))
    first = parse(body(THINKING, FUTURE_BLOCK, tool_use("shell", {"cmd": "ls"})))
    history = [{"role": "user", "content": "go"}, first.as_message(),
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                             "content": "ok"}]}]
    model.complete(system="", messages=history, tools=[])
    assert stub.kwargs["messages"][1]["content"] == [THINKING, FUTURE_BLOCK,
                                                     tool_use("shell", {"cmd": "ls"})]


def test_from_env_requires_agent_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(ModelConfigError, match="AGENT_MODEL"):
        AnthropicModel.from_env()
    monkeypatch.setenv("AGENT_MODEL", "   ")
    with pytest.raises(ModelConfigError):
        AnthropicModel.from_env()


def test_from_env_reads_the_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")  # client construction only
    assert AnthropicModel.from_env().model_id == "claude-sonnet-5-5"


# --- FakeModel -----------------------------------------------------------------------------

def test_fake_model_satisfies_the_model_protocol() -> None:
    m: Model = FakeModel([])
    assert m.model_id == "fake-model"


def test_fake_model_replays_script_records_calls_and_fails_when_exhausted() -> None:
    fake = FakeModel([tool_reply(tool_use("shell", {"cmd": "ls"})), RuntimeError("api down"),
                      text_reply("done")])
    history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    assert fake.complete(system="s", messages=history, tools=[]).stop_reason == "tool_use"
    history.append({"role": "user", "content": "mutated after the call"})
    with pytest.raises(RuntimeError, match="api down"):
        fake.complete(system="s", messages=history, tools=[])
    assert fake.complete(system="s", messages=history, tools=[]).text() == "done"
    assert fake.calls[0].messages == [{"role": "user", "content": "go"}]  # deep-copied
    with pytest.raises(AssertionError, match="script exhausted"):
        fake.complete(system="s", messages=history, tools=[])
