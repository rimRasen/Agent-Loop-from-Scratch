"""Offline stand-ins for the model. Every test runs against these, never the network."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from agent_loop.model import (
    DEFAULT_MAX_TOKENS,
    Attempt,
    Block,
    Message,
    ModelResponse,
    ToolSpec,
)


@dataclass(frozen=True)
class RecordedCall:
    """What the loop sent on one call, deep-copied so later mutation can't rewrite history."""

    system: str
    messages: list[Message]
    tools: list[ToolSpec]
    max_tokens: int


class FakeModel:
    """Returns scripted responses in order. A scripted exception is raised instead of returned.

    Running past the end of the script is a test failure, not a silent default: it means the
    loop made more calls than the test expected.
    """

    def __init__(self, script: Iterable[ModelResponse | BaseException],
                 model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self._script = list(script)
        self.calls: list[RecordedCall] = []

    def complete(self, *, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec], max_tokens: int = DEFAULT_MAX_TOKENS) -> ModelResponse:
        self.calls.append(RecordedCall(system, copy.deepcopy(list(messages)),
                                       copy.deepcopy(list(tools)), max_tokens))
        if len(self.calls) > len(self._script):
            raise AssertionError(
                f"FakeModel script exhausted: call {len(self.calls)} but only "
                f"{len(self._script)} scripted responses"
            )
        item = self._script[len(self.calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item


def reply(*blocks: Block, stop_reason: str = "end_turn", input_tokens: int = 10,
          output_tokens: int = 5, model: str = "fake-model") -> ModelResponse:
    """A reply served by `model` in one attempt (no fallback)."""
    return ModelResponse(content=list(blocks), stop_reason=stop_reason, model=model,
                         requested_model=model,
                         attempts=(Attempt("message", model, input_tokens, output_tokens),))


def text(t: str) -> Block:
    return {"type": "text", "text": t}


def tool_use(name: str, input: dict[str, Any], id: str = "toolu_1") -> Block:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def text_reply(t: str, **kw: Any) -> ModelResponse:
    return reply(text(t), **kw)


def tool_reply(*calls: Block, **kw: Any) -> ModelResponse:
    return reply(*calls, stop_reason="tool_use", **kw)
