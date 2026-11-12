"""Thin wrapper over the Anthropic SDK. The ONLY module that talks to the API.

The rest of the code sees `Model` and `ModelResponse`, never SDK types (DECISIONS.md,
"Model interface"). Conversation history is a list of plain dicts in the API's wire format.

Responses are read from the raw HTTP JSON instead of the SDK's parsed objects. That makes
"every block comes back unchanged" true by construction: thinking blocks keep their signatures
byte for byte, and block types this SDK version doesn't know about are kept as-is.

Server-side fallback is on (DECISIONS.md, "Server-side model fallback"): if the requested model
declines for policy reasons, the API re-runs the request on another model inside the same call.
`ModelResponse` exposes what happened so the trace can show every model change.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import anthropic

# One message in API wire format, e.g. {"role": "user", "content": [...]}.
Message = dict[str, Any]
# One content block in API wire format, e.g. {"type": "text", "text": "..."}.
Block = dict[str, Any]
# One tool definition in API wire format: {"name", "description", "input_schema"}.
ToolSpec = dict[str, Any]

DEFAULT_MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ModelConfigError(RuntimeError):
    """The model can't be constructed, e.g. AGENT_MODEL is not set."""


class MalformedResponseError(RuntimeError):
    """The API returned 2xx but the body isn't a message we can use."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Attempt:
    """One billed sampling pass inside a single API call (an entry of `usage.iterations`).

    A call with no fallback has one attempt. A fallback call has one for the model that declined
    and one for the model that answered (type "fallback_message"). `model` can be None on plain
    "message" entries; see `ModelResponse.cost_usd` for how that is handled.
    """

    type: str
    model: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FallbackHop:
    """One hand-over inside a call: `from_model` declined and `to_model` took over."""

    from_model: str
    to_model: str
    category: str | None


@dataclass(frozen=True)
class ModelResponse:
    """One model reply.

    `content` is every block the API returned, in order, unmodified, including types we don't
    handle (and `fallback` marker blocks). Append it to history with `as_message()`.

    `model` is the model that produced the reply; `requested_model` is the one we asked for.
    Token counts are summed over every attempt, so they are what the call was billed for.
    """

    content: list[Block]
    stop_reason: str | None
    model: str
    requested_model: str
    attempts: tuple[Attempt, ...]
    request_id: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)

    @property
    def fell_back(self) -> bool:
        """True if a fallback model served this reply.

        Read from `usage.iterations`, not from `fallback` blocks: once a conversation has fallen
        back, later calls are routed straight to the fallback model ("sticky" routing) and carry
        no block, only the iteration entry.
        """
        return any(a.type == "fallback_message" for a in self.attempts)

    def fallback_hops(self) -> list[FallbackHop]:
        hops = []
        for b in self.content:
            if b.get("type") == "fallback":
                trigger = b.get("trigger")
                category = trigger.get("category") if isinstance(trigger, dict) else None
                hops.append(FallbackHop(from_model=str(b["from"]["model"]),
                                        to_model=str(b["to"]["model"]),
                                        category=category if isinstance(category, str) else None))
        return hops

    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b.get("type") == "text")

    def tool_calls(self) -> list[ToolCall]:
        return [ToolCall(id=b["id"], name=b["name"], input=b["input"])
                for b in self.content if b.get("type") == "tool_use"]

    def as_message(self) -> Message:
        return {"role": "assistant", "content": self.content}

    def cost_usd(self) -> float | None:
        """Sum of each attempt priced at its own model's rate, or None if any part is unknown.

        An attempt with no model is priced as `self.model` only when no fallback happened (then
        there is only one model it can be). In a fallback call it could be either model, so the
        cost is None rather than a guess.
        """
        ambiguous = self.fell_back or bool(self.fallback_hops())
        total = 0.0
        for a in self.attempts:
            model = a.model or (None if ambiguous else self.model)
            part = cost_usd(model, a.input_tokens, a.output_tokens) if model else None
            if part is None:
                return None
            total += part
        return total

    def trace_attrs(self) -> dict[str, object]:
        """Facts for the model span's `attrs`. `fallback` is always present, true or false."""
        attrs: dict[str, object] = {
            "requested_model": self.requested_model,
            "served_model": self.model,
            "fallback": self.fell_back,
        }
        hops = self.fallback_hops()
        if hops:
            attrs["fallback_hops"] = [
                {"from": h.from_model, "to": h.to_model, "category": h.category} for h in hops
            ]
        return attrs


class Model(Protocol):
    """What the loop needs from a model. `AnthropicModel` and the tests' `FakeModel` provide it."""

    model_id: str

    def complete(self, *, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec], max_tokens: int = DEFAULT_MAX_TOKENS) -> ModelResponse:
        ...


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_response(body: object, requested_model: str,
                   request_id: str | None = None) -> ModelResponse:
    """Turn the raw JSON body of a Messages API response into a ModelResponse.

    Strict about the fields the loop relies on, permissive about everything else.
    """
    if not isinstance(body, dict):
        raise MalformedResponseError(f"expected a JSON object, got {type(body).__name__}")
    content = body.get("content")
    if not isinstance(content, list) or not all(
        isinstance(b, dict) and isinstance(b.get("type"), str) for b in content
    ):
        raise MalformedResponseError("content must be a list of blocks that each have a 'type'")
    for b in content:
        if b["type"] == "tool_use" and not (
            isinstance(b.get("id"), str) and isinstance(b.get("name"), str)
            and isinstance(b.get("input"), dict)
        ):
            raise MalformedResponseError(f"tool_use block missing id/name/input: {b!r}")
        if b["type"] == "fallback" and not (
            isinstance(b.get("from"), dict) and isinstance(b["from"].get("model"), str)
            and isinstance(b.get("to"), dict) and isinstance(b["to"].get("model"), str)
        ):
            raise MalformedResponseError(f"fallback block missing from/to model: {b!r}")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise MalformedResponseError("model missing")
    usage = body.get("usage")
    if not isinstance(usage, dict) or not all(
        _is_int(usage.get(k)) for k in ("input_tokens", "output_tokens")
    ):
        raise MalformedResponseError("usage.input_tokens/output_tokens missing")
    iterations = usage.get("iterations")
    if iterations is None:  # no breakdown: the whole call was one attempt by `model`
        attempts: tuple[Attempt, ...] = (
            Attempt("message", model, usage["input_tokens"], usage["output_tokens"]),
        )
    else:
        if not isinstance(iterations, list) or not iterations or not all(
            isinstance(it, dict) and isinstance(it.get("type"), str)
            and _is_int(it.get("input_tokens")) and _is_int(it.get("output_tokens"))
            and (it.get("model") is None or isinstance(it.get("model"), str))
            for it in iterations
        ):
            raise MalformedResponseError("usage.iterations entries need type and token counts")
        attempts = tuple(Attempt(it["type"], it.get("model"), it["input_tokens"],
                                 it["output_tokens"]) for it in iterations)
    stop_reason = body.get("stop_reason")
    return ModelResponse(
        content=cast(list[Block], content),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        model=model,
        requested_model=requested_model,
        attempts=attempts,
        request_id=request_id,
    )


class AnthropicModel:
    """Calls the Messages API (beta endpoint, for server-side fallback).

    API errors (anthropic.APIStatusError, APIConnectionError, ...) are not caught here: the
    caller's span records them. The SDK retries 408/409/429/5xx and connection errors twice
    with backoff before raising, so one model span can include those retries.

    `fallbacks="default"` lets the API pick the fallback model by refusal category. On a model
    with no fallback support the request runs on that model alone and a refusal stays a refusal.

    The API key is read by the SDK from the environment. It is never passed or stored here.
    """

    def __init__(self, model_id: str, client: anthropic.Anthropic | None = None,
                 fallback: bool = True) -> None:
        self.model_id = model_id
        self.fallback = fallback
        self._client = client or anthropic.Anthropic()

    @classmethod
    def from_env(cls) -> AnthropicModel:
        model_id = os.environ.get("AGENT_MODEL", "").strip()
        if not model_id:
            raise ModelConfigError("set AGENT_MODEL to the model ID to use")
        return cls(model_id)

    def complete(self, *, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec], max_tokens: int = DEFAULT_MAX_TOKENS) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = list(tools)
        if self.fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        raw = self._client.beta.messages.with_raw_response.create(**kwargs)
        return parse_response(raw.json(), requested_model=self.model_id,
                              request_id=raw.request_id)


# --- Pricing -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    """USD per million tokens. `long_*` rates apply to the whole request once its input exceeds
    `long_threshold` tokens (some models price long prompts higher)."""

    input: float
    output: float
    long_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None


# Anthropic first-party prices, checked 2026-11-12. This table only prices calls; the model to
# call comes from AGENT_MODEL. The older models are here because they are fallback targets.
# Cache reads/writes are not priced (we don't use caching yet).
PRICES: dict[str, Price] = {
    "claude-fable-5-2": Price(input=8.00, output=40.00),
    "claude-opus-5-5": Price(input=4.00, output=20.00),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00),
    "claude-haiku-5-5": Price(input=0.10, output=0.50,
                              long_threshold=100_000, long_input=0.50, long_output=2.50),
    "claude-opus-5": Price(input=5.00, output=25.00),
    "claude-opus-4-8": Price(input=5.00, output=25.00),
    "claude-sonnet-5": Price(input=2.00, output=10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one attempt, or None if the model isn't in the table. Never a guess."""
    price = PRICES.get(model)
    if price is None:
        return None
    rate_in, rate_out = price.input, price.output
    if price.long_threshold is not None and input_tokens > price.long_threshold:
        assert price.long_input is not None and price.long_output is not None
        rate_in, rate_out = price.long_input, price.long_output
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
