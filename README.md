# agent-loop

An agent loop written from scratch in Python, with no agent framework, where every run can be
opened as a trace. The point is to understand what frameworks like LangChain or CrewAI hide: the
model call, tool dispatch, stop conditions, context management and delegation between agents.

Built on the Anthropic Python SDK. Every design choice, with the options considered and what would
make us revisit it, is logged in [DECISIONS.md](DECISIONS.md).

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Model wrapper, trace format, offline `FakeModel` | Done |
| 2 | The loop, stop conditions, sandboxed tools (file, shell, HTTP) | Next |
| 3 | Context management: truncation, summarisation, token budget | Planned |
| 4 | One agent delegating a subtask to another, as nested spans | Planned |
| 5 | `agent trace` viewer, CI | Planned |

The `agent run` and `agent trace` commands below arrive in milestones 2 and 5.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned and installed by uv.

```
uv sync
```

Set these to call the real API (tests don't need them):

- `AGENT_MODEL`: the model ID to use, e.g. `claude-opus-5-5`. Never hardcoded.
- `ANTHROPIC_API_KEY`: read by the SDK from the environment, never written to disk, logs or traces.

## Commands

```
uv run pytest                 # all tests, offline against FakeModel
uv run pytest -m live         # tests that call the real API (costs money)
uv run ruff check . && uv run mypy src
uv run agent run "<task>"     # one run, writes runs/<run_id>.jsonl      (milestone 2)
uv run agent trace <run_id>   # the run as a tree: tokens, latency, cost (milestone 5)
```

## How it works so far

**Model wrapper (`src/agent_loop/model.py`).** The only module that talks to the API. The rest of
the code sees a small `Model` protocol and `ModelResponse` type, never SDK types. Responses are read
from the raw JSON, so every content block goes back into history unchanged, including thinking
blocks and block types newer than the installed SDK. Server-side model fallback is on: if the
model declines a request, the API retries it on another model, and the trace records the requested
model, the model that answered, and each hand-over. Cost is summed per attempt at each model's own
rate, and is `null` when a price is unknown rather than a guess.

**Trace format (`src/agent_loop/trace.py`).** One JSON line per span (a `run`, a `model` call or a
`tool` call), written and flushed when the span ends:

```
run_id, span_id, parent_id, kind, name, start_ms, end_ms,
input_tokens, output_tokens, cost_usd, status, error, attrs
```

Spans nest through `parent_id`, so a delegated agent's run appears under the tool call that started
it. If code inside a span raises, the span is still written with `status: "error"` and the
exception is re-raised. `attrs` holds kind-specific facts such as a run's stop reason or whether a
tool's output was truncated. Later phases of the project reuse this format, so changes to it are
logged in `DECISIONS.md`.

**Offline tests (`tests/fakes.py`).** `FakeModel` replays scripted responses, so every test runs
without the network. Tests that call the real API are marked `live` and skipped by default.

## Layout

```
src/agent_loop/
  model.py     # SDK wrapper, fallback tracking, price table
  trace.py     # span writer and reader
  cli.py       # `agent` command (placeholder)
tests/
  fakes.py     # FakeModel and reply helpers
DECISIONS.md   # every design decision, with options and trade-offs
CLAUDE.md      # working rules for the AI assistant on this repo
```
