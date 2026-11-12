# agent-loop — CLAUDE.md

An agent loop written from scratch, with no framework, where every run can be opened as a trace.
Phase 0 of my roadmap. Later phases reuse this loop and its trace format, so keep both small and stable.

## How we work

You write the code. I make the decisions, and I need to understand everything that ships.

- **Decisions are mine.** Before anything costly to reverse (architecture, interfaces, data formats,
  dependencies, concurrency model, failure handling), give me 2 or 3 options with trade-offs and your
  recommendation, then wait for my choice. Make small decisions yourself and list them at the end of
  the change.
- **Log every choice** in `DECISIONS.md`: date, options, what I chose, why, and what would make us
  revisit it.
- **Explain what you write.** After each change, give a short explanation of how it works and why,
  focused on what isn't obvious from reading the code. For the core components below, walk me through
  the control flow and the failure cases.
- **Show how it could be wrong.** Every component ships with tests that would catch it being wrong.
  Tell me what you didn't test and why.
- **Check my understanding.** At the end of each milestone, ask me three questions about how the core
  works or why we chose it. If I get one wrong, explain it before we move on.

**Core components for this project:** the loop (`loop.py`), context management (`context.py`), delegation between agents (`agents.py`), the trace format (`trace.py`).

- Don't use agent frameworks (LangChain, LangGraph, CrewAI, etc.). The point of this repo is understanding what they hide.

## Stack and commands

- Python 3.12, `uv` for env and deps, `pytest`, `ruff`, `mypy --strict` on `src/`.
- Anthropic Python SDK. Model ID comes from the `AGENT_MODEL` env var; never hardcode a model string.
- API key from `ANTHROPIC_API_KEY`; never write it to disk, logs or traces.

```
uv sync                       # install
uv run pytest                 # all tests (no network)
uv run pytest -m live         # tests that call the real API (costs money, run on purpose)
uv run ruff check . && uv run mypy src
uv run agent run "<task>"     # one run, writes runs/<run_id>.jsonl
uv run agent trace <run_id>   # print the run as a tree with tokens, latency, cost
```

## Layout

```
src/agent_loop/
  loop.py        # the loop: model call -> tool dispatch -> feed result back -> stop condition
  context.py     # truncation, summarisation, per-run token budget
  agents.py      # agent definitions; one agent delegating a subtask to another
  tools/         # file_rw.py, shell.py (with timeout), http_fetch.py
  model.py       # thin wrapper over the SDK; the ONLY place that talks to the API
  trace.py       # span writer (JSONL) and reader
  cli.py
tests/
  fakes.py       # FakeModel that returns scripted responses, for offline tests
runs/            # trace output, gitignored
DECISIONS.md
```

## Design rules

- **Stop conditions are explicit**: final answer, max steps, token budget exceeded, or tool error limit.
  Every run ends with a recorded reason. No loop may run unbounded.
- **Tools are sandboxed**: file tools restricted to a working directory passed at startup; shell has a
  timeout and an allow-list; HTTP fetch has a timeout and a size cap. Tool errors go back to the model as
  results, they don't crash the loop.
- **One span per model call and per tool call, plus one `run` span per agent run.** Fields: `run_id,
  span_id, parent_id, kind, name, start_ms, end_ms, input_tokens, output_tokens, cost_usd, status, error,
  attrs`. Kinds: `run`, `model`, `tool`. `attrs` is an object of kind-specific facts (stop reason,
  truncation, served model, fallbacks); keys are added, never repurposed. Delegated runs are child spans
  of the delegating tool call. This JSONL format is reused in Phases 3 and 4, so changes need a
  `DECISIONS.md` entry.
- Truncate tool output before it enters context and record that it was truncated.
- Every test runs offline against `FakeModel`. Network tests are marked `live`.

## Done when

- [ ] README explains each design decision in the loop (link `DECISIONS.md`)
- [ ] Any run can be printed as a trace showing where the time and tokens went
- [ ] Two-agent delegation works and shows up as nested spans
- [ ] CI runs lint, types and offline tests

## Don't

- Don't add dependencies without asking.
- Don't put prompts inside `loop.py`; prompts live in `agents.py` next to the agent they belong to.
- Don't catch broad exceptions to hide failures; record them in the span.
