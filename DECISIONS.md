# Decisions

Each entry: date, options, what was chosen, why, and what would make us revisit it.

---

## 2026-11-12 — Restart on the CLAUDE.md spec; replace the Ollama prototype

**Context.** The repo held an early prototype (`agent/tools.py`, `scripts/hello*.py`) that called a
local Ollama model (`qwen3:4b`) through `requests`, used OpenAI-style tool schemas, and had no tests,
types or traces. CLAUDE.md specifies the Anthropic SDK, a `src/agent_loop/` layout and `uv`.

**Options.**
1. Keep Ollama behind `model.py` and follow the rest of the spec.
2. Follow CLAUDE.md exactly: Anthropic SDK, rewrite from scratch.
3. Support both providers behind one interface.

**Chosen.** 2. The prototype was an earlier version of the learning project and could be replaced.

**Why.** It matches the spec the later phases are built on, and one provider keeps `model.py` thin.
Every test runs offline against `FakeModel`, so we don't need a local model for development.

**Consequences.** The prototype was deleted. It was never committed, so it is not in git history. The read-only codebase tools (`list_dir`, `search_text`, `calculator`)
are dropped in favour of the spec's `file_rw`, `shell` and `http_fetch`. Python is pinned to 3.12.
Dependencies: `anthropic` at runtime; `pytest`, `ruff`, `mypy` for dev.

**Revisit if.** API cost during manual runs becomes a problem, or a later phase needs local models.
In either case `model.py` is the only file that would change.

---

## 2026-11-12 — Concurrency model: synchronous

**Options.** A: synchronous; parallel tool calls run one after another, delegation is a nested call.
B: `asyncio`; concurrent tools and sub-agents, async through every layer.

**Chosen.** A.

**Why.** The control flow reads top to bottom, tests need no event loop, and nested delegation maps
directly onto nested spans (the tracer can keep the current parent on a simple stack).

**Revisit if.** Tool latency dominates runs that request several independent tools at once, or a
later phase needs sub-agents running side by side. The change is confined to `loop.py`, `model.py`
and the tracer's parent tracking.

---

## 2026-11-12 — Spans are written when they end

**Options.** A: one JSONL line per span, written and flushed when the span ends. B: a start line and
an end line per span, merged by the reader. C: buffer in memory, write at end of run.

**Chosen.** A.

**Why.** One line equals one span with exactly the CLAUDE.md fields, so the format stays one record
type. Spans are closed in a `finally`, so a Python exception still produces a line with
`status: "error"`. Lines appear in end order (children before parents); the reader rebuilds the tree
from `parent_id`, not from line order.

**Cost.** A hard kill (`kill -9`, power loss) loses the spans still open at that moment, and can
leave a torn final line. The reader skips an unterminated final line and reports corruption anywhere
else.

**Revisit if.** We need live progress on long runs, or hard kills become a real failure mode.

---

## 2026-11-12 — `cost_usd` comes from a price table in `model.py`

**Options.** A: a price table in `model.py` keyed by model ID. B: prices from env vars.

**Chosen.** A.

**Why.** No per-model setup. A model missing from the table gets `cost_usd: null`, never a guess.
The table names models but never chooses one; the model to call still comes only from `AGENT_MODEL`.

**Cost.** Prices go stale and must be updated by hand. Prompt-cache reads and writes are not priced
yet (we don't use caching).

**Revisit if.** We turn on prompt caching, or prices change often enough that the table drifts.

---

## 2026-11-12 — Model interface: our own small types, history as wire-format dicts

**Options.** A: `model.py` returns our own `ModelResponse`; history is plain dicts in the API's wire
format; the loop never imports `anthropic`. B: pass the SDK's `Message` objects through the loop.

**Chosen.** A, with the explicit requirement that unknown block types are kept, not dropped.

**How.** `ModelResponse.content` is every block the API returned, unmodified and in order. It is
read from the raw HTTP JSON (`with_raw_response` + `.json()`), not from the SDK's parsed objects, so
thinking-block signatures survive byte for byte and block types newer than the installed SDK pass
through untouched. `text()` and `tool_calls()` are read-only views over those blocks. The assistant
turn goes back into history via `as_message()`, which reuses `content` as is.

**Why.** The loop and `FakeModel` stay independent of SDK types, and scripted test responses are
plain dicts. Passing blocks back unchanged is required on models that check thinking blocks.

**Cost.** We validate the response shape ourselves (`parse_response`) instead of relying on the
SDK's models; only the fields the loop uses are checked strictly.

**Revisit if.** We need streaming (the raw-JSON path is for non-streaming calls), or the SDK's
typed objects start carrying something we need that the JSON body doesn't.

---

## 2026-11-12 — Trace format: `run` span kind and an `attrs` field

**Context.** The CLAUDE.md span fields had nowhere to record a run's stop reason or the fact that a
tool's output was truncated, both of which CLAUDE.md requires.

**Options.** A: add a `run` kind plus one `attrs` object for kind-specific facts. B: add a `run`
kind plus dedicated `stop_reason` and `truncated` fields. C: no new fields; overload `name` and add
`event` spans.

**Chosen.** A.

**Format change.** Every span line now has a 13th field, `attrs` (an object, `{}` when empty).
Kinds: `run` (one per agent run; root of that run's tree), `model`, `tool`. Known keys so far:
`stop_reason` on `run` spans; `truncated` and `original_chars` on `tool` spans. Keys are added over
time, never repurposed. Values are checked to be JSON-serialisable when set, not when written.

**Why.** One field covers all current needs, and later phases can add keys without another format
change.

**Cost.** `attrs` keys aren't type-checked; a typo in a key name won't be caught by mypy. Tests on
the loop must assert the exact keys.

**Revisit if.** The set of keys stabilises and we want them typed, or Phase 3/4 needs a key to be
queryable as a top-level field.

---

## 2026-11-12 — Server-side model fallback: on, and every model change is traced

**Context.** Current models can decline a request (`stop_reason: "refusal"`) when a safety
classifier fires, sometimes on benign work. Without a fallback, the run just stops.

**Options.** A: server-side `fallbacks: "default"`; the API re-runs a declined request on a model
it picks by refusal category, inside the same call. B: no fallback; a refusal ends the run.
C: our own client-side retry on a model we name.

**Chosen.** A, with the condition that we always know when the model changed.

**How.** `AnthropicModel` calls the beta endpoint with `betas=["server-side-fallback-2026-07-01"]`
and `fallbacks="default"` (switchable with `fallback=False`). On a model with no fallback support
the request runs on that model alone, so it is safe to always send. Every model span gets:
- `attrs.requested_model` and `attrs.served_model` (top-level `model` in the response);
- `attrs.fallback`: true/false, always present. Read from a `fallback_message` entry in
  `usage.iterations`, because after a conversation falls back, later calls go straight to the
  fallback model ("sticky", ~1 hour) and carry no marker block;
- `attrs.fallback_hops`: `[{from, to, category}]` from the `fallback` blocks in `content`, when a
  hand-over happened in this call.

**Tokens and cost.** Top-level `usage` covers only the attempt that answered. We sum
`usage.iterations` instead, so a span's tokens and cost include the declined attempt. Each attempt
is priced at its own model's rate; fallback targets (`claude-opus-5`, `claude-opus-4-8`,
`claude-sonnet-5`) are in the price table. An attempt with no model in a fallback call makes the
cost `null` rather than a guess.

**Cost of the choice.** A run can switch model mid-way and stay switched for later calls. The
fallback model can't read the original model's thinking blocks (the API drops them). Billed
input tokens on a fallback call are roughly double the context size, so the context budget must
not use them as "how full is the context".

**Revisit if.** We want a run to stop on refusal instead of switching model (set `fallback=False`),
or we move to a platform without server-side fallback (Bedrock, Vertex, Foundry).
