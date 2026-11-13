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

---

## 2026-11-13 — Run limits: a `Limits` dataclass with defaults, overridable from the CLI

**Options.** A: a frozen `Limits` dataclass passed to the loop, with defaults; each field has a CLI
flag. B: environment variables. C: a TOML config file.

**Chosen.** A.

**Defaults.** `max_steps` 20 (model calls per run); `token_budget` 200,000 billed input + output
tokens summed over the run; `max_tool_errors` 3 consecutive (a successful tool call resets the
count); `tool_output_chars` 10,000; `shell_timeout_s` 30; `http_timeout_s` 10; `http_max_bytes`
1,000,000. The values used are recorded in the `run` span's attrs.

**Why.** Tests set limits directly with no env patching; a run never picks up a hidden setting;
the trace shows which limits a run had. `token_budget` is a spending cap on billed tokens, not a
measure of how full the context is (fallback calls bill ~2x the input; context fullness is
milestone 3's job). The budget is checked after each model call, so a run can overshoot by at
most one call.

**User access.** Every field is a flag on `agent run` (`--max-steps`, `--token-budget`, ...),
listed in `agent run --help` and in the README with what each one counts.

**Revisit if.** We need per-agent limits for delegation (milestone 4), or the same overrides are
typed on every run (then a config file earns its place).

---

## 2026-11-13 — Run outcome: `ok` only on final answer; a refusal stops the run

**Options for refusal.** A: stop immediately, `stop_reason: "refusal"`. B: treat it as a final
answer. C: retry once with a note added.

**Chosen.** A, plus a general rule: a run's `status` is `ok` only when it ends with
`final_answer`. Every other stop (`max_steps`, `token_budget`, `tool_errors`, `refusal`) is
`status: "error"` with that `stop_reason`. Partial text from the refused reply is kept in the
result.

**Why.** With server-side fallback on, a refusal means every model in the chain declined, so a
retry repeats the same request. A caller (including a delegating agent in milestone 4) must be able
to tell success from every other ending by one field.

**Revisit if.** We turn fallback off (then a client-side retry on another model becomes an option).

---

## 2026-11-13 — `http_fetch` uses the standard library

**Options.** A: `urllib.request`. B: `httpx2`, declared as a direct dependency (it is already
installed as an SDK dependency, but importing it undeclared would break if the SDK drops it).

**Chosen.** A.

**How.** Only `http`/`https` URLs. The body is read in chunks up to `http_max_bytes + 1` to detect
an over-size response without downloading it all. `urllib`'s timeout applies per socket operation,
so the read loop also checks an overall deadline. Tests run against a local `http.server` on
127.0.0.1 (no external network).

**Not done.** Private/loopback addresses are not blocked in Phase 0, so the model can reach
services on the local network.

**Revisit if.** We need HTTP/2, connection pooling, or a mock transport becomes worth a dependency;
or the agent runs anywhere a local-network request would be harmful (then block private ranges).

---

## 2026-11-13 — Tool interface: `Tool(spec, run)`, expected failures raise `ToolError`

**Options.** A: a `Tool` dataclass with `spec: ToolSpec` and `run(input) -> str`; expected failures
raise `ToolError`. B: tools return `ToolResult(text, is_error)` and never raise on purpose.

**Chosen.** A.

**How.** The loop catches only `ToolError`: it becomes a `tool_result` with `is_error: true`, is
recorded in the tool span (`status: "error"`), and counts toward `max_tool_errors`. Any other
exception is a bug in the tool: it is recorded in the span and re-raised, ending the run.

**Why.** Separates "the model asked for something that failed" (a missing file, a timeout, a
command not on the allow-list) from "our code is broken", which is the line CLAUDE.md draws
against catching broad exceptions.

**Revisit if.** A tool needs to return structured or non-text results (images, multiple blocks).

---

## 2026-11-13 — Asking to spend more: token budget and cut-off replies

**Context.** You wanted a run that hits a token limit to pause, ask whether to pay for more, and
continue if you say yes or stop if you say no.

**Options for what triggers the question.** (a) only a cut-off reply (`stop_reason:
"max_tokens"`); (b) only the run's `token_budget`; (c) both.
**Options for how you are asked.** A: the loop calls an `approve` function mid-run; the CLI asks
y/N on the terminal. B: the run stops and is resumed later (`agent resume`), which needs the
conversation saved to disk.
**Options for headroom on a cut-off reply.** (i) lower the per-reply default to 8,000 so a retry
can go to 16,000 without streaming; (ii) keep 16,000 and add streaming to `model.py`.

**Chosen.** (c), A, (i).

**How.**
- Token budget: checked before each model call. Approval raises it by another `token_budget`.
- Cut-off reply: the reply is dropped (current models reject a partial assistant turn, and a
  cut-off `tool_use` can't be run), and the same request is re-sent with `max_tokens` doubled, up
  to `max_tokens_ceiling` (16,000, `model.MAX_NON_STREAMING_TOKENS`). It stays raised for the
  rest of the run. At the ceiling the run stops without asking. The dropped call is still billed,
  counted toward the budget and `max_steps`, and traced.
- The approver gets the limit, current and proposed values, tokens and cost so far, and steps.
  The default approver declines, so tests and CI never block; the CLI uses the terminal only
  when stdin is a terminal and `--no-ask` isn't set. EOF counts as no.
- Every question is recorded in the run span: `attrs.extensions` =
  `[{limit, from, to, approved, after_step}]`.
- `model.DEFAULT_MAX_TOKENS` changed from 16,000 to 8,000.

**Cost.** A cut-off reply is paid for twice (input re-sent, output thrown away). Time spent waiting for your answer
is included in the run span's duration, so it looks like latency in the trace.

**Revisit if.** Cut-off replies are common at 8,000 (then streaming, option ii), or runs need to
survive the process ending (option B).

---

## 2026-11-13 — Shell tool: argv list, small allow-list with per-program rules

**Context.** You chose an argument list with no shell, and asked that dangerous arguments be
handled for every default program, keeping the list small where we don't know how.

**Chosen.** Default allow-list `git` and `rg` only; the user can add programs with `--allow` or
drop the defaults with `--no-default-allow`.

**Checks for every program.** The name is bare (no path) and on the list; it resolves on PATH to a
file outside the working directory (the model can't plant its own `git.exe`); `.bat`/`.cmd` are
refused (cmd.exe re-parses their arguments); every argument and `--opt=value` value that resolves
outside the working directory is refused; the child's environment has `ANTHROPIC_*` (the API
key), `GIT_*` and `RIPGREP_CONFIG_PATH` removed; the timeout kills the whole process tree
(`taskkill /T /F` on Windows, the process group elsewhere).

**git rules.** Only `status diff log show ls-files grep`, as the first argument (so no global
options such as `-c`, `-C`, `--git-dir`). Refused: `--output`, `--ext-diff`, `--textconv`,
`--show-signature`, `-O`/`--open-files-in-pager`, `--no-index`, any `%G` format code. Injected:
`--no-pager -c core.fsmonitor=false -c core.pager=cat -c log.showSignature=false`, plus
`--no-ext-diff --no-textconv` on diff/log/show. A test shows a hostile `core.fsmonitor` no
longer runs. The file tools refuse paths inside `.git`, so the model can't write that config.

**rg rules.** Refused: `--pre`, `--pre-glob`, `--hostname-bin`, `-z`/`--search-zip`, and bundled
short options (`-iz`), because parsing the bundles correctly is fragile.
Checked against ripgrep 15.2.0 (2026-11-13): these are still the only options that run a program.
Tests run real `rg` searches when it is installed, including one through a junction pointing
outside the working directory (refused).

**Programs added by the user** get the general checks only. Their tool spans carry
`program_policy: "unvetted"`; the CLI prints a warning, stronger for interpreters.

**Known gaps.** A `.git/config` that already defines a filter driver (`filter.<name>.clean`) can
still run on `git status`/`diff`; we can't neutralise drivers we don't know the names of. Output
is held in memory until the command ends, so a command printing gigabytes within the timeout uses
that much memory. The path check can refuse harmless text that looks like an absolute path (a
regex like `\bword` outside `-e`).

**Revisit if.** We add a program to the defaults (it needs its own rules first), or need real
isolation (a container or a restricted user), which an allow-list is not.

---

## 2026-11-13 — New `attrs` keys for milestone 2

**Context.** The trace format allows adding `attrs` keys without a format change, but keys must
never be repurposed, so the set in use is recorded here.

**Added.**
- `run`: `stop_reason` (one of `final_answer`, `max_steps`, `token_budget`, `tool_errors`,
  `refusal`, `max_tokens`, `unexpected_stop_reason`, `exception`), `limits` (every `Limits`
  field), `steps` (model calls), `extensions` (`[{limit, from, to, approved, after_step}]`).
- `model`: `step` (1-based call number in the run), `max_tokens` (the limit sent),
  `stop_reason` (the API's, e.g. `end_turn`), alongside the existing fallback keys.
- `tool`: `tool_use_id`, `truncated`, `original_chars` on every tool span. `shell` adds
  `program`, `exit_code`, `program_policy` (`rules` or `unvetted`). `http_fetch` adds
  `http_status`, `content_type`, `bytes`, and `final_url` after a redirect.

**Note.** A run's `stop_reason` (in the `run` span) and a model call's `stop_reason` (in a `model`
span) are different things with the same key name on different kinds: the run's is our outcome,
the model's is the API's.

**Revisit if.** `agent trace` (milestone 5) or Phase 3/4 needs any of these as top-level fields.
