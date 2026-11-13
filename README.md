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
| 2 | The loop, stop conditions, sandboxed tools (file, shell, HTTP) | Done |
| 3 | Context management: summarisation, context-size budget | Next |
| 4 | One agent delegating a subtask to another, as nested spans | Planned |
| 5 | `agent trace` viewer, CI | Planned |

The `agent trace` command below arrives in milestone 5.

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
uv run agent run "<task>"     # one run, writes runs/<run_id>.jsonl
uv run agent trace <run_id>   # the run as a tree: tokens, latency, cost (milestone 5)
```

## Running the agent and changing its limits

```
uv run agent run "Summarise what changed in the last 3 commits" --workdir path/to/repo
uv run agent run --help       # every setting with its default
```

The answer goes to stdout; a summary line (status, stop reason, steps, tokens, cost, trace file)
goes to stderr. The exit code is 0 only if the run ended with a final answer.

Every limit is a flag. Defaults, and what each one counts:

| Flag | Default | What it limits |
|---|---|---|
| `--max-steps` | 20 | model calls per run, retries included |
| `--token-budget` | 200000 | billed input + output tokens per run (a spending cap) |
| `--max-tool-errors` | 3 | tool failures **in a row**; any success resets the count |
| `--tool-output-chars` | 10000 | characters of each tool result kept in context (head and tail) |
| `--max-tokens` | 8000 | output tokens per model reply |
| `--max-tokens-ceiling` | 16000 | the highest a cut-off reply's limit can be raised to |
| `--shell-timeout-s` | 30 | seconds before a shell command and its children are killed |
| `--http-timeout-s` | 10 | seconds for an `http_fetch` request |
| `--http-max-bytes` | 1000000 | largest `http_fetch` body accepted |

**When a limit is hit, you are asked before more money is spent.** Two limits can be extended
mid-run:

- *Token budget reached*: you see the tokens and cost so far and are asked whether to raise the
  budget by another `--token-budget`. Answer `y` to continue, anything else to stop.
- *A reply was cut off at `--max-tokens`*: the cut-off reply can't be resumed (current models
  don't accept a partial reply back), so continuing means re-sending the same request with
  double the limit, up to `--max-tokens-ceiling`. The cut-off reply is still billed. Once raised,
  the limit stays raised for the rest of the run.

If you decline, the run stops with that reason. With `--no-ask`, or when stdin is not a terminal
(scripts, CI), every extension is declined automatically. Every question and answer is recorded
in the `run` span's `attrs.extensions`.

**The working directory** (`--workdir`, default `.`) is the only directory the file and shell
tools can touch. Paths that resolve outside it, and anything inside a `.git` directory, are
refused.

**The shell allow-list.** By default the shell tool can run only `git` and `rg`, each with its
own argument rules:

- `git`: only `status`, `diff`, `log`, `show`, `ls-files`, `grep`, with no global options.
  Options that run another program or write files (`--ext-diff`, `--textconv`, `--output`,
  `-O`, `--show-signature`, `%G` format codes, `--no-index`) are refused, and settings that a
  hostile `.git/config` could use to run a program (`core.fsmonitor`, the pager) are overridden.
- `rg`: `--pre`, `--pre-glob`, `--hostname-bin` and `-z` (all run other programs) are refused, as
  are bundled short options like `-iz`.

To change the list:

```
uv run agent run "..." --allow python          # add a program (repeatable)
uv run agent run "..." --no-default-allow      # start from an empty list
uv run agent run "..." --no-default-allow --allow git
```

A program you add has no argument rules: it gets only the general protections (no shell, the
path check, the timeout, an environment with `ANTHROPIC_*` and `GIT_*` variables removed), and
its tool spans are marked `program_policy: "unvetted"`. A warning is printed at startup. For an
interpreter (`python`, `uv`, `node`, `powershell`, ...) allowing it means allowing any code.

## Testing by hand

Some tests depend on what's installed or allowed on your machine; pytest reports them as skipped
(`uv run pytest -rs` shows why).

**`git` and `rg` tests** run only when the program is on PATH. If you just installed one, open a
new terminal so the updated PATH is picked up, then:

```
uv run pytest tests/test_shell.py -v -rs
```

**Symlink test (Windows).** Creating a symlink needs a privilege that normal accounts don't have,
so `test_symlink_pointing_outside_is_rejected` is skipped. (Directory junctions escape the same
way without any privilege, and their tests always run.) To run it, turn on Developer Mode, which
lets your account create symlinks:

```
start ms-settings:developers        # Settings > System > For developers > Developer Mode: On
```

or, from an **administrator** PowerShell, set the same switch in the registry:

```
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v AllowDevelopmentWithoutDevLicense /d 1
```

Then open a new terminal and run:

```
uv run pytest tests/test_file_rw.py -k symlink -v -rs
```

Developer Mode also lets other programs you run create symlinks; turn it off again afterwards if
you don't want that.

**The "ask before spending more" prompts, offline and free.** A scripted fake model triggers
both questions (a cut-off reply, then the token budget):

```
uv run python -m tests.manual_approver
```

Answer `y`/`n` in different orders and check the final `status` and `stop_reason`.

**The same prompts against the real API** (costs a few cents; needs `AGENT_MODEL` and
`ANTHROPIC_API_KEY`). A tiny budget makes the budget question appear after the first call, and a
tiny `--max-tokens` makes a reply get cut off:

```
uv run agent run "List the files here, then summarise README.md" --token-budget 1000
uv run agent run "Write a 400-word story about a lighthouse" --max-tokens 200 --max-tokens-ceiling 400
uv run agent run "List the files here" --token-budget 1000 --no-ask   # stops without asking
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

**The loop (`src/agent_loop/loop.py`).** Calls the model, runs the tools it asks for, sends all
results from one turn back in a single message, and repeats. Every run ends with one recorded
`stop_reason`: `final_answer` (the only `ok` outcome), `max_steps`, `token_budget`,
`tool_errors`, `refusal`, `max_tokens`, `unexpected_stop_reason`, or `exception`. A tool's expected
failure (`ToolError`) goes back to the model as an error result; any other exception is a bug,
so it is recorded in the spans and re-raised.

**Tools (`src/agent_loop/tools/`).** `read_file`, `write_file`, `list_dir` (sandboxed to the
working directory), `shell` (allow-list, argument rules, timeout that kills the whole process
tree), and `http_fetch` (http/https only, text only, size cap, overall deadline). Output is
truncated to `--tool-output-chars` before it enters context, and each tool span records
`truncated` and `original_chars`.

**Offline tests (`tests/fakes.py`).** `FakeModel` replays scripted responses, so every test runs
without the network. Tests that call the real API are marked `live` and skipped by default.

## Layout

```
src/agent_loop/
  loop.py      # the loop and its stop conditions; Limits
  context.py   # tool-output truncation (more in milestone 3)
  agents.py    # agent definitions and their prompts
  tools/       # file_rw.py, shell.py, http_fetch.py
  model.py     # SDK wrapper, fallback tracking, price table
  trace.py     # span writer and reader
  cli.py       # `agent run`
tests/
  fakes.py     # FakeModel and reply helpers
DECISIONS.md   # every design decision, with options and trade-offs
CLAUDE.md      # working rules for the AI assistant on this repo
```
