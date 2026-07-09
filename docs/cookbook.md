# Cookbook

Task-oriented recipes for [pyhar](https://pypi.org/project/pyhar-agents/). Every
snippet is self-contained and runs with **no API key** — they use `ScriptedModel`
so you can paste, run, and see the result immediately.

```bash
pip install pyhar-agents   # distribution name
python -c "import pyhar"    # import name
```

`ScriptedModel([...])` plays back a fixed sequence: a **string** item is a final
text answer; a **`("tool", name, {args})`** tuple is one tool call. Swap it for a
real backend ([`AnthropicModel`, `OpenAIModel`, …](models.md)) when you go live —
nothing else changes.

See also: [Concepts](concepts.md) · [Components](components.md) ·
[Model backends](models.md) · [Adapters, MCP & subagents](adapters-and-mcp.md)

---

## 1. A coding agent, out of the box

**When:** you want a Claude-Code-shaped agent — tool-output budgeting plus staged
compaction — without wiring components by hand. `presets.coding_agent` is that
composition; pass a `check` to also get verify→retry (recipe 4).

```python
from pyhar import ScriptedModel, tool
from pyhar.presets import coding_agent


@tool
def read_file(path: str) -> str:
    """Return the contents of a file (here, a big fake blob)."""
    return "decision: use SQLite\n" + ("some verbose log line\n" * 400) + "TODO: add index"


model = ScriptedModel([
    ("tool", "read_file", {"path": "db.py"}),
    "Done — the store uses SQLite; I added the index. The answer is 42.",
])


def check(state):
    text = (state.result or state.messages[-1].content).lower()
    return ("42" in text, "answer must contain '42'")


harness = coding_agent(model, tools=[read_file], check=check, context_tokens=300)
state = harness.run("Inspect db.py and tell me the answer.")

print("result:  ", state.result)
print("verified:", state.memory.get("_verified"))
print("tool tokens saved:", state.memory.get("_tool_savings", 0))
```

`coding_agent(model, tools=..., check=..., context_tokens=..., tool_output_tokens=...)`
returns a plain `Harness` — inspect or extend `harness` like any other. For the
bare loop with nothing added, use `presets.minimal_react(model, tools=...)`.

---

## 2. Deny destructive tools

**When:** the model has access to dangerous tools and you must guarantee some
never fire. `Permissions(deny=[...])` runs in `before_tool` and blocks the call
*before* it executes; the denial string is returned to the model as the tool
result, so it can recover.

```python
from pyhar import Harness, Permissions, ScriptedModel, tool


@tool
def read_file(path: str) -> str:
    """A safe tool."""
    return "contents of " + path


@tool
def delete_everything(path: str) -> str:
    """A destructive tool we never want called."""
    raise RuntimeError("this should never run — it's denied")


model = ScriptedModel([
    ("tool", "delete_everything", {"path": "/"}),   # blocked
    ("tool", "read_file", {"path": "README.md"}),   # allowed
    "Done — I read README.md and left everything else alone.",
])

harness = Harness(
    model,
    components=[Permissions(deny=["delete_everything"])],
    tools=[read_file, delete_everything],
)
state = harness.run("Clean up the repo.")

print("result:", state.result)
print("denied:", state.memory.get("_denied"))
```

`Permissions` is just a [Component](components.md). Any component's
`before_tool(state, call) -> str | None` can gate a call the same way — return a
string to deny.

---

## 3. Observe every step

**When:** you want a structured event log — for debugging, dashboards, or audit.
`Tracer(sink=...)` emits an event at each lifecycle stage; the sink is any
callable (print, append to a list, ship to your log pipeline).

```python
from pyhar import Harness, ScriptedModel, Tracer, tool


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return "contents of " + path


events = []
model = ScriptedModel([
    ("tool", "read_file", {"path": "README.md"}),
    "Read the README.",
])

harness = Harness(
    model,
    components=[Tracer(sink=events.append)],   # or sink=lambda e: print("trace:", e)
    tools=[read_file],
)
state = harness.run("Summarize the README.")

for e in events:
    print(e)
print("trace also in memory:", len(state.memory.get("_trace", [])), "events")
```

Pair it with recipe 2 for a **safe *and* observable** agent — both are just
components in the same list.

---

## 4. Verify → retry until a check passes

**When:** the model's first answer isn't trustworthy and you have a real check —
run tests, an eval, an LLM judge, an end-to-end drive. `Verifier(check)` runs your
check when the model produces a candidate answer (a turn with no tool calls); on
failure it injects feedback and re-opens the task, up to `max_retries`.

```python
from pyhar import Harness, ScriptedModel, Verifier

# First answer is wrong; after feedback the model corrects itself.
model = ScriptedModel([
    "The answer is 7.",    # fails the check -> retry
    "The answer is 42.",   # passes
])


def check(state):
    text = (state.result or state.messages[-1].content)
    return ("42" in text, "the answer must contain 42")


harness = Harness(model, components=[Verifier(check, max_retries=2)])
state = harness.run("What is the answer?")

print("result:  ", state.result)
print("verified:", state.memory.get("_verified"))
```

`check(state) -> (passed: bool, feedback: str)`. The feedback is fed back to the
model verbatim, so make it actionable. `coding_agent(..., check=check)` (recipe 1)
wires this in for you.

---

## 5. Keep tool output small — and prove it

**When:** a tool returns a wall of text (logs, file dumps) that would blow up your
context. `ToolOutputBudget(max_tokens=...)` truncates the result the model sees
while preserving the full output elsewhere. `bench` lets you A/B it and read the
savings as a number.

```python
from pyhar import Compactor, Harness, ScriptedModel, ToolOutputBudget, bench, tool


@tool
def read_log(path: str) -> str:
    """Return a large log blob."""
    return "decision: use SQLite\n" + ("verbose log line\n" * 500) + "TODO: add index"


def make_config(tuned: bool):
    def factory() -> Harness:
        model = ScriptedModel([
            ("tool", "read_log", {"path": "app.log"}),
            "Analysis complete — the store is SQLite; add the index.",
        ])
        components = []
        if tuned:
            components = [ToolOutputBudget(max_tokens=200), Compactor(target_tokens=1500)]
        return Harness(model, components=components, tools=[read_log])
    return factory


report = bench(
    "read a big log then summarize",
    {
        "baseline": make_config(tuned=False),
        "tuned (budget+compact)": make_config(tuned=True),
    },
    success=lambda s: s.done,
    trials=3,   # run each config 3 times; the report carries means + success rate
)
print(report.table())

base = next(r for r in report.runs if r.name == "baseline")
tuned = next(r for r in report.runs if r.name.startswith("tuned"))
saved = base.input_tokens - tuned.input_tokens
print(f"mean input tokens saved: {saved:.0f}")
```

`bench(task, {name: factory}, success=...)` runs each config from a **fresh
factory** and returns a `BenchReport` (`.table()`, `.runs`). Runs that include a
`ToolOutputBudget` also record `state.memory["_tool_savings"]`.

**New in 0.3.0:** pass `trials=N` to run each config N times. Each `RunReport`
then carries **means** for turns/tokens/cost, a `success_rate` (the table's `ok`
column becomes a percentage and a `trials` column appears), and
`input_tokens_std` / `output_tokens_std` standard deviations — so noisy
real-model comparisons don't hinge on a single lucky run. With a deterministic
`ScriptedModel` the trials are identical; with a live backend they won't be.

---

## 6. Resume across a fresh context

**When:** long-horizon work that outlives a single context window. `StateArtifact`
writes progress/decisions to a store; a brand-new harness reading the same file
reconstructs "where am I" — no shared memory required.

```python
import tempfile
from pathlib import Path

from pyhar import Harness, ScriptedModel, StateArtifact
from pyhar.components.state_artifact import FileStore

progress = str(Path(tempfile.mkdtemp()) / "progress.json")

# run 1: make a decision, persist it to disk
Harness(
    ScriptedModel(["decision: use event sourcing for the ledger; append-only log"]),
    components=[StateArtifact(FileStore(progress))],
).run("How should we design the ledger?")

# run 2: a fresh harness over the SAME file reconstructs the context
r2 = Harness(
    ScriptedModel(["Continuing from the restored plan — event sourcing it is."]),
    components=[StateArtifact(FileStore(progress))],
).run("Pick up where we left off.")

restored = [m for m in r2.messages if m.meta.get("state_artifact") == "restored"]
print("restored from disk:", restored[0].content if restored else "(nothing)")
print("result:", r2.result)
```

`FileStore(path)` is the on-disk backend; swap in any `MemoryStore` for tests.
Combine with `Memory(core=..., ...)` to also pin a core block and recall past
notes.

---

## 7. Delegate to an isolated subagent

**When:** a subtask deserves its **own** context window (research, a noisy
sub-search) and you only want the *result* back in the parent. `subagent_tool`
exposes a fresh sub-harness as a callable tool — the parent model invokes it with
a task string; the subagent runs in isolation; only its final answer returns.

```python
from pyhar import Harness, ScriptedModel, subagent_tool


def build_researcher() -> Harness:
    # a fresh, isolated harness — its own context every call
    return Harness(ScriptedModel(["SQLite is the store; the index is on tenant_id."]))


research = subagent_tool("research", build_researcher)

parent = Harness(
    ScriptedModel([
        ("tool", "research", {"task": "What store and index does the ledger use?"}),
        "Confirmed: SQLite, indexed on tenant_id.",
    ]),
    tools=[research],
)
state = parent.run("Find out how the ledger is stored.")
print("result:", state.result)
```

`build_harness` is called **per invocation**, so each subagent starts clean. Only
`state.result` (or a fallback excerpt) crosses back — the parent's context stays
lean. More in [Adapters, MCP & subagents](adapters-and-mcp.md).

---

## 8. Drop a component into your own loop

**When:** you already have a runtime (your own `while` loop, LangGraph, another
framework) and want to reuse a pyhar component inside it. `component_hooks` folds
components into a dict of stage callables you invoke by hand — the same hooks
`Harness.run` calls internally.

```python
from pyhar import Message, ScriptedModel, ToolOutputBudget
from pyhar.adapters import component_hooks
from pyhar.core.state import HarnessState

model = ScriptedModel([
    ("tool", "grep", {"q": "TODO"}),
    "Found the TODOs; wrapping up.",
])

hooks = component_hooks([ToolOutputBudget(max_tokens=30)])

state = HarnessState()
state.add_message(Message(role="user", content="find the TODOs"))

for _ in range(5):
    hooks["before_model"](state)
    resp = model(state.messages, [])
    state.add_message(Message(role="assistant", content=resp.text or "",
                              tool_calls=list(resp.tool_calls)))
    if resp.tool_calls:
        for call in resp.tool_calls:
            result = "match at line 0\n" + ("x" * 2000)      # a big tool result
            result = hooks["after_tool"](state, call, result)  # component runs here
            state.add_message(Message(role="tool", content=result,
                                      tool_call_id=call.id, name=call.name))
    else:
        state.result = resp.text
        break

print("result:", state.result)
print("tokens saved by ToolOutputBudget:", state.memory.get("_tool_savings", 0))
```

`component_hooks(components)` returns callables keyed by stage — `before_model`,
`after_model`, `before_tool` (returns a denial string or `None`), `after_tool`
(chains the result), `after_turn`, `should_stop` (list of votes), `on_start`,
`on_end`. The [LangGraph and OpenAI-Agents adapters](adapters-and-mcp.md) are thin
binders over this same dict.

---

## 9. Go async — and run a turn's tool calls in parallel

**When:** your tools are I/O-bound (HTTP, DB, MCP servers) and you're inside an
event loop. `AsyncHarness` (new in 0.3.0) is a `Harness` subclass with
`await harness.arun(task)`: async models and tools are awaited natively, plain
sync ones are offloaded to a thread via `asyncio.to_thread` so they never block
the loop, and **components stay sync and work unchanged**. With
`parallel_tools=True`, all tool calls issued in one turn run concurrently via
`asyncio.gather` — results still come back in call order.

```python
import asyncio
import time

from pyhar import AsyncHarness, Response, ScriptedModel, ToolCall, tool


async def fetch_page(url: str) -> str:
    """An async tool — e.g. an aiohttp call in real life."""
    await asyncio.sleep(0.2)  # simulated network latency
    return f"<html>content of {url}</html>"


fetch = tool(fetch_page, name="fetch_page")

# One scripted turn issues TWO fetches — they run concurrently.
model = ScriptedModel([
    Response(tool_calls=[
        ToolCall(id="a", name="fetch_page", arguments={"url": "https://a.example"}),
        ToolCall(id="b", name="fetch_page", arguments={"url": "https://b.example"}),
    ]),
    "Fetched both pages in parallel.",
])


async def main() -> None:
    harness = AsyncHarness(model, tools=[fetch], parallel_tools=True)
    t0 = time.monotonic()
    state = await harness.arun("Fetch a.example and b.example.")
    print("result: ", state.result)
    print(f"elapsed: {time.monotonic() - t0:.2f}s  (two 0.2s fetches, run concurrently)")


asyncio.run(main())
```

Mixing is free: sync `@tool` functions, async `def` models, and sync closures
that return coroutines (how MCP-wrapped tools arrive) all work — awaitables are
awaited, everything else runs in a thread. Gating still happens first:
`before_tool` hooks run **in call order** before any parallel execution, so
`Permissions` and `LoopGuard` keep their guarantees. The sync `Harness` gets the
same per-turn concurrency (threads instead of `gather`) via
`Harness(..., parallel_tools=True)`. Full version:
[`examples/async_agent.py`](../examples/async_agent.py).

---

## 10. Survive flaky providers / route by cost

**When:** a provider 529s at the worst moment, or you want frontier quality
without frontier cost on every turn. The 0.3.0 combinators in `pyhar.models`
are just `Model`s, so they nest and drop into any harness:
`RetryModel` absorbs transient failures with exponential backoff,
`FallbackModel` fails over to a backup, and `RouterModel` picks a model per
call — pair it with `BudgetPolicy` for the strong-then-cheap tiering pattern.

```python
from pyhar import BudgetPolicy, Harness, ScriptedModel, tool
from pyhar.models import FallbackModel, RetryModel, RouterModel


class FlakyModel:
    """Simulates a provider that fails twice before recovering."""

    def __init__(self, inner):
        self.inner, self.attempts = inner, 0

    def __call__(self, messages, tools):
        self.attempts += 1
        if self.attempts <= 2:
            raise ConnectionError("simulated outage")
        return self.inner(messages, tools)


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


# -- Retry: ride out transient outages -------------------------------------
flaky = FlakyModel(ScriptedModel(["recovered and answered"]))
state = Harness(RetryModel(flaky, max_retries=3, base_delay=0.01)).run("q")
print(f"retry:    {state.result!r} after {flaky.attempts} attempts")

# -- Fallback: the primary is truly down, serve from the backup ------------
dead = FlakyModel(ScriptedModel(["never"]))
dead.attempts = -10**9  # always failing
fb = FallbackModel([dead, ScriptedModel(["served by the backup model"])])
state = Harness(fb).run("q")
print(f"fallback: {state.result!r} (served index {fb.last_served})")

# -- Tiering: start strong, downshift to cheap when the budget trips -------
tier = {"key": "strong"}
router = RouterModel(
    {
        "strong": ScriptedModel([("tool", "lookup", {"q": "deep question"}),
                                 "strong model finished the hard part"]),
        "cheap": ScriptedModel(["cheap model wrapped up the rest"]),
    },
    route=lambda messages, tools: tier["key"],
    default="strong",
)
budget = BudgetPolicy(
    max_total_tokens=100_000,
    soft_fraction=0.0000001,  # trip immediately for the demo
    on_over_soft=lambda state: tier.update(key="cheap"),
)
state = Harness(router, components=[budget], tools=[lookup]).run("hard task")
print(f"tiering:  {state.result!r} (last served: {router.last_key})")
```

They compose: `RetryModel(FallbackModel([primary, backup]))` retries the whole
failover chain. `RetryModel(model, max_retries=3, base_delay=1.0, max_delay=30.0,
retry_on=(Exception,))` controls what and how long; `FallbackModel.last_served`
tells you which model answered (`None` after a failed call); `RouterModel.last_key`
is set only after a successful call. Full version:
[`examples/model_routing.py`](../examples/model_routing.py).

---

## 11. Get structured JSON output

**When:** downstream code needs machine-readable output, not prose. Combine
recipe 4's `Verifier` with the ready-made checks in `pyhar.checks` (new in
0.3.0): `json_schema_check(schema)` validates the final answer against a
pragmatic, zero-dependency JSON-Schema subset, and on failure feeds the model
the **exact violation** so the retry is targeted.

```python
from pyhar import Harness, ScriptedModel, Verifier
from pyhar.checks import json_schema_check, parse_json_result

schema = {
    "type": "object",
    "required": ["answer", "confidence"],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "additionalProperties": False,
}

model = ScriptedModel([
    # 1st try: prose + wrong shape -> Verifier feeds back the exact violations
    'Sure! Here you go: {"answer": 42}',
    # 2nd try: valid JSON (an answer wrapped in a json code fence works too)
    '{"answer": "use SQLite", "confidence": 0.9}',
])

harness = Harness(model, components=[Verifier(json_schema_check(schema), max_retries=2)])
state = harness.run("Which store should we use? Respond as JSON.")

print("verified:", state.memory.get("_verified"))
print("parsed:  ", parse_json_result(state))   # -> a real dict
```

The subset covers `type` (including list form, e.g. `["string", "null"]`),
`properties`, `required`, `items`, `enum` (JSON equality — `true` never matches
`1`), and `additionalProperties: false`. Typos are caught early: an unknown type
name like `"int"` raises `ValueError` at construction, not silently at run time.
`parse_json_result(state)` tries the whole answer as JSON first, then fenced
```` ```json ```` blocks **last-first** (the final fence is usually the real
answer). Also available: `contains_check("42", case_sensitive=False)` and
`regex_check(r"answer:\s*\d+")` for lighter-weight assertions.

---

## 12. Break tool-call loops

**When:** the agent calls the same tool with the same arguments over and over —
a classic failure mode where each retry burns tokens and changes nothing.
`LoopGuard` (new in 0.3.0) watches tool calls in `before_tool`: once an
identical `(name, arguments)` pair repeats `max_repeats` times in a row, further
identical calls are **denied** and the model gets a nudge to change approach.
Arguments are canonicalized (key order doesn't matter) and `max_total_repeats`
backstops non-consecutive repeats across the whole run.

```python
from pyhar import Harness, LoopGuard, ScriptedModel, tool


@tool
def grep(q: str) -> str:
    """Search the repo."""
    return "no matches"


model = ScriptedModel([
    ("tool", "grep", {"q": "TODO"}),
    ("tool", "grep", {"q": "TODO"}),   # identical again — streak of 2 allowed
    ("tool", "grep", {"q": "TODO"}),   # 3rd in a row -> DENIED with a nudge
    "grep keeps returning nothing — answering with what I have: no TODOs.",
])

harness = Harness(model, components=[LoopGuard(max_repeats=2)], tools=[grep])
state = harness.run("Find the TODOs.")

print("result:", state.result)
print("denied:", state.memory.get("_loop_guard"))
```

The denial string comes back to the model as the tool result, so it can recover
in-band instead of crashing the run. Defaults are `max_repeats=3` and
`max_total_repeats=8`; counters reset in `on_start`, so a reused `Harness` starts
every run clean. `presets.coding_agent` (recipe 1) now includes a `LoopGuard`
out of the box, and — like every built-in — it's registered by name:
`registry.get("loop_guard")`.

---

## 13. Stream the answer as it's generated

**When:** you're building a UI and want tokens on screen as they arrive.
Construct the harness with `stream=True`; any component's
`on_delta(state, delta)` receives the chunks, and the loop's semantics are
otherwise unchanged. Models that can't stream degrade gracefully.

```python
from pyhar import Component, Harness, ScriptedModel


class LivePrinter(Component):
    def on_delta(self, state, delta):
        print(delta, end="", flush=True)


model = ScriptedModel(["streaming answers feel alive"])
state = Harness(model, components=[LivePrinter()], stream=True).run("say something")
print()
print("final result:", state.result)
```

Every stock backend streams (`AnthropicModel`, `OpenAIModel`, `OllamaModel`,
`ScriptedModel`), and the [combinators](models.md#combinators--retry-fallback-routing)
forward streaming, so `RetryModel(AnthropicModel(...))` still streams.
`Tracer(include_deltas=True)` records a `delta` event per chunk.

---

## 14. Ship a harness as shareable config

**When:** you want harness compositions in JSON/YAML — checked into a repo,
A/B-tested, or published alongside a component package. Components resolve by
their registered name; third-party packages join via the `pyhar.components`
entry-point group (`registry.load_entrypoints()`).

```python
from pyhar import ScriptedModel, harness_from_config, tool


@tool
def read_log(path: str) -> str:
    """Return a big log blob."""
    return "log line\n" * 500


config = {                       # this dict could come straight from JSON/YAML
    "system": "You are a careful log analyst.",
    "components": [
        {"name": "tool_output_budget", "args": {"max_tokens": 200}},
        "loop_guard",
        "tracer",
    ],
    "budget": {"max_context_tokens": 2000},
    "max_turns": 10,
}

model = ScriptedModel([("tool", "read_log", {"path": "app.log"}), "all quiet"])
harness = harness_from_config(config, model=model, tools=[read_log])
state = harness.run("scan the log")

print("result:", state.result)
print("tokens saved:", state.memory.get("_tool_savings", 0))
```

Unknown config keys raise `ValueError` (typos fail loudly), keyword overrides
win over config values, and `harness_cls=AsyncHarness` builds the async twin
from the same config.

---

## 15. Let the traces tune the harness

**When:** you've composed a harness but the knobs (`max_tokens`, `target_tokens`,
`max_turns`, which optional components to include) are guesses. `tune` searches a
config space, using the trace signals components leave in `state.memory` as
directional hints, and keeps only changes that measurably improve the objective.

```python
from pyhar import Range, ScriptedModel, tool, tune


@tool
def read_log(path: str) -> str:
    """Return a big log blob."""
    return "log line\n" * 500


def make_model():   # fresh model per run — ScriptedModel is consumed
    return ScriptedModel([("tool", "read_log", {"path": "app.log"}), "done"])


space = {
    "components": [
        {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
    ],
}

report = tune(space, model_factory=make_model, tasks=["summarize the log"],
              tools=[read_log], budget_runs=12, seed=7)

print(report.table())            # every candidate: score, success, tokens
print(report.explain())          # e.g. "tool-output budget never fired — tighten it"
print(report.best_config)        # plain JSON -> harness_from_config (recipe 14)
```

The winning config is shareable data — check it in, load it with
`harness_from_config`, re-verify it any time with `bench` (recipe 5). Full
signal→hint table and objective weighting → [Tuning](tuning.md).

---

## Semantics worth knowing in 0.3.0

Small behavioral fixes that make harnesses safer to reuse and reason about:

- **`Budget(max_turns=0)` means zero turns.** `None` means unlimited; `0` is no
  longer treated as "no limit".
- **Your `Budget` is never mutated.** `Harness` copies the budget you pass in,
  so one `Budget` object can be shared across harnesses safely.
- **`on_end` always runs.** Component teardown executes in a `try/finally`, so
  tracers flush and stores close even when the run raises `BudgetExceeded`.
- **Reused harnesses start clean.** `Verifier` and `LoopGuard` reset their
  per-run state in `on_start`, so calling `harness.run(...)` twice never leaks
  retry counts or repeat counters between runs.
- **`Response.stop_reason`** carries the provider's normalized stop/finish
  reason across the Anthropic, OpenAI, and Ollama backends.
- **Built-in components self-register** in `pyhar.registry` under their `.name`
  (e.g. `registry.get("compactor")`).

---

Every recipe above is one small composition of the same shared parts. Mix them:
`Permissions` + `Tracer` + `Verifier` + `ToolOutputBudget` all live in one
`components=[...]` list. See [Components](components.md) for the full catalog and
[Concepts](concepts.md) for how the loop and lifecycle fit together.
