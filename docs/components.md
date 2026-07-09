# Components

Components are the pluggable pieces that shape a run. Each one is a small object
with no-op-by-default lifecycle hooks; the [Harness](concepts.md) calls those
hooks at fixed points in the loop, and components cooperate by reading and
writing `state.memory`. This page is a reference for every built-in component
shipped with `pyhar` (installed as `pip install pyhar-agents`, imported as
`import pyhar`).

```python
from pyhar import (
    Harness, ScriptedModel, tool,
    Compactor, ToolOutputBudget, Verifier, BudgetPolicy,
    ContextBuilder, Memory, StateArtifact, MemoryStore, FileStore,
    Permissions, Tracer, LoopGuard,
)
```

Every built-in component is also auto-registered by its `name` in
`pyhar.registry`, so you can look one up by string — e.g.
`registry.get("compactor")` returns the `Compactor` class.

All snippets below run without an API key — they drive the harness with
[`ScriptedModel`](models.md), which replays a fixed list of steps: a plain
string is a final text answer, and `("tool", name, {args})` is a single tool
call.

## The hooks, in one place

Every component subclasses `Component` and overrides some subset of these hooks
(all no-ops by default):

| Hook | When it fires | Notable return |
| --- | --- | --- |
| `on_start(state)` | once, before the loop | — |
| `before_model(state)` | before each model call | — |
| `after_model(state, response)` | after each model call | — |
| `on_delta(state, delta)` | per streamed text chunk (harness built with `stream=True`) | — |
| `before_tool(state, call)` | before a tool runs | return a `str` to **deny** the call (the string becomes the tool result) |
| `after_tool(state, call, result)` | after a tool runs | return value replaces `result` (chained across components) |
| `after_turn(state)` | after each turn | — |
| `should_stop(state)` | end of a no-tool-call turn | `True`/`False`/`None`; a single `False` re-opens the task |
| `on_end(state)` | once, after the loop | — |

See [Concepts](concepts.md) for the full lifecycle and how these compose.

---

## Compactor

**What it does:** staged context compaction — the Claude-Code-style heuristic,
packaged. When the working context grows past a target, it first trims old tool
outputs to a snippet, then (if still too big) collapses older turns into a
decision-preserving synopsis.

**Fires on:** `before_model`.

**Constructor:**

```python
Compactor(
    target_tokens: int | None = None,   # positional; falls back to budget.max_context_tokens
    *,
    keep_last: int = 4,                 # recent messages never touched
    tool_snippet_tokens: int = 40,      # size old tool outputs are trimmed to
    preserve=default_preserve,          # (line: str) -> bool: keep this line in the synopsis?
    summarizer: Model | None = None,    # if set, an LLM writes the synopsis instead of the heuristic
)
```

`default_preserve` keeps lines mentioning markers like `decision`, `chose`,
`todo`, `bug`, `error`, `must`, `constraint`, `next step`, `open question`.

**Writes to `state.memory`:** `_compactions` — a list of `(turn, stage)` tuples,
where `stage` is `"stage1_trim"` or `"stage2_collapse"`.

```python
from pyhar import Harness, ScriptedModel, Budget, Compactor

model = ScriptedModel(["done"])
state = Harness(
    model,
    components=[Compactor(target_tokens=10)],  # tiny target forces compaction
    budget=Budget(max_context_tokens=10),
).run("We decided to shard by tenant_id. Also TODO: add retries, and fix the login bug.")
print(state.memory.get("_compactions"))  # e.g. [(1, 'stage1_trim')] or [(1, 'stage2_collapse')]
```

---

## ToolOutputBudget

**What it does:** keeps a single large tool result from blowing up the context.
If a result exceeds `max_tokens` it is shrunk (head+tail, or your `compressor`),
and the full-fidelity output is stashed out-of-context so it can be fetched on
demand.

**Fires on:** `after_tool`.

**Constructor:**

```python
ToolOutputBudget(
    max_tokens: int = 400,        # positional; results at/under this pass through untouched
    *,
    head_fraction: float = 0.6,   # share of the budget spent on the head vs the tail
    compressor=None,              # (text: str) -> str: custom shrinker; default is head+tail
)
```

**Writes to `state.memory`:**
- `_sandbox` — a dict keyed by `call.id` holding the full, untruncated output.
- `_tool_savings` — running integer count of tokens saved.

```python
from pyhar import Harness, ScriptedModel, tool, ToolOutputBudget

@tool
def dump() -> str:
    """Return a big blob."""
    return "x" * 10_000

model = ScriptedModel([("tool", "dump", {}), "done"])
state = Harness(
    model,
    components=[ToolOutputBudget(max_tokens=20)],
    tools=[dump],
).run("dump the blob")

print(state.memory["_tool_savings"] > 0)      # True
print(len(next(iter(state.memory["_sandbox"].values()))))  # 10000 — full output retained
```

---

## Verifier

**What it does:** first-class verify → retry, driven by *your* check. When the
model produces a candidate final answer (a turn with no tool calls), the check
runs; on failure it injects feedback and re-opens the task (votes `False` on
`should_stop`) up to `max_retries`.

**Fires on:** `after_turn` (runs the check, injects feedback) and `should_stop`
(re-opens the task).

**Constructor:**

```python
Verifier(
    check: Callable[[HarnessState], tuple[bool, str]],  # positional: (passed, feedback_if_failing)
    *,
    max_retries: int = 2,
)
```

**Writes to `state.memory`:** `_verified` — the boolean from the most recent
check.

The retry counter resets in `on_start`, so a reused `Harness` gets its full
retry budget back on every run — it is safe to call `.run(...)` repeatedly on
the same instance.

```python
from pyhar import Harness, ScriptedModel, Verifier

def must_say_ok(state):
    text = (state.result or "")
    return ("ok" in text.lower(), "answer must contain 'ok'")

model = ScriptedModel(["nope", "all ok"])  # first try fails the check, second passes
state = Harness(
    model,
    components=[Verifier(must_say_ok, max_retries=2)],
).run("give me the status")

print(state.memory["_verified"])  # True
print(state.result)               # "all ok"
```

---

## BudgetPolicy

**What it does:** explicit token/cost ceilings plus a soft-warning hook. Raises
`BudgetExceeded` when a hard ceiling is crossed, and fires `on_over_soft` once
when usage passes `soft_fraction` of the token ceiling (the seam for
model-tiering).

**Fires on:** `after_turn`.

**Constructor (all keyword-only):**

```python
BudgetPolicy(
    *,
    max_cost: float | None = None,
    max_total_tokens: int | None = None,
    soft_fraction: float = 0.8,
    on_over_soft: Callable[[HarnessState], None] | None = None,
)
```

**Writes to `state.memory`:** nothing. It reads `state.usage` and either raises
or calls your `on_over_soft` callback. (Hard breaches raise
`pyhar.core.harness.BudgetExceeded`.)

> Note: `BudgetPolicy` is the *component* form of ceilings. The `Budget` passed
> to `Harness(..., budget=...)` also carries `max_context_tokens`,
> `max_total_tokens`, `max_turns`, and `max_cost` for the harness itself — see
> [Concepts](concepts.md).

```python
from pyhar import Harness, ScriptedModel, BudgetPolicy

warnings = []
policy = BudgetPolicy(
    max_total_tokens=1_000_000,      # generous hard ceiling — won't trip here
    soft_fraction=0.0,               # fire the soft warning immediately
    on_over_soft=lambda s: warnings.append(s.usage.total_tokens),
)
state = Harness(model=ScriptedModel(["done"]), components=[policy]).run("hello")
print(warnings)  # [<some token count>] — soft hook fired once
```

---

## ContextBuilder

**What it does:** budget-aware context assembly as a testable object. Ensures a
system prompt is present, optionally injects retrieved snippets, and keeps the
window under a token target by dropping the oldest whole turn-groups (an
assistant-with-tool-calls plus its tool results are dropped atomically, so no
`tool_use`/`tool_result` pair is ever orphaned).

**Fires on:** `before_model`.

**Constructor (all keyword-only):**

```python
ContextBuilder(
    *,
    system: str | None = None,                       # inserted if no system message exists yet
    retriever: Callable[[HarnessState], list[str]] | None = None,  # snippets to inject each step
    max_tokens: int | None = None,                   # falls back to budget.max_context_tokens
    keep_last: int = 6,                              # recent messages protected from dropping
)
```

**Writes to `state.memory`:** `_dropped` — running count of messages dropped to
stay under budget.

```python
from pyhar import Harness, ScriptedModel, ContextBuilder

state = Harness(
    ScriptedModel(["answer"]),
    components=[ContextBuilder(
        system="You are terse.",
        retriever=lambda s: ["fact: the sky is blue"],
    )],
).run("what colour is the sky?")

# A system message and a "[retrieved context]" message were injected.
print([m.role for m in state.messages][:2])  # ['system', 'user']
```

---

## Memory

**What it does:** tiered memory as one primitive. A pinned `core` block is always
injected at start; `archival` entries are recalled each step by naive keyword
overlap with the latest real user message.

**Fires on:** `on_start` (inject core) and `before_model` (recall archival,
snapshot memory).

**Public API:** `remember(text)` appends an archival entry; `set_core(text)`
replaces the pinned block.

**Constructor (all keyword-only):**

```python
Memory(
    *,
    core: str = "",       # pinned block, always injected as a system message
    recall_k: int = 3,    # max archival entries recalled per step
)
```

**Writes to `state.memory`:** `_memory` — `{"core": ..., "archival": [...]}`,
refreshed each step.

```python
from pyhar import Harness, ScriptedModel, Memory

mem = Memory(core="User prefers SQLite.")
mem.remember("Decided to shard by tenant_id in 2026-Q3.")

state = Harness(ScriptedModel(["ok"]), components=[mem]).run(
    "How should I shard the tenant data?"
)
print(state.memory["_memory"]["core"])       # "User prefers SQLite."
print(state.memory["_memory"]["archival"])   # ["Decided to shard by tenant_id in 2026-Q3."]
```

---

## StateArtifact (+ MemoryStore / FileStore)

**What it does:** externalizes progress + decisions to a store outside the
context, so a fresh window can reconstruct "where am I". On start it loads the
artifact into context; after each turn it records the turn number and harvests
new decisions from the latest assistant message.

**Fires on:** `on_start` (restore), `after_turn` (harvest + save), `on_end`
(final save).

**Constructor:**

```python
StateArtifact(
    store: Store | None = None,          # positional; defaults to MemoryStore()
    *,
    preserve=default_preserve,           # (line: str) -> bool: which assistant lines count as decisions
)
```

**Stores:**
- `MemoryStore()` — in-process, ideal for tests / ephemeral runs.
- `FileStore(path)` — JSON-file-backed; survives process restarts and fresh
  contexts.

**Writes to `state.memory`:** `_artifact` — `{"decisions": [...], "turns": int}`.

```python
from pyhar import Harness, ScriptedModel, StateArtifact, MemoryStore

store = MemoryStore()
Harness(
    ScriptedModel(["Decision: we will cache results in Redis."]),
    components=[StateArtifact(store)],
).run("pick a cache")

print(store.load())  # {'decisions': ['Decision: we will cache results in Redis.'], 'turns': 1}

# A later run with the same store restores that artifact into context on start.
state2 = Harness(
    ScriptedModel(["continuing"]),
    components=[StateArtifact(store)],
).run("keep going")
print(state2.messages[1].content)  # "[restored state]\n..." — prior decisions reloaded
```

Swap `MemoryStore()` for `FileStore("run.json")` to persist across processes.

---

## Permissions

**What it does:** the authorization seam as a component. Decides allow/deny per
tool call by allowlist, denylist, or a policy callback. A denied call never runs —
the denial string is returned to the model as the tool result so it can adapt.

**Fires on:** `before_tool`.

**Constructor (all keyword-only):**

```python
Permissions(
    *,
    allow: Iterable[str] | None = None,   # if set, ONLY these tools may run
    deny: Iterable[str] | None = None,    # these tools are blocked
    policy: Callable[[HarnessState, ToolCall], str | None] | None = None,  # None=allow, str=deny reason
)
```

Evaluation order: `policy` first, then `deny`, then `allow`.

**Writes to `state.memory`:** `_denied` — a list of `{"tool": name, "reason": ...}`.

```python
from pyhar import Harness, ScriptedModel, tool, Permissions

@tool
def deploy() -> str:
    """Ship to prod."""
    return "shipped"

model = ScriptedModel([("tool", "deploy", {}), "handled it"])
state = Harness(
    model,
    components=[Permissions(deny=["deploy"])],
    tools=[deploy],
).run("deploy the service")

print(state.memory["_denied"])  # [{'tool': 'deploy', 'reason': "tool 'deploy' is blocked"}]
```

---

## LoopGuard

*New in 0.3.0.*

**What it does:** breaks repeated-tool-call loops — the classic stuck-agent
failure mode. It watches tool calls by identity key `(name, canonicalized
arguments)` (argument dicts are canonicalized at every nesting level, so
`{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` count as the same call). Once an
identical call has run `max_repeats` times *in a row*, further identical calls
are denied with a nudge telling the model to change approach. As a backstop,
`max_total_repeats` bounds how many times an identical call may run across the
whole run, consecutive or not. `LoopGuard` is included in the `coding_agent`
preset.

**Fires on:** `before_tool` (the denial string becomes the tool result, so the
call never runs) and `on_start` (counters reset per run, so a reused `Harness`
never inherits stale counters).

**Constructor (all keyword-only):**

```python
LoopGuard(
    *,
    max_repeats: int = 3,        # identical calls allowed in a row before denial
    max_total_repeats: int = 8,  # identical calls allowed across the whole run
)
```

**Writes to `state.memory`:** `_loop_guard` — a list with one entry per denied
repeat: `{"tool": name, "args": arguments, "streak": ..., "total": ...}`.

```python
from pyhar import Harness, ScriptedModel, tool, LoopGuard

@tool
def search(q: str) -> str:
    """Search for q."""
    return "no results"

model = ScriptedModel([
    ("tool", "search", {"q": "answer"}),
    ("tool", "search", {"q": "answer"}),
    ("tool", "search", {"q": "answer"}),
    ("tool", "search", {"q": "answer"}),  # 4th identical call in a row — denied
    "giving up",
])
state = Harness(model, components=[LoopGuard()], tools=[search]).run("find the answer")

print(state.memory["_loop_guard"])
# [{'tool': 'search', 'args': {'q': 'answer'}, 'streak': 4, 'total': 4}]
```

The first three calls run normally (`max_repeats=3`); the fourth is denied and
the model instead receives a tool result like
`[loop guard: search was already called with these exact arguments 3 times in a
row. The result will not change — try a different tool, different arguments, or
answer with what you have.]`.

---

## Tracer

**What it does:** observability — records the run as a structured event stream,
one event per lifecycle step, and optionally streams events live to a `sink`.
Zero cost when you don't add it.

**Fires on:** `on_start`, `after_model`, `before_tool`, `after_tool`, `on_end`.

**Constructor (keyword-only):**

```python
Tracer(
    *,
    sink: Callable[[dict], None] | None = None,   # e.g. print, for a live event log
)
```

**Writes to `state.memory`:** `_trace` — the ordered list of event dicts. Event
`event` values are `"start"`, `"model"`, `"tool_call"`, `"tool_result"`, `"end"`.

```python
from pyhar import Harness, ScriptedModel, tool, Tracer

@tool
def ping() -> str:
    """Reply pong."""
    return "pong"

tracer = Tracer(sink=print)  # live log to stdout
state = Harness(
    ScriptedModel([("tool", "ping", {}), "done"]),
    components=[tracer],
    tools=[ping],
).run("ping once")

print([e["event"] for e in state.memory["_trace"]])
# ['start', 'model', 'tool_call', 'tool_result', 'model', 'end']
```

---

## See also

- [Concepts](concepts.md) — the harness loop, state, and how hooks compose.
- [Model backends](models.md) — `ScriptedModel`, `AnthropicModel`, `OpenAIModel`, and friends.
- [Adapters, MCP & subagents](adapters-and-mcp.md) — reuse these components under LangGraph / OpenAI Agents, and pull in MCP tools.
- [Cookbook](cookbook.md) — end-to-end recipes combining several components.
