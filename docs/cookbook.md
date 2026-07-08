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
)
print(report.table())

base = next(r for r in report.runs if r.name == "baseline")
tuned = next(r for r in report.runs if r.name.startswith("tuned"))
saved = base.input_tokens - tuned.input_tokens
print(f"input tokens saved: {saved}")
```

`bench(task, {name: factory}, success=...)` runs each config from a **fresh
factory** and returns a `BenchReport` (`.table()`, `.runs`). Runs that include a
`ToolOutputBudget` also record `state.memory["_tool_savings"]`.

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

Every recipe above is one small composition of the same shared parts. Mix them:
`Permissions` + `Tracer` + `Verifier` + `ToolOutputBudget` all live in one
`components=[...]` list. See [Components](components.md) for the full catalog and
[Concepts](concepts.md) for how the loop and lifecycle fit together.
