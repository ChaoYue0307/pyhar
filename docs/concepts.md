# Concepts

pyhar is **not** another agent framework. It owns one layer — the
**harness–component layer** — and leaves the runtime, the tools, and the model
provider to you. An agent's harness (compaction, tool-output budgeting,
verification, context assembly, permissions, budgets) becomes a *composition of
shared, swappable parts* that all implement one interface. Swap the model, keep
the components; swap a component, keep the loop.

```
pip install pyhar-agents      # distribution name
import pyhar                   # import name
```

Version 0.2.0.

The mental model has four pieces:

1. **[Component](components.md)** — the keystone interface (the "nn.Module" of
   pyhar). Eight lifecycle hooks, all no-op by default.
2. **`HarnessState`** — the single shared object every component reads and
   mutates (pyhar's analog of Inspect AI's `TaskState`).
3. **`Harness.run`** — a small, standard tool-calling loop with the component
   hooks woven in at fixed points.
4. **Portability** — the *same* components run in your own `while` loop, or in
   another runtime via an [adapter](adapters-and-mcp.md).

---

## The Component interface

A `Component` hooks into the agent loop lifecycle. Every hook has a no-op
default, so a component overrides only what it needs. All eight hooks, with
exact signatures and return semantics:

| Hook | Signature | Purpose / return semantics |
| --- | --- | --- |
| `on_start` | `on_start(state) -> None` | Called once, before the first model call. Seed memory, load state. |
| `before_model` | `before_model(state) -> None` | Shape the working context just before the model is called — compaction, retrieval, budget-aware assembly. |
| `after_model` | `after_model(state, response) -> None` | Inspect/react to the raw model response. The harness has already appended it as an assistant message. |
| `before_tool` | `before_tool(state, call) -> str \| None` | **Gate** a tool call before it runs. Return `None` to allow; return a **string to DENY** — that string becomes the tool result instead of running the tool. The first component to return a string wins. |
| `after_tool` | `after_tool(state, call, result) -> result` | Transform a tool result before it enters the context. Return the (possibly modified) result. **Chained** across components in order. |
| `after_turn` | `after_turn(state) -> None` | Post-turn housekeeping: verification, checkpointing, memory writes. |
| `should_stop` | `should_stop(state) -> bool \| None` | Vote on stopping. `True` forces a stop; `False` forces continue; `None` abstains. On a candidate-final turn (no tool calls) a single `False` **re-opens** the task (e.g. a failed `Verifier`). |
| `on_end` | `on_end(state) -> None` | Called once, after the loop ends. Flush, summarize, close. |

Two hooks carry the interesting control-flow semantics:

- **`before_tool` deny** — returning a string short-circuits the tool. The
  string is used verbatim as the tool result (a permission-denied message, a
  cached answer, a stub). This is how `Permissions` blocks a call. *All*
  `before_tool` hooks still run even after a denial, so observers like `Tracer`
  always see every call regardless of component order; only the first denial is
  used.
- **`should_stop` veto** — when the model emits no tool calls, the harness
  treats the turn as a candidate final answer. Any single component returning
  `False` re-opens the task and forces another turn. Any single component
  returning `True` (on a turn that *did* have tool calls) forces an early stop.

See [Components](components.md) for the batteries-included set (`Compactor`,
`ToolOutputBudget`, `Verifier`, `BudgetPolicy`, `ContextBuilder`, `Memory`,
`Permissions`, `Tracer`, …).

---

## HarnessState — the shared object

`HarnessState` is the single object threaded through a run. Components read and
mutate it; keeping it explicit (rather than hidden inside a graph runtime) is
what makes components portable across loops.

Key fields:

| Field | Type | What it holds |
| --- | --- | --- |
| `messages` | `list[Message]` | The live conversation (system / user / assistant / tool). |
| `tools` | `dict[str, Tool]` | Name → tool available this run. |
| `memory` | `dict` | Component scratch space. Components stash things here (`_tool_savings`, `_denied`, `_trace`, `_compactions`, `_stop_reason`, …). |
| `usage` | `Usage` | `input_tokens`, `output_tokens`, `cost`, and the `total_tokens` property. |
| `budget` | `Budget` | `max_context_tokens`, `max_total_tokens`, `max_turns`, `max_cost`. |
| `turn` | `int` | Turns taken so far. |
| `done` | `bool` | Set `True` when the loop should end. |
| `result` | `Any` | The final answer (defaults to the last text response). |
| `last_response` | `Response` | The most recent raw model response. |
| `last_turn_had_tool_calls` | `bool` | Whether the last turn dispatched tools. |

Methods:

- `state.count_tokens(messages=None)` — token estimate over `messages` (or the
  live context) using the state's `token_counter`.
- `state.add_message(message)` — append a `Message`.

`Budget.max_context_tokens` is the *target working-window size* a `Compactor`
keeps the live context under; the other three are hard caps on the whole run
(the harness raises `BudgetExceeded` on `max_total_tokens` / `max_cost`, and
stops with `memory["_stop_reason"] = "max_turns"` on `max_turns`).

---

## The Harness.run loop

A `Harness` is `model + ordered components + tools + budget`:

```python
Harness(
    model,
    components=(),
    tools=(),
    *,
    system=None,
    budget=None,
    token_counter=default_token_counter,
    max_turns=20,
).run(task: str | list[Message]) -> HarnessState
```

`run()` is a standard tool-calling loop with the component lifecycle woven in at
fixed points, in this exact order:

1. **`on_start`** — every component, once.
2. **Loop** while not `done` (and under `max_turns` / hard budget):
   1. **`before_model`** — every component (shape the context).
   2. **model call** — `model(messages, tools)`; usage accumulated; the response
      is appended as an assistant message.
   3. **`after_model`** — every component (react to the response).
   4. **tool calls**, if any — for each call, in order:
      - **gate**: run every `before_tool`; the first returned string denies the
        call and becomes the result, otherwise the tool is **dispatched**.
      - **`after_tool`** — chained across all components to transform the result.
      - the result is appended as a `tool` message.
   5. **`after_turn`** — every component (housekeeping).
   6. **stop decision**:
      - *No tool calls* → candidate final answer. If any `should_stop` returns
        `False`, re-open and `continue`; else set `done = True` and record the
        result.
      - *Had tool calls* → if any `should_stop` returns `True`, set `done`.
3. **`on_end`** — every component, once. `run` returns the `HarnessState`.

```
on_start
  └─ loop: before_model → model → after_model
           → [ gate/dispatch tools + after_tool ]
           → after_turn → stop decision
on_end
```

### A runnable example

Everything below runs with **`ScriptedModel`** — no API key. A `ScriptedModel`
returns its queued items in order: a plain string is a final text answer, and
`("tool", name, {args})` is a single tool call.

```python
from pyhar import Harness, ScriptedModel, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Turn 1: call add(2, 3). Turn 2: give the final answer.
model = ScriptedModel([
    ("tool", "add", {"a": 2, "b": 3}),
    "The sum is 5.",
])

harness = Harness(model, tools=[add])
state = harness.run("What is 2 + 3?")

print(state.result)                 # -> The sum is 5.
print(state.turn)                   # -> 2
print(state.messages[-2].content)   # -> 5   (the tool message)
```

`@tool` auto-generates the JSON input schema from the function's type hints (via
`schema_from_signature`); pass `schema=` to `@tool` to override it.

### A tiny custom Component

Components are plain objects. Here is a gate that denies a tool by name using
`before_tool`, and records what it blocked in `state.memory`:

```python
from pyhar import Component, Harness, ScriptedModel, tool

@tool
def delete_everything() -> str:
    """Danger."""
    return "deleted!"

class Firewall(Component):
    name = "firewall"

    def on_start(self, state):
        state.memory["_denied"] = []

    def before_tool(self, state, call):
        if call.name == "delete_everything":
            state.memory["_denied"].append(call.name)
            return "[denied: not allowed]"   # string => DENY, becomes the result
        return None                          # None => allow

model = ScriptedModel([
    ("tool", "delete_everything", {}),
    "Okay, I won't do that.",
])

state = Harness(model, components=[Firewall()], tools=[delete_everything]).run("clean up")

print(state.memory["_denied"])       # -> ['delete_everything']
print(state.messages[-2].content)    # -> [denied: not allowed]
```

A `Verifier`-style component instead overrides `should_stop`: return `False` on
a candidate-final turn to force another turn until some condition holds.

---

## The same components in your own loop

The payoff: components are portable. Because `HarnessState` is an explicit
object and each hook is an ordinary method, you can drive the exact same
`Component` instances from a hand-written loop — no `Harness` required. The hooks
fire in the same order shown above:

```python
from pyhar import Component, HarnessState, Message, ScriptedModel

model = ScriptedModel(["done"])
components: list[Component] = [Firewall()]

state = HarnessState()
state.add_message(Message(role="user", content="hi"))

for c in components:
    c.on_start(state)

while not state.done:
    for c in components:
        c.before_model(state)

    response = model(list(state.messages), list(state.tools.values()))
    state.usage.add(response.usage)
    state.last_response = response
    state.last_turn_had_tool_calls = bool(response.tool_calls)
    state.add_message(Message(role="assistant",
                              content=response.text or "",
                              tool_calls=list(response.tool_calls)))
    for c in components:
        c.after_model(state, response)

    # (gate + dispatch + after_tool for each response.tool_call here)

    for c in components:
        c.after_turn(state)

    if not response.tool_calls:
        if any(c.should_stop(state) is False for c in components):
            continue
        state.done = True
        state.result = response.text

for c in components:
    c.on_end(state)
```

The same components also drop into other runtimes through
[adapters](adapters-and-mcp.md) (`component_hooks`, `to_langgraph_middleware`,
`to_openai_agents_hooks`).

---

## Where to go next

- [Components](components.md) — the built-in component library and how to write
  your own.
- [Model backends](models.md) — `AnthropicModel`, `OpenAIModel`,
  `OllamaModel`, `EchoModel`, and the `ScriptedModel` used throughout these docs.
- [Adapters, MCP & subagents](adapters-and-mcp.md) — run pyhar components inside
  other frameworks, pull tools from MCP, and spawn subagents.
- [Cookbook](cookbook.md) — end-to-end recipes and presets
  (`minimal_react`, `coding_agent`).
