# Adapters, MCP & subagents

pyhar's runtime is small and its components are pure. That combination makes it
easy to *lift the components out of the pyhar loop* and run them somewhere else —
inside a hand-rolled loop, a LangGraph agent, or an OpenAI-Agents `Runner` — and
to *pull other ecosystems in*, most notably tools from any MCP server. This page
covers three interop surfaces:

- **`component_hooks`** — the pure hook dict that every adapter is built on.
- **Framework adapters** — `to_langgraph_middleware` and `to_openai_agents_hooks` (both experimental).
- **MCP tools** — `tools_from_mcp` / `tools_from_mcp_session`.

It closes with **subagents** (`subagent_tool` / `spawn`), which use nothing but a
second `Harness` to get context isolation.

See also: [Components](components.md) · [Concepts](concepts.md) ·
[Model backends](models.md) · [Cookbook](cookbook.md).

```bash
pip install pyhar-agents   # import name is `pyhar`
```

---

## `component_hooks` — the pure core

`pyhar.adapters.component_hooks(components)` folds an iterable of
[components](components.md) into a plain `dict[str, Callable]`, one entry per
lifecycle stage. No `Harness` is involved — it is the framework-agnostic seam
that both adapters below are thin binders over, and the thing you call directly
when you own the loop.

The returned dict has these keys:

| Key | Signature | Runs components in order and… |
| --- | --- | --- |
| `on_start` | `(state)` | fans out to each `on_start` |
| `before_model` | `(state)` | fans out to each `before_model` |
| `after_model` | `(state, response)` | fans out to each `after_model` |
| `before_tool` | `(state, call) -> str \| None` | returns the **first** non-`None` denial string (or `None`) |
| `after_tool` | `(state, call, result) -> result` | **chains** the result through every `after_tool` |
| `after_turn` | `(state)` | fans out to each `after_turn` |
| `should_stop` | `(state) -> list[bool \| None]` | collects each vote into a list |
| `on_end` | `(state)` | fans out to each `on_end` |

Two keys behave specially and are worth calling out:

- **`before_tool`** returns a *denial string* — if any component returns a
  string, that string is meant to become the tool result and the real tool is
  skipped. `component_hooks` returns the first such string it sees.
- **`after_tool`** is a *reducer*: the result of each component is fed to the
  next, so an output-shrinking component like `ToolOutputBudget` actually
  changes what the following components (and your model) see.
- **`should_stop`** returns the raw list of votes; your loop decides how to
  combine them. (pyhar's own loop treats a single `False` on a no-tool-call
  turn as "re-open the task".)

### Driving components from your own loop

If you already have a loop and only want pyhar's components, wire the hooks in at
the matching points. The snippet below is a complete, runnable mini-loop built on
`ScriptedModel` (no API key) — it shows where each hook goes:

```python
from pyhar import HarnessState, Message, ScriptedModel, ToolOutputBudget
from pyhar.adapters import component_hooks

# A component that caps tool output at ~10 tokens (see the Components page).
hooks = component_hooks([ToolOutputBudget(max_tokens=10)])

model = ScriptedModel([
    ("tool", "readfile", {"path": "big.txt"}),   # turn 1: call a tool
    "done",                                        # turn 2: final answer
])

def readfile(path: str) -> str:
    return "X" * 500   # a deliberately huge tool result

state = HarnessState()
state.add_message(Message(role="user", content="summarize big.txt"))

hooks["on_start"](state)
for _ in range(5):
    hooks["before_model"](state)
    resp = model(state.messages, state.tools)
    hooks["after_model"](state, resp)

    if resp.tool_calls:
        for call in resp.tool_calls:
            denial = hooks["before_tool"](state, call)      # None = allowed
            if denial is not None:
                result = denial
            else:
                result = readfile(**call.arguments)
                result = hooks["after_tool"](state, call, result)  # shrinks it
            state.add_message(Message(role="tool", content=str(result)))
        hooks["after_turn"](state)
        continue

    # no tool calls -> a candidate final answer
    hooks["after_turn"](state)
    votes = hooks["should_stop"](state)
    if not any(v is False for v in votes):
        state.result = resp.text
        break

hooks["on_end"](state)
print(state.result)                       # -> "done"
print(len(state.messages[-1].content))    # the tool result was capped, not 500
```

The point: you never had to construct a `Harness`. If you *do* want the batteries
(budgets, turn limits, token counting), just use `Harness(...).run(task)` and skip
all of this — the manual path exists for when you are embedding components in a
foreign runtime.

---

## LangGraph middleware (experimental)

> **Experimental.** LangChain's middleware surface evolves upstream; treat this
> as a starting point and adjust the hook mapping to your installed version.

`pyhar.adapters.to_langgraph_middleware(components)` returns a LangChain 1.0
`AgentMiddleware` instance that forwards to your pyhar components, so a
`Compactor` or `ToolOutputBudget` runs *inside* a LangGraph `create_agent`
unchanged.

The mapping is:

| LangChain middleware hook | pyhar hook |
| --- | --- |
| `before_model` | `before_model` |
| `wrap_tool_call` | `after_tool` (so `ToolOutputBudget` really shrinks results) |
| `after_model` | `after_turn` |

Because `wrap_tool_call` wraps the real call, this adapter *can* substitute the
tool result: pyhar's `after_tool` return value is written back onto the tool
message's `content`. The middleware instance exposes the pyhar `HarnessState` as
`.pyhar_state` for inspection.

```python
from pyhar import Compactor, ToolOutputBudget
from pyhar.adapters import to_langgraph_middleware

middleware = to_langgraph_middleware([ToolOutputBudget(max_tokens=500)])

# from langchain.agents import create_agent
# agent = create_agent(model, tools=[...], middleware=[middleware])
# ... after a run:
# print(middleware.pyhar_state.memory.get("_tool_savings"))
```

pyhar itself has **no** LangChain dependency — importing the adapter module never
imports LangChain, and calling `to_langgraph_middleware` raises a clear
`ImportError` if `langchain` is not installed.

---

## OpenAI-Agents hooks (experimental)

> **Experimental.** Lazy-imports the `agents` SDK; importing the adapter module
> never requires it.

`pyhar.adapters.to_openai_agents_hooks(components)` returns an `agents.RunHooks`
instance that forwards to your components, so the same `Verifier` /
`ToolOutputBudget` you use elsewhere runs inside an OpenAI-Agents `Runner`.

The mapping is:

| OpenAI-Agents `RunHooks` | pyhar hook |
| --- | --- |
| `on_llm_start` | `before_model` |
| `on_tool_end` | `after_tool` |
| `on_llm_end` | `after_turn` |

**Important limitation:** `RunHooks.on_tool_end` cannot substitute the tool
result in place. This adapter therefore runs `after_tool` **for its side effects
only** — sandbox stashing, savings accounting, tracing — and *cannot* shrink what
reaches the model. If you need output to actually be rewritten before the model
sees it, prefer the LangGraph adapter or pyhar's own loop.

```python
from pyhar import Verifier
from pyhar.adapters import to_openai_agents_hooks

run_hooks = to_openai_agents_hooks([Verifier(...)])

# from agents import Runner
# await Runner.run(agent, "task", hooks=run_hooks)
```

---

## MCP interop

MCP already won the tool-interface layer, so pyhar sits *on top of* it rather than
reinventing tools. Point the adapter at a server's tool descriptors plus a
`call_tool` callable and get back ordinary pyhar `Tool` objects — which you can
hand to a `Harness` and then budget with `ToolOutputBudget`.

### `tools_from_mcp(descriptors, call_tool)`

The pure adapter. It takes *already-fetched* descriptors, so it is fully testable
without a live server. Each descriptor may be a dict or an object; the adapter
reads `name`, `description`, and `inputSchema` (or `input_schema`). `call_tool` is
`call_tool(name, arguments) -> result`.

```python
import pyhar.mcp
from pyhar import Harness, ScriptedModel

# Descriptors as you'd get from an MCP `list_tools` call:
descriptors = [
    {
        "name": "search",
        "description": "Search the docs.",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
]

def call_tool(name: str, arguments: dict) -> str:
    # your MCP client call goes here; stubbed for the example
    return f"results for {arguments.get('q')!r}"

tools = pyhar.mcp.tools_from_mcp(descriptors, call_tool)

model = ScriptedModel([
    ("tool", "search", {"q": "budgets"}),
    "Found the budgets page.",
])
state = Harness(model, tools=tools).run("look up budgets")
print(state.result)     # -> "Found the budgets page."
```

### `tools_from_mcp_session(session)`

An async convenience for the official `mcp` SDK's `ClientSession`. It calls
`session.list_tools()` and wires each returned descriptor to `session.call_tool`:

```python
import pyhar.mcp

# async with ClientSession(read, write) as session:
#     await session.initialize()
#     tools = await pyhar.mcp.tools_from_mcp_session(session)
```

Note that the returned tools invoke an **async** `call_tool`, while pyhar's
default loop is synchronous. For a sync `Harness`, either bridge the async call
yourself and pass it to `tools_from_mcp`, or drive the tools from an async
runtime. This surface is experimental and depends on the `mcp` SDK, which pyhar
does not require.

---

## Subagents

A subagent is just another `Harness` with its *own* context window. The elegant
way to relate it to the parent is to expose it as a `Tool`: the parent model
calls it with a task string, the sub-harness runs in isolation, and only a
relevant excerpt — its result — comes back. This is the
"return-only-relevant-excerpts" contract, decoupled from any runtime.

### `subagent_tool(name, build_harness, *, description=..., task_field="task")`

Wraps a fresh, isolated sub-harness as a callable tool. `build_harness` is invoked
**per call**, so every subagent starts from a clean context (that is the
isolation). Only the subagent's final result flows back to the parent. The
generated input schema has a single string field (`task_field`, default `"task"`).

### `spawn(harness, task, *, excerpt=None)`

Runs a sub-harness on `task` and returns an excerpt of its result. Resolution
order:

1. `excerpt(state)` if you passed a custom `excerpt` callable;
2. `state.result` (only set on a clean finishing turn) as a string;
3. otherwise the subagent's **last assistant text** — so a subagent that
   exhausted its turn budget returns real content instead of the string
   `"None"`;
4. a `"[subagent ended without a result: …]"` placeholder if there is nothing.

`subagent_tool` uses `spawn` under the hood.

Here is a runnable, key-free example. The parent delegates once to a `research`
subagent; the subagent has its own `ScriptedModel` and its own context:

```python
from pyhar import Harness, ScriptedModel, subagent_tool

def build_research_harness() -> Harness:
    # Isolated context: its own model, tools, and message history.
    model = ScriptedModel(["The capital of France is Paris."])
    return Harness(model)

parent_model = ScriptedModel([
    ("tool", "research", {"task": "What is the capital of France?"}),
    "According to research, it's Paris.",
])

tools = [subagent_tool("research", build_research_harness)]
state = Harness(parent_model, tools=tools).run("Answer using the research tool.")

print(state.result)   # -> "According to research, it's Paris."
```

The parent never saw the subagent's internal turns — only the excerpt returned by
`spawn`. Because `build_research_harness` is called on every invocation, two calls
to the same subagent tool never share context.

For a custom excerpt (e.g. return the last N chars, or a summary field), call
`spawn` directly instead of going through `subagent_tool`:

```python
from pyhar import Harness, ScriptedModel, spawn

sub = Harness(ScriptedModel(["a very long research dump ... final answer: 42"]))
short = spawn(sub, "compute the answer", excerpt=lambda s: s.result[-2:])
print(short)   # -> "42"
```

---

## Where to go next

- [Components](components.md) — the lifecycle hooks these adapters forward to.
- [Concepts](concepts.md) — `Harness`, `HarnessState`, and the run loop.
- [Model backends](models.md) — swapping `ScriptedModel` for a real provider.
- [Cookbook](cookbook.md) — end-to-end recipes combining budgets, MCP, and subagents.
