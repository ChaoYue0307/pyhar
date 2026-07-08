# pyhar

**Composable primitives for agent harnesses.** The swappable parts that every
serious agent's harness — Claude Code, Devin, deepagents — re-implements by hand,
packaged as small, typed modules that all share one interface and drop into *any*
loop.

> Not another agent framework. pyhar owns only the model-facing scaffolding
> (compaction, tool-output budgeting, verification, context assembly, budgets).
> Bring your own runtime (a plain `while` loop, LangGraph, the OpenAI SDK) and
> your own tools (MCP). It **composes with** your framework, not against it.

## The idea in one picture

An agent is `Model + Harness`. Today the *harness* — the code around the model —
is where reliability actually comes from, yet nobody ships it as reusable parts.
Every layer of the 2026 stack is crowded except this one:

| Layer | Who owns it |
| --- | --- |
| Agent products ("the harness") | Claude Code · Devin · deepagents (closed / bundled) |
| Orchestration runtime | LangGraph · OpenAI/MS/Google SDKs · CrewAI (crowded) |
| **Harness components** | **← unowned. This is pyhar.** |
| Programming / compiler | DSPy · TextGrad · GEPA |
| Model + tools | MCP (settled) |

pyhar relates *all* those agents' harnesses under one representation: a
harness is just an **ordered composition of shared `Component` parts**. Swap a
part, keep everything else.

## The keystone: `Component`

`Component` is the "`nn.Module` of harnesses." It hooks into the agent loop
lifecycle; every hook has a no-op default, so you override only what you need.

```python
class Component:
    def on_start(self, state): ...       # once, before the first model call
    def before_model(self, state): ...   # shape context (compaction, retrieval)
    def after_model(self, state, response): ...
    def after_tool(self, state, call, result): return result   # budget tool output
    def after_turn(self, state): ...     # verify, checkpoint, write memory
    def should_stop(self, state): ...    # vote on stopping (Verifier re-opens here)
    def on_end(self, state): ...
```

## Quickstart (no API key — ships with a scripted model)

```python
from pyhar import ScriptedModel, tool
from pyhar.presets import coding_agent

@tool
def read_file(path: str) -> str:
    return "decision: use SQLite\n" + "log\n" * 400 + "TODO: add index"

model = ScriptedModel([
    ("tool", "read_file", {"path": "db.py"}),
    "Done. The answer is 42.",
])

def check(state):
    return ("42" in (state.result or ""), "answer must contain '42'")

harness = coding_agent(model, tools=[read_file], check=check, context_tokens=300)
state = harness.run("Inspect db.py and tell me the answer.")

print(state.result, state.usage, state.memory["_tool_savings"])
```

## Model backends (bring your own, or use one of these)

A `Model` is anything mapping messages + tools to a `Response`. Provider SDKs are
**optional extras**, lazy-imported — `import pyhar` never requires them.

```python
from pyhar.models import AnthropicModel, OpenAIModel, OllamaModel
from pyhar.presets import coding_agent

harness = coding_agent(AnthropicModel("claude-opus-4-8"), tools=[...])   # pip install "pyhar-agents[anthropic]"
harness = coding_agent(OpenAIModel("gpt-4o-mini"), tools=[...])          # pip install "pyhar-agents[openai]"
harness = coding_agent(OllamaModel("llama3.1"), tools=[...])             # local OSS, zero deps
```

| Backend | Notes |
| --- | --- |
| `AnthropicModel` | official SDK; adaptive thinking + `effort`, no `temperature`/`budget_tokens`; defaults to `claude-opus-4-8` |
| `OpenAIModel` / `OpenAICompatibleModel` | OpenAI SDK; pass `base_url=` for vLLM / Together / LM Studio / any OSS OpenAI-compatible server |
| `OllamaModel` | local models over stdlib `urllib` — **zero dependencies** |
| `ScriptedModel` / `EchoModel` | deterministic, key-free — for tests, examples, CI |

## Components

| Component | What it packages | Hook |
| --- | --- | --- |
| `Compactor` | staged compaction — trim tool outputs, then collapse history preserving decisions + open items | `before_model` |
| `ToolOutputBudget` | shrink oversized tool results, stash full output in a sandbox (the seam MCP leaves open) | `after_tool` |
| `Verifier` | verify→retry driven by *your* check (tests / eval / judge), not just schema shape | `after_turn` / `should_stop` |
| `ContextBuilder` | budget-aware per-step context assembly (system prompt, retrieval, window trimming) | `before_model` |
| `Memory` | tiered core/recall/archival (Letta / LangMem mental model), storage-agnostic | `on_start` / `before_model` |
| `StateArtifact` | externalized progress + decisions so a fresh context reconstructs "where am I" (`MemoryStore`/`FileStore`) | `on_start` / `after_turn` |
| `BudgetPolicy` | explicit token/cost ceilings + soft-warning hook for model tiering | `after_turn` |
| `Harness` | the batteries-included loop that runs a composition | — |
| `subagent_tool` | expose an isolated sub-harness as a `Tool` (return-only-relevant-excerpt) | — |
| `bench` | A/B two harness configs on one task; report tokens/cost/turns | — |

## Compose with your runtime and with MCP

```python
from pyhar.adapters import component_hooks, to_langgraph_middleware
from pyhar.mcp import tools_from_mcp

hooks = component_hooks([Compactor(...), ToolOutputBudget(...)])   # drive from any loop
mw    = to_langgraph_middleware([Compactor(...)])                  # experimental: LangGraph
tools = tools_from_mcp(mcp_descriptors, call_tool)                 # import MCP tools as Tool objects
```

Run the demos:

```bash
python examples/react_agent.py     # full coding-agent harness
python examples/minimal_loop.py    # SAME components in a hand-rolled loop
python examples/real_model.py      # Anthropic / OpenAI / Ollama by env, else ScriptedModel
pytest                             # the suite (30 tests, no keys needed)
```

## Design principles

1. **Narrow waist.** pyhar is the model-facing scaffolding only. It never
   tries to be the runtime or the tool standard.
2. **Adopt one part at a time.** Every component works standalone and in any loop.
3. **Measured, not asserted.** `bench` exists so a claim like "60% fewer tokens"
   is a number you can reproduce.
4. **Zero runtime dependencies.** Bring your own model and tools.

## Roadmap

Shipped in 0.1.0: model backends (Anthropic/OpenAI/Ollama), runtime adapters
(`component_hooks` + experimental LangGraph / OpenAI-Agents binders), MCP
interop, and the `ContextBuilder` / `Memory` / `StateArtifact` / `subagent_tool`
primitives.

Still deliberately deferred:

- **Registry** — a torchvision/timm-style catalog. Only pays off at adoption
  critical mass, so it stays a seed (`pyhar.registry`) for now.
- **Runtime-structure optimization** ("autograd from production traces") — the
  ambitious v2 bet; research-grade, kept out of the launch claim.
- **Hardened framework adapters** — the LangGraph / OpenAI-Agents binders are
  marked experimental until pinned against a released middleware surface.

## Install & naming

The PyPI *distribution* name `pyhar` is already taken (an abandoned 2022 stub),
so the package publishes as **`pyhar-agents`** — but the **import name is `pyhar`**
(same split as `opencv-python` → `import cv2`):

```bash
pip install pyhar-agents               # then:  import pyhar
pip install "pyhar-agents[anthropic]"  # + the Anthropic backend
pip install "pyhar-agents[openai]"     # + the OpenAI backend
```

## License

MIT.
