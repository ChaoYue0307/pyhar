# pyhar documentation

Start here, then dive into whichever guide fits what you're building. Every code
snippet runs offline with `ScriptedModel` (no API key).

| Guide | What's in it |
| --- | --- |
| **[Concepts](concepts.md)** | The mental model, the `Component` interface, `HarnessState`, the `Harness.run` loop step-by-step, and the async loop (`AsyncHarness` / `arun`, parallel tool calls). Read this first. |
| **[Components](components.md)** | Reference for every built-in component, including the new `LoopGuard` — constructor arguments, which hooks it fires on, what it writes to `state.memory`, and a runnable snippet. |
| **[Model backends & tools](models.md)** | The `Model` protocol, `AnthropicModel` / `OpenAIModel` / `OllamaModel` / `EchoModel`, the `RetryModel` / `FallbackModel` / `RouterModel` combinators, normalized `Response.stop_reason`, writing your own backend, and automatic tool schemas. |
| **[Adapters, MCP & subagents](adapters-and-mcp.md)** | `component_hooks` for your own loop, the LangGraph / OpenAI-Agents binders, MCP tool import, and isolated subagents. |
| **[Cookbook](cookbook.md)** | Copy-paste recipes: safe agents, observability, verify→retry, long-horizon resume, token budgeting, subagents — plus new 0.3.0 recipes for resilient model stacks, loop guarding, output checks, and benchmarking with `bench`. |

New here? The project [README](../README.md) has the 60-second pitch, the
architecture diagram, and the use-case table.
