# pyhar documentation

Start here, then dive into whichever guide fits what you're building. Every code
snippet runs offline with `ScriptedModel` (no API key).

| Guide | What's in it |
| --- | --- |
| **[Concepts](concepts.md)** | The mental model, the `Component` interface, `HarnessState`, and the `Harness.run` loop step-by-step. Read this first. |
| **[Components](components.md)** | Reference for every built-in component — constructor arguments, which hooks it fires on, what it writes to `state.memory`, and a runnable snippet. |
| **[Model backends & tools](models.md)** | The `Model` protocol, `AnthropicModel` / `OpenAIModel` / `OllamaModel` / `EchoModel`, writing your own backend, and automatic tool schemas. |
| **[Adapters, MCP & subagents](adapters-and-mcp.md)** | `component_hooks` for your own loop, the LangGraph / OpenAI-Agents binders, MCP tool import, and isolated subagents. |
| **[Cookbook](cookbook.md)** | Copy-paste recipes: safe agents, observability, verify→retry, long-horizon resume, token budgeting, subagents. |

New here? The project [README](../README.md) has the 60-second pitch, the
architecture diagram, and the use-case table.
