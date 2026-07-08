# Changelog

All notable changes to pyhar are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

Published to PyPI as `pyhar-agents` (the bare `pyhar` name is taken); the import
name is `pyhar`.

## [0.2.0] — 2026-07-08

### Added
- **Automatic tool schemas** — `@tool` now derives a JSON `input_schema` from the
  function's type hints (so real models actually see a tool's parameters).
  Public helper `schema_from_signature`; pass `schema=` to override.
- **`before_tool` lifecycle hook** on `Component` — gate a tool call before it
  runs; return a string to deny (it becomes the tool result). Wired into the
  `Harness` loop and the `component_hooks` adapter.
- **`Permissions`** component — allow/deny lists or a policy callback for
  tool authorization; denials recorded in `state.memory['_denied']`.
- **`Tracer`** component — records the run as a structured event stream in
  `state.memory['_trace']`, with an optional live `sink`.
- `docs/` — comprehensive guides (concepts, components, models, adapters,
  cookbook) plus more runnable examples.

## [0.1.0] — 2026-07-08

### Added
- **Model backends** (`pyhar.models`), all lazy-importing their SDKs:
  - `AnthropicModel` — official Anthropic SDK; adaptive thinking + effort, no
    `temperature`/`budget_tokens`; defaults to `claude-opus-4-8`.
  - `OpenAIModel` / `OpenAICompatibleModel` — OpenAI SDK, `base_url` for
    vLLM/Together/LM Studio/OSS servers.
  - `OllamaModel` — local OSS models over stdlib `urllib` (zero deps).
  - `EchoModel` — trivial key-free backend for smoke tests.
  - `pricing` table so `Usage.cost` is populated.
- **Runtime adapters** (`pyhar.adapters`): `component_hooks` (pure, tested)
  plus experimental `to_langgraph_middleware` and `to_openai_agents_hooks`.
- **MCP interop** (`pyhar.mcp`): `tools_from_mcp` to import MCP tools as
  `Tool` objects.
- **New primitives**: `ContextBuilder`, `Memory` (tiered), `StateArtifact`
  (with `MemoryStore` / `FileStore`).
- **Subagents** (`pyhar.subagent`): `subagent_tool` / `spawn` — isolated
  sub-harness exposed as a tool.
- Project hardening: MIT `LICENSE`, `py.typed`, GitHub Actions CI (ruff + mypy +
  pytest on 3.10–3.13), `CONTRIBUTING.md`, ruff/mypy config.

## [0.0.1] — 2026-07-08

### Added
- Core: `Component` interface, `Harness` loop, `HarnessState`, `Model` protocol,
  `ScriptedModel`, `Tool`.
- Components: `Compactor`, `ToolOutputBudget`, `Verifier`, `BudgetPolicy`.
- `bench` (A/B configs), `presets` (`minimal_react`, `coding_agent`), seed
  `registry`. Runnable examples and a passing test suite.
