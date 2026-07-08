# Changelog

All notable changes to pyhar are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

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
