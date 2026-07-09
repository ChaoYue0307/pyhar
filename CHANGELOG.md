# Changelog

All notable changes to pyhar are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

Distributed on PyPI as `pyhar-agents`; the import name is `pyhar`.

## [0.3.0] — 2026-07-09

### Added
- **`AsyncHarness`** — the async twin of `Harness` (`await harness.arun(task)`).
  Awaits async models and tools; sync ones are offloaded via `asyncio.to_thread`
  so they never block the event loop. Components stay sync and work in both
  loops unchanged.
- **Model combinators** (`pyhar.models`): `RetryModel` (exponential backoff),
  `FallbackModel` (ordered provider failover), `RouterModel` (policy-routed
  cheap/strong tiering — pairs with `BudgetPolicy.on_over_soft`).
- **`LoopGuard`** component — denies a tool call once the identical
  `(name, arguments)` pair repeats too many times, with a nudge to change
  approach. Added to the `coding_agent` preset.
- **`pyhar.checks`** — ready-made `Verifier` checks: `contains_check`,
  `regex_check`, and `json_schema_check` (dependency-free JSON-Schema subset,
  tolerant of ```json fences) plus `parse_json_result`.
- **Parallel tool execution** — `Harness(..., parallel_tools=True)` runs a
  turn's tool calls concurrently (threads in the sync loop, `asyncio.gather`
  in the async loop); results always return in call order.
- **`Response.stop_reason`** — normalized provider stop/finish reason surfaced
  by the Anthropic, OpenAI, and Ollama backends.
- **`bench(trials=N)`** — per-config means, standard deviations, and success
  rate instead of single-run numbers.
- Built-in components are auto-registered in `pyhar.registry` by name.
- New examples: `async_agent.py`, `model_routing.py`.

### Fixed
- `Budget(max_turns=0)` now means zero turns; `None` means unlimited (0 was
  previously treated as unlimited).
- `Harness` copies the caller's `Budget` instead of mutating it in place — a
  Budget shared across harnesses is no longer silently corrupted.
- `on_end` hooks always run (try/finally), even when a run raises
  `BudgetExceeded` or a model error.
- `Verifier` and `LoopGuard` reset their per-run state in `on_start`, so a
  reused `Harness` gets a fresh retry budget and clean loop counters.
- `AsyncHarness` awaits coroutines returned by *sync* tool closures (e.g.
  MCP-wrapped tools) and async models hidden behind `functools.partial`;
  parallel async tools use join-then-raise semantics.
- `parse_json_result` tries the whole text as JSON before fence extraction
  (valid JSON containing backticks is no longer mangled) and prefers the last
  valid fenced block.
- `json_schema_check` supports list-form `type` (e.g. `["string", "null"]`),
  rejects unknown type names at construction, and uses JSON equality for
  `enum` (booleans no longer match numbers).
- Checks no longer fall back past an empty final answer to stale earlier text.
- `RouterModel.last_key` / `FallbackModel.last_served` read `None` after a
  failed call instead of a stale or never-served key.

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
