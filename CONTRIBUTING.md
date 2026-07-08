# Contributing to pyhar

Thanks for your interest! pyhar is a small, unopinionated library of
composable harness primitives. The bar for a new primitive is: **does it
implement `Component`, work standalone, and drop into any loop?**

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # tests (no API keys needed — ships a ScriptedModel)
ruff check src tests
mypy src
```

## Design principles (please keep them)

1. **Narrow waist.** Model-facing scaffolding only — never the runtime or the
   tool standard. It composes *with* LangGraph / OpenAI SDK / MCP, not against them.
2. **One interface.** Every primitive subclasses `Component` and overrides only
   the hooks it needs.
3. **Adopt one at a time.** Each component must be useful on its own and in a
   plain `while` loop.
4. **Measured, not asserted.** New primitives should come with a `bench`-able
   demonstration of the win where applicable.
5. **Zero required runtime deps.** Provider SDKs are optional extras and
   lazy-imported. `import pyhar` must never require them.

## Adding a component

1. Add `src/pyhar/components/your_thing.py` subclassing `Component`.
2. Export it from `components/__init__.py` and the top-level `__init__.py`.
3. Add tests in `tests/` using `ScriptedModel` (deterministic, key-free).
4. Update the README table and the CHANGELOG.

## Adding a model backend

Implement the `Model` protocol (`__call__(messages, tools) -> Response`). Real
providers must lazy-import their SDK and accept an injected `client=` for
testing. See `models/anthropic.py` for the pattern.
