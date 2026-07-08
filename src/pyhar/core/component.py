"""The keystone abstraction — the "nn.Module" of pyhar.

Everything in pyhar lives or dies on this interface. A ``Component`` hooks
into the agent loop lifecycle; every hook has a no-op default, so a component
overrides only what it needs (exactly like ``nn.Module.forward``). The *same*
Component object can:

  * run inside pyhar's own ``Harness.run`` loop,
  * be applied by hand in a plain ``while`` loop (see examples/minimal_loop.py),
  * or drop into another runtime via an adapter (planned).

Because a harness is just an ordered list of these parts, *any* agent's harness
becomes expressible as a composition of shared, swappable components. That is
what it means to "relate all the harnesses for various agents."
"""
from __future__ import annotations

from typing import Any

from .model import Response
from .state import HarnessState, ToolCall


class Component:
    name: str = "component"

    # --- lifecycle hooks; override the ones you need ---

    def on_start(self, state: HarnessState) -> None:
        """Called once, before the first model call."""

    def before_model(self, state: HarnessState) -> None:
        """Shape the working context just before the model is called
        (compaction, retrieval, budget-aware assembly)."""

    def after_model(self, state: HarnessState, response: Response) -> None:
        """Inspect/react to the raw model response (already appended as an
        assistant message by the harness)."""

    def before_tool(self, state: HarnessState, call: ToolCall) -> str | None:
        """Gate a tool call before it executes. Return ``None`` to allow it, or
        a string to DENY it — the string is used as the tool result instead of
        running the tool (e.g. a permission-denied message). The first component
        to return a string wins."""
        return None

    def after_tool(self, state: HarnessState, call: ToolCall, result: Any) -> Any:
        """Transform a tool result before it enters the context. Return the
        (possibly modified) result. Chained across components in order."""
        return result

    def after_turn(self, state: HarnessState) -> None:
        """Post-turn housekeeping: verification, checkpointing, memory writes."""

    def should_stop(self, state: HarnessState) -> bool | None:
        """Vote on stopping. ``True`` forces a stop, ``False`` forces continue,
        ``None`` abstains. On a candidate-final turn (no tool calls) a single
        ``False`` re-opens the task (e.g. a failed Verifier)."""
        return None

    def on_end(self, state: HarnessState) -> None:
        """Called once, after the loop ends."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
