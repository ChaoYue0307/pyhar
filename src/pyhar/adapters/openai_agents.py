"""EXPERIMENTAL — bridge pyhar components to the OpenAI Agents SDK.

The OpenAI Agents SDK exposes ``RunHooks`` (on_agent_start / on_llm_start /
on_tool_end / …). This adapter returns a ``RunHooks`` subclass that forwards to
your pyhar components, so the same ``Verifier`` / ``ToolOutputBudget`` you
use elsewhere runs inside an OpenAI-Agents ``Runner``.

Lazy-imports ``agents``; importing this module never requires it.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.component import Component


def to_openai_agents_hooks(components: Iterable[Component]) -> Any:
    """Return an ``agents.RunHooks`` instance wrapping the components."""
    try:
        from agents import RunHooks  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_openai_agents_hooks needs the OpenAI Agents SDK — "
            "`pip install openai-agents`. pyhar itself has no dependency on it."
        ) from e

    from ..adapters.manual import component_hooks
    from ..core.state import HarnessState, ToolCall

    hooks = component_hooks(components)

    class pyharRunHooks(RunHooks):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._state = HarnessState()

        async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            hooks["before_model"](self._state)

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            # RunHooks.on_tool_end cannot substitute the result in-place, so this
            # runs after_tool for its SIDE EFFECTS only (sandbox stash + savings
            # accounting). To actually shrink what reaches the model, prefer the
            # LangGraph adapter or pyhar's own loop.
            call = ToolCall(id="", name=str(getattr(tool, "name", "") or ""), arguments={})
            hooks["after_tool"](self._state, call, result)

        async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
            hooks["after_turn"](self._state)

    return pyharRunHooks()
