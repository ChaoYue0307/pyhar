"""EXPERIMENTAL — expose pyhar components as LangChain/LangGraph middleware.

LangChain 1.0's ``create_agent`` supports middleware with ``before_model`` /
``after_model`` / ``wrap_tool_call`` hooks. This adapter builds an
``AgentMiddleware`` subclass that forwards those hooks to your pyhar
components, so a ``Compactor`` or ``ToolOutputBudget`` runs inside a LangGraph
agent unchanged.

Lazy-imports langchain, so importing this module never requires it. The exact
middleware surface evolves upstream — treat this as a starting point and adjust
the hook mapping to your installed version if needed.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.component import Component


def to_langgraph_middleware(components: Iterable[Component]) -> Any:
    """Return a LangChain ``AgentMiddleware`` instance wrapping the components.

    Maps ``before_model`` -> component ``before_model``, ``wrap_tool_call`` ->
    component ``after_tool`` (so ``ToolOutputBudget`` actually shrinks results),
    and ``after_model`` -> component ``after_turn``. The pyhar ``HarnessState``
    is exposed on the middleware instance as ``.pyhar_state`` for inspection.
    """
    try:
        from langchain.agents.middleware import AgentMiddleware  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_langgraph_middleware needs LangChain 1.0+ — `pip install langchain`. "
            "pyhar itself has no LangChain dependency."
        ) from e

    from ..adapters.manual import component_hooks
    from ..core.state import HarnessState, ToolCall

    hooks = component_hooks(components)

    class pyharMiddleware(AgentMiddleware):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.pyhar_state = HarnessState()

        def before_model(self, state: Any) -> Any:  # noqa: D401
            hooks["before_model"](self.pyhar_state)
            return None

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            result = handler(request)
            call = ToolCall(
                id=str(getattr(request, "id", "") or ""),
                name=str(getattr(request, "name", "") or ""),
                arguments=getattr(request, "args", {}) or {},
            )
            payload = getattr(result, "content", result)
            new = hooks["after_tool"](self.pyhar_state, call, payload)
            if hasattr(result, "content"):
                try:
                    result.content = new
                except Exception:  # pragma: no cover - result may be immutable
                    return new
                return result
            return new

        def after_model(self, state: Any) -> Any:
            hooks["after_turn"](self.pyhar_state)
            return None

    return pyharMiddleware()
