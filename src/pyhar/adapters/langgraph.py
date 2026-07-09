"""Run pyhar components inside a LangChain/LangGraph agent — as middleware.

Hardened against LangChain 1.x (``langchain>=1.0,<2``; exercised in integration
tests against 1.3), sync AND async (``invoke``/``ainvoke``/``astream``).

**Supported components** — those that act through the tool channel or observe:

- ``Permissions`` / ``LoopGuard``   (``before_tool`` gate — a denial SKIPS tool
  execution; the denial string reaches the model as the ``ToolMessage``)
- ``ToolOutputBudget``              (``after_tool`` — shrinks results in place)
- ``Tracer`` and other observers    (``on_start``/``before_model``/``after_turn``/``on_end``)

**Not supported here** — components that shape the *message channel* (inject,
rewrite, or re-open the conversation): ``Compactor``, ``Memory``,
``StateArtifact``, ``ContextBuilder``, and ``Verifier``. LangGraph owns its
message list; those components need pyhar's own loop (``Harness`` /
``AsyncHarness``). Passing one raises ``ValueError`` up front rather than
silently doing nothing.

    from langchain.agents import create_agent
    mw = to_langgraph_middleware([Permissions(deny=["rm"]), ToolOutputBudget(max_tokens=400)])
    agent = create_agent(model, tools=[...], middleware=[mw])

Hook mapping (LangChain -> pyhar): ``before_agent -> on_start``,
``before_model -> before_model``, ``wrap_tool_call/awrap_tool_call ->
before_tool + after_tool``, ``after_model -> after_turn``, ``after_agent ->
on_end``. The pyhar ``HarnessState`` lives on the middleware as
``.pyhar_state`` — read ``state.memory`` (``_denied``, ``_tool_savings``,
``_trace``, ...) after a run.

Limitations: one middleware instance holds one ``HarnessState``, so don't share
an instance across concurrently-running invokes (create one agent+middleware
per worker, or accept interleaved trace/memory entries). Each call to
``to_langgraph_middleware`` gets a unique middleware name, so multiple
instances can coexist in one agent.

Lazy-imports langchain; importing this module never requires it.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from ..core.component import Component

# component classes that require pyhar's own loop (message-channel semantics)
_UNSUPPORTED = ("Compactor", "Memory", "StateArtifact", "ContextBuilder", "Verifier")

_instance_counter = itertools.count(1)


def to_langgraph_middleware(components: Iterable[Component]) -> Any:
    """Return a LangChain ``AgentMiddleware`` instance wrapping the components."""
    try:
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import ToolMessage
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_langgraph_middleware needs LangChain 1.x — install "
            "'pyhar-agents[langgraph]' or 'langchain>=1.0'. "
            "pyhar itself has no LangChain dependency."
        ) from e

    from ..adapters.manual import component_hooks
    from ..core.state import HarnessState, ToolCall

    comps = list(components)
    bad = [type(c).__name__ for c in comps if type(c).__name__ in _UNSUPPORTED]
    if bad:
        raise ValueError(
            f"{', '.join(bad)} shape the message channel and cannot run as LangGraph "
            f"middleware — use them in pyhar's own Harness/AsyncHarness loop. "
            f"Supported here: Permissions, LoopGuard, ToolOutputBudget, Tracer, "
            f"and other tool-channel/observer components."
        )

    hooks = component_hooks(comps)

    def _to_call(request: Any) -> ToolCall:
        tc = getattr(request, "tool_call", None) or {}
        return ToolCall(
            id=str(tc.get("id") or ""),
            name=str(tc.get("name") or ""),
            arguments=dict(tc.get("args") or {}),
        )

    def _apply_after_tool(state: HarnessState, call: ToolCall, result: Any) -> Any:
        content = getattr(result, "content", None)
        if isinstance(content, str):
            new = hooks["after_tool"](state, call, content)
            if isinstance(new, str) and new != content and hasattr(result, "model_copy"):
                return result.model_copy(update={"content": new})
        else:
            # Command / non-text result: run after_tool for side effects only
            hooks["after_tool"](state, call, content)
        return result

    class PyharMiddleware(AgentMiddleware):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.pyhar_state = HarnessState()

        def before_agent(self, state: Any, runtime: Any = None) -> Any:
            hooks["on_start"](self.pyhar_state)
            return None

        def before_model(self, state: Any, runtime: Any = None) -> Any:
            hooks["before_model"](self.pyhar_state)
            return None

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            call = _to_call(request)
            denial = hooks["before_tool"](self.pyhar_state, call)
            if denial is not None:
                return ToolMessage(
                    content=denial, tool_call_id=call.id or "denied", name=call.name or None
                )
            return _apply_after_tool(self.pyhar_state, call, handler(request))

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            # async twin — langchain builds the async chain from THIS method
            # whenever either variant is overridden, so both must exist
            call = _to_call(request)
            denial = hooks["before_tool"](self.pyhar_state, call)
            if denial is not None:
                return ToolMessage(
                    content=denial, tool_call_id=call.id or "denied", name=call.name or None
                )
            return _apply_after_tool(self.pyhar_state, call, await handler(request))

        def after_model(self, state: Any, runtime: Any = None) -> Any:
            # keep observers accurate about whether this turn called tools
            try:
                msgs = state.get("messages") if isinstance(state, dict) else None
                last = msgs[-1] if msgs else None
                self.pyhar_state.last_turn_had_tool_calls = bool(
                    getattr(last, "tool_calls", None)
                )
            except Exception:  # never let bookkeeping break the agent
                pass
            hooks["after_turn"](self.pyhar_state)
            return None

        def after_agent(self, state: Any, runtime: Any = None) -> Any:
            hooks["on_end"](self.pyhar_state)
            return None

    # unique per-instance middleware name: create_agent requires distinct names
    PyharMiddleware.__name__ = f"PyharMiddleware{next(_instance_counter)}"
    PyharMiddleware.__qualname__ = PyharMiddleware.__name__
    return PyharMiddleware()
