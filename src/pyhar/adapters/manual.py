"""Drive pyhar components from your own loop, or map them to any runtime.

``component_hooks`` is the pure, framework-agnostic core: it folds a list of
components into a dict of callables keyed by lifecycle stage. The LangGraph and
OpenAI-Agents adapters are thin binders over this dict; you can also wire it into
a hand-rolled loop directly (see examples/minimal_loop.py for the manual style).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..core.component import Component
from ..core.model import Response
from ..core.state import HarnessState, ToolCall


def component_hooks(components: Iterable[Component]) -> dict[str, Callable]:
    """Fold components into stage callables that run them in order.

    Returns a dict with: on_start, before_model, after_model, after_tool,
    after_turn, should_stop, on_end. ``after_tool`` chains the result through
    each component; ``should_stop`` returns the list of votes.
    """
    comps = list(components)

    def on_start(state: HarnessState) -> None:
        for c in comps:
            c.on_start(state)

    def before_model(state: HarnessState) -> None:
        for c in comps:
            c.before_model(state)

    def after_model(state: HarnessState, response: Response) -> None:
        for c in comps:
            c.after_model(state, response)

    def before_tool(state: HarnessState, call: ToolCall) -> str | None:
        denial: str | None = None
        for c in comps:
            d = c.before_tool(state, call)
            if d is not None and denial is None:
                denial = d
        return denial

    def after_tool(state: HarnessState, call: ToolCall, result: Any) -> Any:
        for c in comps:
            result = c.after_tool(state, call, result)
        return result

    def after_turn(state: HarnessState) -> None:
        for c in comps:
            c.after_turn(state)

    def should_stop(state: HarnessState) -> list[bool | None]:
        return [c.should_stop(state) for c in comps]

    def on_end(state: HarnessState) -> None:
        for c in comps:
            c.on_end(state)

    return {
        "on_start": on_start,
        "before_model": before_model,
        "after_model": after_model,
        "before_tool": before_tool,
        "after_tool": after_tool,
        "after_turn": after_turn,
        "should_stop": should_stop,
        "on_end": on_end,
    }
