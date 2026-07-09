"""Observability — record the run as a structured event stream.

Every serious harness needs a way to see what happened; ``Tracer`` records one
event per lifecycle step into ``state.memory['_trace']`` and (optionally) streams
them to a ``sink`` callback live. Zero cost when you don't add it.

    tracer = Tracer(sink=print)          # live event log
    state = Harness(model, components=[tracer, ...]).run(task)
    state.memory["_trace"]               # the full event list
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.component import Component
from ..core.model import Response
from ..core.state import HarnessState, ToolCall

Sink = Callable[[dict[str, Any]], None]


class Tracer(Component):
    name = "tracer"

    def __init__(self, *, sink: Sink | None = None, include_deltas: bool = False):
        self.sink = sink
        self.include_deltas = include_deltas

    def on_delta(self, state: HarnessState, delta: str) -> None:
        if self.include_deltas:
            self._emit(state, {"event": "delta", "turn": state.turn, "chars": len(delta)})

    def _emit(self, state: HarnessState, event: dict[str, Any]) -> None:
        state.memory.setdefault("_trace", []).append(event)
        if self.sink is not None:
            self.sink(event)

    def on_start(self, state: HarnessState) -> None:
        self._emit(state, {"event": "start", "tools": sorted(state.tools)})

    def after_model(self, state: HarnessState, response: Response) -> None:
        self._emit(
            state,
            {
                "event": "model",
                "turn": state.turn,
                "text": bool(response.text),
                "tool_calls": [tc.name for tc in response.tool_calls],
                "output_tokens": response.usage.output_tokens,
            },
        )

    def before_tool(self, state: HarnessState, call: ToolCall) -> str | None:
        self._emit(state, {"event": "tool_call", "turn": state.turn, "name": call.name, "args": call.arguments})
        return None

    def after_tool(self, state: HarnessState, call: ToolCall, result: Any) -> Any:
        chars = len(result) if isinstance(result, str) else len(repr(result))
        self._emit(state, {"event": "tool_result", "turn": state.turn, "name": call.name, "chars": chars})
        return result

    def on_end(self, state: HarnessState) -> None:
        self._emit(
            state,
            {
                "event": "end",
                "turns": state.turn,
                "has_result": state.result is not None,
                "input_tokens": state.usage.input_tokens,
                "output_tokens": state.usage.output_tokens,
                "cost": state.usage.cost,
            },
        )
