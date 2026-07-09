"""Break repeated-tool-call loops — a classic agent failure mode, packaged.

An agent that calls the same tool with the same arguments over and over is
stuck. ``LoopGuard`` watches consecutive tool calls; once an identical
``(name, arguments)`` pair repeats ``max_repeats`` times in a row, further
identical calls are denied (via ``before_tool``) with a nudge telling the model
to change approach. As a backstop, ``max_total_repeats`` bounds how many times
an identical call may run across the whole run, consecutive or not.

Counters reset at ``on_start``, so a reused Harness starts each run clean.
Denied repeats are recorded in ``state.memory['_loop_guard']``.
"""
from __future__ import annotations

import json

from ..core.component import Component
from ..core.state import HarnessState, ToolCall


def _key(call: ToolCall) -> str:
    # canonical at every nesting level so {"a": 1, "b": 2} == {"b": 2, "a": 1}
    args = json.dumps(call.arguments, sort_keys=True, default=repr)
    return f"{call.name}:{args}"


class LoopGuard(Component):
    name = "loop_guard"

    def __init__(self, *, max_repeats: int = 3, max_total_repeats: int = 8):
        self.max_repeats = max_repeats
        self.max_total_repeats = max_total_repeats
        self._last_key: str | None = None
        self._streak = 0
        self._totals: dict[str, int] = {}

    def on_start(self, state: HarnessState) -> None:
        # reset per-run so a reused Harness never inherits stale counters
        self._last_key = None
        self._streak = 0
        self._totals = {}

    def before_tool(self, state: HarnessState, call: ToolCall) -> str | None:
        key = _key(call)
        self._streak = self._streak + 1 if key == self._last_key else 1
        self._last_key = key
        self._totals[key] = self._totals.get(key, 0) + 1

        streak_hit = self._streak > self.max_repeats
        total_hit = self._totals[key] > self.max_total_repeats
        if streak_hit or total_hit:
            state.memory.setdefault("_loop_guard", []).append(
                {"tool": call.name, "args": call.arguments, "streak": self._streak,
                 "total": self._totals[key]}
            )
            if streak_hit:
                detail = f"{self._streak - 1} times in a row"
            else:
                detail = f"{self._totals[key] - 1} times this run"
            return (
                f"[loop guard: {call.name} was already called with these exact arguments "
                f"{detail}. The result will not change — try a different tool, different "
                f"arguments, or answer with what you have.]"
            )
        return None
