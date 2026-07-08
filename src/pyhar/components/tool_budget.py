"""Keep large tool results from blowing up the context.

This is the seam MCP explicitly leaves open: MCP standardizes *how* a tool is
called and what its schema is, but not what happens when a tool dumps 50 KB of
output into your window. Fires in ``after_tool``: if a result exceeds
``max_tokens`` it is shrunk (head+tail, or your ``compressor``), and the full
result is stashed out-of-context in ``state.memory['_sandbox'][call.id]`` so it
can be fetched on demand.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.state import HarnessState, ToolCall

Compressor = Callable[[str], str]


class ToolOutputBudget(Component):
    name = "tool_output_budget"

    def __init__(
        self,
        max_tokens: int = 400,
        *,
        head_fraction: float = 0.6,
        compressor: Compressor | None = None,
    ):
        self.max_tokens = max_tokens
        self.head_fraction = head_fraction
        self.compressor = compressor

    def after_tool(self, state: HarnessState, call: ToolCall, result):
        text = result if isinstance(result, str) else repr(result)
        n = state.token_counter(text)
        if n <= self.max_tokens:
            return result

        state.memory.setdefault("_sandbox", {})[call.id] = text  # full fidelity
        shrunk = self.compressor(text) if self.compressor else self._head_tail(text)
        saved = n - state.token_counter(shrunk)
        state.memory["_tool_savings"] = state.memory.get("_tool_savings", 0) + max(0, saved)
        return shrunk

    def _head_tail(self, text: str) -> str:
        budget_chars = self.max_tokens * 4
        head = int(budget_chars * self.head_fraction)
        tail = budget_chars - head
        omitted = len(text) - head - tail
        if omitted <= 0:
            return text
        return (
            f"{text[:head]}\n"
            f"…[{omitted} chars elided — full output in sandbox['{{call_id}}']]…\n"
            f"{text[-tail:]}"
        )
