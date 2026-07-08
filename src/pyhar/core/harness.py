"""The batteries-included loop.

A ``Harness`` is ``model + ordered components + tools + budget``. ``run()`` is a
standard tool-calling agent loop with the component lifecycle woven in. It is
deliberately small: the value is in the *components*, and they are portable —
you can apply them to your own loop or another runtime instead of using this.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .component import Component
from .model import Model
from .state import (
    Budget,
    HarnessState,
    Message,
    TokenCounter,
    ToolCall,
    default_token_counter,
)
from .tool import Tool


class BudgetExceeded(RuntimeError):
    pass


class Harness:
    def __init__(
        self,
        model: Model,
        components: Iterable[Component] = (),
        tools: Iterable[Tool] = (),
        *,
        system: str | None = None,
        budget: Budget | None = None,
        token_counter: TokenCounter = default_token_counter,
        max_turns: int = 20,
    ):
        self.model = model
        self.components: list[Component] = list(components)
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.system = system
        self.budget = budget or Budget()
        if self.budget.max_turns is None:
            self.budget.max_turns = max_turns
        self.token_counter = token_counter

    # -- public API ------------------------------------------------------

    def run(self, task: str | list[Message]) -> HarnessState:
        state = self._new_state(task)
        for c in self.components:
            c.on_start(state)

        while not state.done:
            if state.turn >= (self.budget.max_turns or 10**9):
                state.memory.setdefault("_stop_reason", "max_turns")
                break
            self._check_hard_budget(state)
            state.turn += 1

            for c in self.components:
                c.before_model(state)

            response = self.model(list(state.messages), list(state.tools.values()))
            state.usage.add(response.usage)
            state.last_response = response
            state.last_turn_had_tool_calls = bool(response.tool_calls)

            state.add_message(
                Message(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=list(response.tool_calls),
                )
            )
            for c in self.components:
                c.after_model(state, response)

            if response.tool_calls:
                for call in response.tool_calls:
                    result = self._dispatch(call)
                    for c in self.components:
                        result = c.after_tool(state, call, result)
                    state.add_message(
                        Message(
                            role="tool",
                            content=_as_text(result),
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )

            for c in self.components:
                c.after_turn(state)

            if not response.tool_calls:
                # candidate final answer — components may veto stopping
                if any(c.should_stop(state) is False for c in self.components):
                    continue
                state.done = True
                if state.result is None:
                    state.result = response.text
            elif any(c.should_stop(state) is True for c in self.components):
                state.done = True

        for c in self.components:
            c.on_end(state)
        return state

    # -- internals -------------------------------------------------------

    def _new_state(self, task: str | list[Message]) -> HarnessState:
        state = HarnessState(
            tools=dict(self.tools),
            budget=self.budget,
            token_counter=self.token_counter,
        )
        if self.system:
            state.add_message(Message(role="system", content=self.system))
        if isinstance(task, str):
            state.add_message(Message(role="user", content=task))
        else:
            state.messages.extend(task)
        return state

    def _dispatch(self, call: ToolCall) -> Any:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"[error: unknown tool {call.name!r}]"
        try:
            return tool(**call.arguments)
        except Exception as e:  # tools returning errors is normal agent flow
            return f"[error running {call.name}: {e}]"

    def _check_hard_budget(self, state: HarnessState) -> None:
        b = state.budget
        if b.max_total_tokens is not None and state.usage.total_tokens > b.max_total_tokens:
            raise BudgetExceeded(
                f"total tokens {state.usage.total_tokens} > {b.max_total_tokens}"
            )
        if b.max_cost is not None and state.usage.cost > b.max_cost:
            raise BudgetExceeded(f"cost {state.usage.cost} > {b.max_cost}")


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else repr(result)
