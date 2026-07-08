"""First-class verify -> retry, driven by *your* check.

Unlike schema validation (Pydantic AI) or IO guardrails (OpenAI SDK), the check
is arbitrary: run tests, an eval, an LLM judge, a real end-to-end drive. When the
model produces a candidate final answer (a turn with no tool calls), the check
runs; on failure it injects feedback and re-opens the task (votes against
stopping) up to ``max_retries``.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.state import HarnessState, Message

# A check returns (passed, feedback_when_failing)
Check = Callable[[HarnessState], tuple[bool, str]]


class Verifier(Component):
    name = "verifier"

    def __init__(self, check: Check, *, max_retries: int = 2):
        self.check = check
        self.max_retries = max_retries
        self._retries = 0
        self._reopen = False

    def after_turn(self, state: HarnessState) -> None:
        self._reopen = False
        if state.last_turn_had_tool_calls:
            return

        passed, feedback = self.check(state)
        state.memory["_verified"] = passed
        if not passed and self._retries < self.max_retries:
            self._retries += 1
            self._reopen = True
            state.add_message(
                Message(
                    role="user",
                    content=f"[verification failed — retry {self._retries}/{self.max_retries}] {feedback}",
                    meta={"verifier": True},
                )
            )

    def should_stop(self, state: HarnessState) -> bool | None:
        return False if self._reopen else None
