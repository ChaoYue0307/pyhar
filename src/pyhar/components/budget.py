"""Explicit token/cost ceilings and a soft-warning hook.

Cost control is a first-class concern in production harnesses (Devin's
frontier+sidekick tiering, extended-thinking budgets) yet is almost never a
named primitive. ``BudgetPolicy`` makes it one. Actual model-tiering (swap to a
cheaper model past a threshold) is surfaced via ``on_over_soft`` — wiring the
swap lives in your model wrapper in v0.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.harness import BudgetExceeded
from ..core.state import HarnessState


class BudgetPolicy(Component):
    name = "budget_policy"

    def __init__(
        self,
        *,
        max_cost: float | None = None,
        max_total_tokens: int | None = None,
        soft_fraction: float = 0.8,
        on_over_soft: Callable[[HarnessState], None] | None = None,
    ):
        self.max_cost = max_cost
        self.max_total_tokens = max_total_tokens
        self.soft_fraction = soft_fraction
        self.on_over_soft = on_over_soft
        self._warned = False

    def after_turn(self, state: HarnessState) -> None:
        u = state.usage
        if self.max_total_tokens and u.total_tokens > self.max_total_tokens:
            raise BudgetExceeded(f"tokens {u.total_tokens} > {self.max_total_tokens}")
        if self.max_cost and u.cost > self.max_cost:
            raise BudgetExceeded(f"cost {u.cost:.4f} > {self.max_cost}")
        if (
            self.on_over_soft
            and not self._warned
            and self.max_total_tokens
            and u.total_tokens > self.soft_fraction * self.max_total_tokens
        ):
            self._warned = True
            self.on_over_soft(state)
