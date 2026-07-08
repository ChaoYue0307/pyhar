"""A/B two or more harness configs on the same task.

The composability thesis has to be *demonstrated*, not asserted: adopt one
primitive, keep your runtime, and show the win in tokens/cost/turns. ``bench``
runs each config from a fresh factory (so state never leaks) and reports usage.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .core.harness import Harness
from .core.state import HarnessState

SuccessFn = Callable[[HarnessState], bool]


@dataclass
class RunReport:
    name: str
    success: bool
    turns: int
    input_tokens: int
    output_tokens: int
    cost: float
    result: Any


@dataclass
class BenchReport:
    task: str
    runs: list[RunReport] = field(default_factory=list)

    def table(self) -> str:
        header = f"{'config':<22}{'ok':<5}{'turns':<7}{'in_tok':<9}{'out_tok':<9}{'cost':<9}"
        lines = [header, "-" * len(header)]
        for r in self.runs:
            lines.append(
                f"{r.name:<22}{('yes' if r.success else 'no'):<5}{r.turns:<7}"
                f"{r.input_tokens:<9}{r.output_tokens:<9}{r.cost:<9.4f}"
            )
        return "\n".join(lines)


def bench(
    task: str,
    configs: dict[str, Callable[[], Harness]],
    *,
    success: SuccessFn | None = None,
) -> BenchReport:
    report = BenchReport(task=task)
    for name, factory in configs.items():
        state = factory().run(task)
        ok = success(state) if success else bool(state.done)
        report.runs.append(
            RunReport(
                name=name,
                success=ok,
                turns=state.turn,
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                cost=state.usage.cost,
                result=state.result,
            )
        )
    return report
