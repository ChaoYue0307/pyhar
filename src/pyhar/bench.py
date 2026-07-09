"""A/B two or more harness configs on the same task.

The composability thesis has to be *demonstrated*, not asserted: adopt one
primitive, keep your runtime, and show the win in tokens/cost/turns. ``bench``
runs each config from a fresh factory (so state never leaks) and reports usage.

Pass ``trials=N`` to run each config N times: the report then carries means
(tokens/cost/turns) plus a success rate, so noisy real-model comparisons don't
hinge on a single run.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .core.harness import Harness
from .core.state import HarnessState

SuccessFn = Callable[[HarnessState], bool]


@dataclass
class RunReport:
    """Aggregated result for one config. With ``trials=1`` (the default) the
    numbers are the single run's values; with more trials they are means, and
    ``*_std`` fields carry the standard deviation."""

    name: str
    success: bool                 # every trial succeeded
    turns: float
    input_tokens: float
    output_tokens: float
    cost: float
    result: Any                   # the LAST trial's result
    trials: int = 1
    success_rate: float = 1.0
    input_tokens_std: float = 0.0
    output_tokens_std: float = 0.0


@dataclass
class BenchReport:
    task: str
    runs: list[RunReport] = field(default_factory=list)

    def table(self) -> str:
        multi = any(r.trials > 1 for r in self.runs)
        header = f"{'config':<24}{'ok':<6}{'turns':<7}{'in_tok':<10}{'out_tok':<9}{'cost':<9}"
        if multi:
            header += f"{'trials':<8}"
        lines = [header, "-" * len(header)]
        for r in self.runs:
            ok = f"{r.success_rate:.0%}" if multi else ("yes" if r.success else "no")
            row = (
                f"{r.name:<24}{ok:<6}{r.turns:<7.1f}"
                f"{r.input_tokens:<10.0f}{r.output_tokens:<9.0f}{r.cost:<9.4f}"
            )
            if multi:
                row += f"{r.trials:<8}"
            lines.append(row)
        return "\n".join(lines)


def bench(
    task: str,
    configs: dict[str, Callable[[], Harness]],
    *,
    success: SuccessFn | None = None,
    trials: int = 1,
) -> BenchReport:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    report = BenchReport(task=task)
    for name, factory in configs.items():
        oks: list[bool] = []
        turns: list[int] = []
        in_toks: list[int] = []
        out_toks: list[int] = []
        costs: list[float] = []
        last_result: Any = None
        for _ in range(trials):
            state = factory().run(task)
            oks.append(success(state) if success else bool(state.done))
            turns.append(state.turn)
            in_toks.append(state.usage.input_tokens)
            out_toks.append(state.usage.output_tokens)
            costs.append(state.usage.cost)
            last_result = state.result
        report.runs.append(
            RunReport(
                name=name,
                success=all(oks),
                turns=statistics.fmean(turns),
                input_tokens=statistics.fmean(in_toks),
                output_tokens=statistics.fmean(out_toks),
                cost=statistics.fmean(costs),
                result=last_result,
                trials=trials,
                success_rate=sum(oks) / trials,
                input_tokens_std=statistics.pstdev(in_toks) if trials > 1 else 0.0,
                output_tokens_std=statistics.pstdev(out_toks) if trials > 1 else 0.0,
            )
        )
    return report
