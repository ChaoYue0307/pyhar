"""Preset harnesses — different agents' harness *styles* as compositions.

This module is the thesis made concrete: a "minimal ReAct" agent and a
"Claude-Code-shaped coding agent" are not different frameworks, they are
different *compositions of the same shared parts*. Read these as recipes, then
mix your own.
"""
from __future__ import annotations

from collections.abc import Iterable

from .components.compactor import Compactor
from .components.loop_guard import LoopGuard
from .components.tool_budget import ToolOutputBudget
from .components.verifier import Check, Verifier
from .core.harness import Harness
from .core.model import Model
from .core.state import Budget
from .core.tool import Tool


def minimal_react(model: Model, tools: Iterable[Tool] = (), **kw) -> Harness:
    """The thinnest useful harness: the loop, nothing else."""
    return Harness(model, components=[], tools=tools, **kw)


def coding_agent(
    model: Model,
    tools: Iterable[Tool] = (),
    *,
    check: Check | None = None,
    context_tokens: int = 2000,
    tool_output_tokens: int = 300,
    max_turns: int = 20,
    **kw,
) -> Harness:
    """A Claude-Code-shaped harness: tool-output budgeting + staged compaction
    + a repeated-call loop guard (+ optional verify->retry). The whole agent's
    harness, as swappable parts."""
    components = [
        ToolOutputBudget(max_tokens=tool_output_tokens),
        Compactor(target_tokens=context_tokens),
        LoopGuard(),
    ]
    if check is not None:
        components.append(Verifier(check))
    budget = kw.pop(
        "budget",
        Budget(max_context_tokens=context_tokens, max_turns=max_turns),
    )
    return Harness(model, components=components, tools=tools, budget=budget, **kw)
