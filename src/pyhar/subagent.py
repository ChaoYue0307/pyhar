"""Subagent spawning with context isolation, exposed as a tool.

A subagent is just another ``Harness`` with its *own* context window. The
elegant way to relate it to the parent is to expose it as a ``Tool``: the parent
model calls it with a task string, the sub-harness runs in isolation, and only a
relevant excerpt (its result) comes back — the "return-only-relevant-excerpts"
contract, decoupled from any runtime.

    tools = [subagent_tool("research", build_research_harness)]
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .core.harness import Harness
from .core.tool import Tool

BuildHarness = Callable[[], Harness]


def spawn(harness: Harness, task: str, *, excerpt: Callable[[Any], str] | None = None) -> str:
    """Run a sub-harness on ``task`` in isolation and return an excerpt of its result.

    ``state.result`` is only set on a clean finishing turn; if the subagent
    instead exhausts its turn budget (or is stopped mid-tool), we fall back to
    its last assistant text rather than returning the string ``"None"``.
    """
    state = harness.run(task)
    if excerpt is not None:
        return excerpt(state)
    if isinstance(state.result, str):
        return state.result
    if state.result is not None:
        return str(state.result)
    last = next(
        (m.content for m in reversed(state.messages) if m.role == "assistant" and m.content),
        None,
    )
    if last:
        return last
    reason = state.memory.get("_stop_reason", "no result produced")
    return f"[subagent ended without a result: {reason}]"


def subagent_tool(
    name: str,
    build_harness: BuildHarness,
    *,
    description: str = "Delegate a self-contained subtask to an isolated sub-agent.",
    task_field: str = "task",
) -> Tool:
    """Expose a fresh isolated sub-harness as a callable tool.

    ``build_harness`` is invoked per call so each subagent gets a clean context
    (isolation); only its final result is returned to the parent.
    """

    def fn(**kwargs: Any) -> str:
        task = kwargs.get(task_field, "")
        return spawn(build_harness(), task)

    return Tool(
        name=name,
        fn=fn,
        description=description,
        schema={
            "type": "object",
            "properties": {task_field: {"type": "string", "description": "the subtask to perform"}},
            "required": [task_field],
        },
    )
