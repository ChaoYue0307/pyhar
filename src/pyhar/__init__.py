"""pyhar — composable primitives for agent harnesses.

Not another agent framework. pyhar gives you the model-facing scaffolding —
compaction, tool-output budgeting, verification, context assembly, budgets — as
small, typed, swappable parts that all implement one interface (``Component``)
and drop into *any* loop. Bring your own runtime; bring your own tools (MCP).

    from pyhar import Harness, ScriptedModel, tool
    from pyhar.presets import coding_agent

    harness = coding_agent(model, tools=[...])
    state = harness.run("do the thing")
"""
from . import adapters, models, presets, registry
from .bench import BenchReport, RunReport, bench
from .components import (
    BudgetPolicy,
    Compactor,
    ContextBuilder,
    FileStore,
    Memory,
    MemoryStore,
    StateArtifact,
    ToolOutputBudget,
    Verifier,
)
from .core import (
    Budget,
    BudgetExceeded,
    Component,
    Harness,
    HarnessState,
    Message,
    Model,
    Response,
    ScriptedModel,
    Tool,
    ToolCall,
    Usage,
    default_token_counter,
    tool,
)
from .subagent import spawn, subagent_tool

__version__ = "0.1.0"

__all__ = [
    # core
    "Component",
    "Harness",
    "BudgetExceeded",
    "Model",
    "Response",
    "ScriptedModel",
    "Budget",
    "HarnessState",
    "Message",
    "ToolCall",
    "Usage",
    "Tool",
    "tool",
    "default_token_counter",
    # components
    "Compactor",
    "ToolOutputBudget",
    "Verifier",
    "BudgetPolicy",
    "ContextBuilder",
    "Memory",
    "StateArtifact",
    "MemoryStore",
    "FileStore",
    # subagents
    "spawn",
    "subagent_tool",
    # bench / presets / registry / models / adapters
    "bench",
    "BenchReport",
    "RunReport",
    "presets",
    "registry",
    "models",
    "adapters",
]
