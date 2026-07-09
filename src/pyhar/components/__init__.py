"""Built-in harness primitives — the parts everyone re-implements by hand.

Every component here is also registered by its ``name`` in ``pyhar.registry``
(e.g. ``registry.get("compactor")`` -> ``Compactor``).
"""
from .. import registry as _registry
from .budget import BudgetPolicy
from .compactor import Compactor, default_preserve
from .context_builder import ContextBuilder
from .loop_guard import LoopGuard
from .memory import Memory
from .permissions import Permissions
from .state_artifact import FileStore, MemoryStore, StateArtifact
from .tool_budget import ToolOutputBudget
from .tracer import Tracer
from .verifier import Verifier

__all__ = [
    "Compactor",
    "default_preserve",
    "ToolOutputBudget",
    "Verifier",
    "BudgetPolicy",
    "ContextBuilder",
    "Memory",
    "StateArtifact",
    "MemoryStore",
    "FileStore",
    "Permissions",
    "Tracer",
    "LoopGuard",
]

for _cls in (
    Compactor,
    ToolOutputBudget,
    Verifier,
    BudgetPolicy,
    ContextBuilder,
    Memory,
    StateArtifact,
    Permissions,
    Tracer,
    LoopGuard,
):
    _registry.register()(_cls)
del _cls
