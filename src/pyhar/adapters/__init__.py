"""Runtime adapters — run pyhar components inside other agent runtimes.

``component_hooks`` is the pure core (always importable). The framework-specific
binders lazy-import their runtime, so this module imports cleanly with none of
them installed.
"""
from .manual import component_hooks

__all__ = ["component_hooks", "to_langgraph_middleware", "to_openai_agents_hooks"]


def __getattr__(name: str):  # lazy re-export so we don't import optional runtimes eagerly
    if name == "to_langgraph_middleware":
        from .langgraph import to_langgraph_middleware

        return to_langgraph_middleware
    if name == "to_openai_agents_hooks":
        from .openai_agents import to_openai_agents_hooks

        return to_openai_agents_hooks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
