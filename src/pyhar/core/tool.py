"""Minimal, provider-agnostic tool wrapper.

v0 keeps schemas optional. The plan is to sit *on top of* MCP (import an MCP
server's tools as ``Tool`` objects) rather than reinvent the tool interface —
MCP already won that layer.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
) -> Any:
    """Turn a plain function into a ``Tool``. Usable bare or with arguments."""

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            fn=f,
            description=description or (f.__doc__ or "").strip(),
            schema=schema or {},
        )

    return wrap(fn) if fn is not None else wrap
