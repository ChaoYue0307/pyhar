"""Experimental — a tiny name registry for components.

The long-term dream is a torchvision/timm-style ecosystem where a compactor or
verifier written once is reused across projects. That only pays off at adoption
critical mass, so v0 ships just the seed: register by name, look up, list.
Do not build the ecosystem story on this yet.
"""
from __future__ import annotations

from .core.component import Component

_REGISTRY: dict[str, type[Component]] = {}


def register(name: str | None = None):
    def wrap(cls: type[Component]) -> type[Component]:
        key = str(name or getattr(cls, "name", cls.__name__))
        _REGISTRY[key] = cls
        return cls

    return wrap


def get(name: str) -> type[Component]:
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
