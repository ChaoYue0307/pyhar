"""Declarative harnesses — build a ``Harness`` from a JSON-able config.

This is the shareable half of the ecosystem story: a harness composition is
just data, so it can live in JSON/YAML, be checked into a repo, or be published
alongside a component package::

    config = {
        "system": "You are a careful coding agent.",
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": 300}},
            {"name": "compactor", "args": {"target_tokens": 2000}},
            "loop_guard",
            "tracer",
        ],
        "budget": {"max_context_tokens": 2000, "max_turns": 15},
        "parallel_tools": True,
    }
    harness = harness_from_config(config, model=model, tools=[...])

Component names resolve through ``pyhar.registry`` — built-ins are always
available, and third-party components join via ``@registry.register()`` or the
``pyhar.components`` entry-point group (call ``registry.load_entrypoints()``
first, or pass ``load_entrypoints=True``).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import registry
from .core.harness import Harness
from .core.model import Model
from .core.state import Budget
from .core.tool import Tool

_HARNESS_KEYS = ("system", "max_turns", "parallel_tools", "stream")


def components_from_config(specs: list[Any]) -> list[Any]:
    """Build the component list for a config's ``components`` entry."""
    return registry.build(specs)


def harness_from_config(
    config: dict[str, Any],
    *,
    model: Model,
    tools: Iterable[Tool] = (),
    load_entrypoints: bool = False,
    harness_cls: type[Harness] = Harness,
    **overrides: Any,
) -> Harness:
    """Build a ``Harness`` (or ``AsyncHarness`` via ``harness_cls``) from a
    JSON-able config. Keyword ``overrides`` win over config values.

    Recognized config keys: ``components`` (see ``registry.build``), ``budget``
    (a dict of ``Budget`` fields), and ``system`` / ``max_turns`` /
    ``parallel_tools`` / ``stream``. Unknown keys raise ``ValueError`` so typos
    fail loudly instead of silently configuring nothing.
    """
    known = set(_HARNESS_KEYS) | {"components", "budget"}
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)} (known: {sorted(known)})")

    if load_entrypoints:
        registry.load_entrypoints()

    kwargs: dict[str, Any] = {k: config[k] for k in _HARNESS_KEYS if k in config}
    if "budget" in config:
        kwargs["budget"] = dict(config["budget"])
    kwargs.update(overrides)

    # coerce AFTER the override merge so budget={...} works as an override too
    if isinstance(kwargs.get("budget"), dict):
        kwargs["budget"] = Budget(**kwargs["budget"])

    # documented contract: an explicit max_turns (overrides > config) wins over
    # a budget's max_turns — Harness itself only fills max_turns when the
    # budget leaves it None, so apply the precedence here
    if "max_turns" in kwargs and isinstance(kwargs.get("budget"), Budget):
        kwargs["budget"].max_turns = kwargs.pop("max_turns")

    return harness_cls(
        model,
        components=components_from_config(config.get("components", [])),
        tools=tools,
        **kwargs,
    )
