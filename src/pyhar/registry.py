"""A name registry for components — the seed of the ecosystem story.

Every built-in component registers itself by its ``name`` (see
``pyhar.components``), and third-party packages can join two ways:

1. **Decorate**: ``@registry.register()`` on any ``Component`` subclass.
2. **Entry points**: declare components in your package metadata and they are
   discovered by ``registry.load_entrypoints()`` — no import of your package
   needed by the user::

       # pyproject.toml of a third-party package
       [project.entry-points."pyhar.components"]
       my_compactor = "my_pkg.components:MyCompactor"

Once a component is registered it can be looked up (``get``), instantiated
(``create``), or built in bulk from a JSON-able spec (``build``) — which is what
``pyhar.harness_from_config`` uses to turn a shareable config into a harness.
"""
from __future__ import annotations

from typing import Any

from .core.component import Component

_REGISTRY: dict[str, type[Component]] = {}

ENTRYPOINT_GROUP = "pyhar.components"


def register(name: str | None = None):
    def wrap(cls: type[Component]) -> type[Component]:
        key = str(name or getattr(cls, "name", cls.__name__))
        _REGISTRY[key] = cls
        return cls

    return wrap


def get(name: str) -> type[Component]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no component registered as {name!r} (available: {', '.join(available())}). "
            f"Third-party components may need registry.load_entrypoints() first."
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def create(name: str, **kwargs: Any) -> Component:
    """Instantiate a registered component by name: ``create("compactor", target_tokens=800)``."""
    return get(name)(**kwargs)


def build(specs: list[Any]) -> list[Component]:
    """Build a component list from a JSON-able spec.

    Each entry is either a bare name (``"tracer"``) or a dict with ``name`` and
    optional ``args``: ``{"name": "compactor", "args": {"target_tokens": 800}}``.
    """
    components: list[Component] = []
    for spec in specs:
        if isinstance(spec, str):
            components.append(create(spec))
        elif isinstance(spec, dict) and "name" in spec:
            components.append(create(spec["name"], **spec.get("args", {})))
        else:
            raise ValueError(
                f"bad component spec {spec!r} — expected 'name' or {{'name': ..., 'args': {{...}}}}"
            )
    return components


def load_entrypoints(group: str = ENTRYPOINT_GROUP, *, allow_override: bool = False) -> list[str]:
    """Discover and register components declared by installed third-party
    packages under the ``pyhar.components`` entry-point group. Returns the
    names registered. Safe to call repeatedly.

    An entry point whose name collides with an already-registered component
    (e.g. a built-in) is SKIPPED with a warning unless ``allow_override=True``
    — an installed package must not silently replace ``"compactor"``. Broken
    entries are skipped with a warning naming the cause and recorded in the
    returned list as ``"!name"``.
    """
    import warnings
    from importlib.metadata import entry_points

    loaded: list[str] = []
    for ep in entry_points(group=group):
        try:
            cls = ep.load()
        except Exception as e:  # a broken third-party package must not break pyhar
            warnings.warn(
                f"pyhar.registry: failed to load entry point {ep.name!r} "
                f"({getattr(ep, 'value', '?')}): {e!r}",
                stacklevel=2,
            )
            loaded.append(f"!{ep.name}")
            continue
        existing = _REGISTRY.get(ep.name)
        if existing is not None and existing is not cls and not allow_override:
            warnings.warn(
                f"pyhar.registry: entry point {ep.name!r} collides with the already-"
                f"registered {existing.__module__}.{existing.__qualname__}; skipping "
                f"(pass allow_override=True to accept replacements)",
                stacklevel=2,
            )
            loaded.append(f"!{ep.name}")
            continue
        register(ep.name)(cls)
        loaded.append(ep.name)
    return loaded
