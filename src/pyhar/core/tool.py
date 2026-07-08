"""Provider-agnostic tool wrapper with automatic JSON-schema generation.

``@tool`` builds a proper ``input_schema`` from the function's type hints so real
models (Anthropic/OpenAI/Ollama) actually see the tool's parameters. Pass an
explicit ``schema=`` to override, or import tools from an MCP server via
``pyhar.mcp`` — PyHarness sits on top of MCP rather than reinventing tools.
"""
from __future__ import annotations

import inspect
import typing
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
    """Turn a plain function into a ``Tool``. Usable bare or with arguments.

    The ``input_schema`` is derived from the function signature + type hints
    unless you pass ``schema=`` explicitly.

        @tool
        def read_file(path: str, max_bytes: int = 4096) -> str:
            '''Read a file.'''
            ...
        # -> schema: {type: object, properties: {path: {type: string},
        #             max_bytes: {type: integer}}, required: [path], ...}
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            fn=f,
            description=description or (f.__doc__ or "").strip(),
            schema=schema if schema is not None else schema_from_signature(f),
        )

    return wrap(fn) if fn is not None else wrap


# -- JSON-schema generation ------------------------------------------------

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Best-effort JSON Schema (object) from a function's parameters."""
    try:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
    except (ValueError, TypeError, NameError):
        return {"type": "object", "properties": {}}

    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[pname] = {"type": _json_type(hints.get(pname, str))}
        if p.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def _json_type(annotation: Any) -> str:
    # unwrap Optional[X] / Union[X, None] to X
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args:
            annotation = args[0]
            origin = typing.get_origin(annotation)
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    return _JSON_TYPES.get(annotation, "string")
