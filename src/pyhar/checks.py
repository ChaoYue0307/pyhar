"""Ready-made ``Verifier`` checks — including structured (JSON) final output.

A check is ``Callable[[HarnessState], tuple[bool, str]]``; these factories build
common ones so you don't hand-roll them:

    from pyhar import Verifier
    from pyhar.checks import json_schema_check, contains_check

    Verifier(json_schema_check({"type": "object", "required": ["answer"],
                                "properties": {"answer": {"type": "string"}}}))
    Verifier(contains_check("42"))

``json_schema_check`` validates ``state.result`` against a pragmatic JSON-Schema
subset (``type``, ``properties``, ``required``, ``items``, ``enum``,
``additionalProperties: false``) with zero dependencies, and tolerates answers
wrapped in ```` ```json ```` fences. On failure the feedback names the exact
violation, so the retry is targeted.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .components.verifier import Check
from .core.state import HarnessState

__all__ = ["contains_check", "regex_check", "json_schema_check", "parse_json_result"]


def _result_text(state: HarnessState) -> str:
    # The current candidate answer. Inside the loop, state.result IS the
    # candidate (set before after_turn) — including "" for an empty final turn,
    # which must FAIL checks rather than fall back to stale earlier text. The
    # fallback below only applies when result was never set (standalone use),
    # and never falls back to a tool-calling turn's commentary.
    if isinstance(state.result, str):
        return state.result
    return next(
        (
            m.content
            for m in reversed(state.messages)
            if m.role == "assistant" and m.content and not m.tool_calls
        ),
        "",
    )


def contains_check(*needles: str, case_sensitive: bool = False) -> Check:
    """Pass when the final answer contains every ``needle``."""

    def check(state: HarnessState) -> tuple[bool, str]:
        text = _result_text(state)
        hay = text if case_sensitive else text.lower()
        missing = [n for n in needles if (n if case_sensitive else n.lower()) not in hay]
        if missing:
            return False, f"the answer must mention: {', '.join(missing)}"
        return True, ""

    return check


def regex_check(pattern: str, *, flags: int = 0) -> Check:
    """Pass when the final answer matches ``pattern`` (``re.search``)."""
    compiled = re.compile(pattern, flags)

    def check(state: HarnessState) -> tuple[bool, str]:
        if compiled.search(_result_text(state)):
            return True, ""
        return False, f"the answer must match the pattern /{pattern}/"

    return check


def parse_json_result(state: HarnessState) -> Any:
    """Parse the final answer as JSON, tolerating ```json fences. Raises ValueError.

    The whole text is tried as JSON first (so valid JSON containing backticks is
    never mangled); only on failure are fenced blocks tried, last block first
    (the final fence is usually the real answer after discarded drafts).
    """
    text = _result_text(state).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        blocks = re.findall(r"^```(?:json)?\s*\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
        for block in reversed(blocks):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        raise ValueError(f"not valid JSON: {first_error}") from first_error


def json_schema_check(schema: dict[str, Any]) -> Check:
    """Pass when the final answer is valid JSON matching ``schema`` (subset).

    The schema is sanity-checked once at construction: an unrecognized type
    name (e.g. ``"int"`` instead of ``"integer"``) raises ``ValueError``
    immediately rather than silently validating nothing at run time.
    """
    _assert_valid_schema(schema, "$")

    def check(state: HarnessState) -> tuple[bool, str]:
        try:
            data = parse_json_result(state)
        except ValueError as e:
            return False, f"respond with valid JSON only — {e}"
        errors = _validate(data, schema, "$")
        if errors:
            return False, "fix the JSON: " + "; ".join(errors[:5])
        return True, ""

    return check


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _assert_valid_schema(schema: dict[str, Any], path: str) -> None:
    stype = schema.get("type")
    names = stype if isinstance(stype, list) else [stype] if stype is not None else []
    for name in names:
        if name not in _TYPE_MAP:
            raise ValueError(
                f"json_schema_check: unknown type {name!r} at {path} "
                f"(expected one of {sorted(_TYPE_MAP)})"
            )
    for key, sub in schema.get("properties", {}).items():
        if isinstance(sub, dict):
            _assert_valid_schema(sub, f"{path}.{key}")
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_valid_schema(items, f"{path}[]")


def _type_ok(data: Any, name: str) -> bool:
    expected = _TYPE_MAP.get(name)
    if expected is None:
        return True  # unknown names are rejected at construction; be lenient here
    if name in ("integer", "number") and isinstance(data, bool):
        return False  # bool is an int subclass in Python but not in JSON
    return isinstance(data, expected)


def _enum_match(data: Any, allowed: Any) -> bool:
    # JSON equality: bools never equal numbers (Python's True == 1 must not pass)
    if isinstance(data, bool) != isinstance(allowed, bool):
        return False
    return bool(data == allowed)


def _validate(data: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    stype = schema.get("type")
    if stype is not None:
        names = stype if isinstance(stype, list) else [stype]
        if not any(_type_ok(data, n) for n in names):
            label = stype if isinstance(stype, str) else " or ".join(names)
            errors.append(f"{path} must be {label}")
            return errors  # no point descending with the wrong type

    if "enum" in schema and not any(_enum_match(data, v) for v in schema["enum"]):
        errors.append(f"{path} must be one of {schema['enum']!r}")

    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}.{req} is required")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in data:
                errors.extend(_validate(data[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in props:
                    errors.append(f"{path}.{key} is not an allowed property")

    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(_validate(item, schema["items"], f"{path}[{i}]"))

    return errors
