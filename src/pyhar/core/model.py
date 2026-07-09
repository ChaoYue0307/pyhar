"""The model boundary.

pyhar never imports a provider SDK. A ``Model`` is anything that maps a
message list + tool specs to a ``Response``. Wrap Anthropic/OpenAI/a local model
yourself; use ``ScriptedModel`` for deterministic, key-free tests and examples.

Streaming is an optional extension: a model MAY also implement
``stream(messages, tools, *, on_delta)`` (and/or async ``astream``) — call
``on_delta(text_chunk)`` as text arrives and return the complete ``Response``.
A harness constructed with ``stream=True`` uses it when present.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .state import Message, ToolCall, Usage

OnDelta = Callable[[str], None]


@dataclass
class Response:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None  # provider stop/finish reason, normalized as-is
    raw: Any = None  # the untouched provider payload, if any


@runtime_checkable
class Model(Protocol):
    def __call__(self, messages: list[Message], tools: list[Any]) -> Response: ...


class ScriptedModel:
    """Returns queued responses in order — deterministic, no API key needed.

    Each script item is a ``Response`` or a shorthand:
      * ``"some text"``               -> a final text answer
      * ``("tool", name, {"a": 1})``  -> a single tool call
    Input tokens are attributed from the live context size at call time, so
    context-shrinking components produce visibly lower usage.
    """

    def __init__(self, script: list[Any], *, output_tokens: int = 20):
        self._script = list(script)
        self._i = 0
        self._out = output_tokens

    def __call__(self, messages: list[Message], tools: list[Any]) -> Response:
        if self._i >= len(self._script):
            return Response(text="", usage=Usage(input_tokens=_ctx(messages), output_tokens=1))
        item = self._script[self._i]
        self._i += 1
        resp = self._coerce(item)
        resp.usage.input_tokens = _ctx(messages)
        return resp

    def stream(self, messages: list[Message], tools: list[Any], *, on_delta: OnDelta) -> Response:
        """Streaming variant: emits the text answer in word-sized deltas, then
        returns the same complete ``Response`` as ``__call__`` would."""
        resp = self(messages, tools)
        if resp.text:
            for chunk in re.findall(r"\S+\s*", resp.text):
                on_delta(chunk)
        return resp

    def _coerce(self, item: Any) -> Response:
        if isinstance(item, Response):
            return item
        if isinstance(item, str):
            return Response(text=item, usage=Usage(output_tokens=self._out))
        if isinstance(item, tuple) and item and item[0] == "tool":
            _, name, args = item
            tc = ToolCall(id=f"call_{self._i}", name=name, arguments=args)
            return Response(tool_calls=[tc], usage=Usage(output_tokens=self._out))
        raise TypeError(f"cannot coerce script item: {item!r}")


def _ctx(messages: list[Message]) -> int:
    return sum(len(m.render()) // 4 for m in messages)
