"""A trivial deterministic model — no API key, no dependency.

Useful as a smoke-test backend and as a template for writing your own ``Model``.
For scripted multi-turn behavior use ``pyhar.ScriptedModel`` instead.
"""
from __future__ import annotations

from typing import Any

from ..core.model import Response
from ..core.state import Message, Usage


class EchoModel:
    """Returns a canned reply that echoes the last user message. Never calls tools."""

    def __init__(self, prefix: str = "ok: "):
        self.prefix = prefix

    def __call__(self, messages: list[Message], tools: list[Any]) -> Response:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        text = f"{self.prefix}{last_user}"
        return Response(
            text=text,
            usage=Usage(
                input_tokens=sum(len(m.render()) // 4 for m in messages),
                output_tokens=len(text) // 4,
            ),
        )
