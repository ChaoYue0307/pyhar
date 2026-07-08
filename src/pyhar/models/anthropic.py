"""Anthropic Claude backend (official SDK).

Lazy-imports the ``anthropic`` SDK so ``import pyhar`` never requires it —
the ImportError is raised only if you actually construct an ``AnthropicModel``
without it installed (``pip install "pyhar-agents[anthropic]"``).

Follows the current Messages API: adaptive thinking via ``thinking={"type":
"adaptive"}`` (never ``budget_tokens``), depth via ``output_config.effort``, and
no ``temperature``/``top_p``/``top_k`` (rejected on Opus 4.8 / Sonnet 5 / Fable 5).
Defaults to ``claude-opus-4-8``.
"""
from __future__ import annotations

from typing import Any

from ..core.model import Response
from ..core.state import Message, ToolCall, Usage
from ..core.tool import Tool
from .pricing import cost_of

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 4096,
        system: str | None = None,
        thinking: dict[str, Any] | None = None,   # e.g. {"type": "adaptive"}
        effort: str | None = None,                # "low" | "medium" | "high" | "xhigh" | "max"
        api_key: str | None = None,
        client: Any | None = None,                # inject a client (or a fake, for tests)
        pricing: tuple[float, float] | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.thinking = thinking
        self.effort = effort
        self.pricing = pricing
        self._client = client or _build_client(api_key)

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        system, converted = _to_anthropic_messages(messages, self.system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}

        resp = self._client.messages.create(**kwargs)
        return self._to_response(resp)

    def _to_response(self, resp: Any) -> Response:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        it = getattr(resp.usage, "input_tokens", 0)
        ot = getattr(resp.usage, "output_tokens", 0)
        cost = (
            (it * self.pricing[0] + ot * self.pricing[1]) / 1_000_000
            if self.pricing
            else cost_of(self.model, it, ot)
        )
        return Response(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=Usage(input_tokens=it, output_tokens=ot, cost=cost),
            raw=resp,
        )


def _build_client(api_key: str | None) -> Any:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "AnthropicModel needs the anthropic SDK — `pip install 'pyhar-agents[anthropic]'`. "
            "Or pass client=... to inject your own."
        ) from e
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _to_anthropic_tool(tool: Tool) -> dict[str, Any]:
    schema = tool.schema or {"type": "object", "properties": {}}
    return {"name": tool.name, "description": tool.description, "input_schema": schema}


def _to_anthropic_messages(
    messages: list[Message], extra_system: str | None
) -> tuple[str, list[dict[str, Any]]]:
    """Convert pyhar messages to (system_str, anthropic_messages).

    system messages are hoisted into the top-level ``system`` string; ``tool``
    messages become ``tool_result`` blocks inside a user message; assistant
    tool calls become ``tool_use`` blocks.
    """
    system_parts: list[str] = []
    if extra_system:
        system_parts.append(extra_system)
    out: list[dict[str, Any]] = []

    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.role == "system":
            system_parts.append(m.content)
            i += 1
        elif m.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.content}]})
            i += 1
        elif m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            # skip an empty assistant turn — the API rejects empty text blocks
            if content:
                out.append({"role": "assistant", "content": content})
            i += 1
        elif m.role == "tool":
            # coalesce a run of tool messages into ONE user message with all
            # tool_result blocks (the API requires parallel results grouped)
            blocks: list[dict[str, Any]] = []
            while i < n and messages[i].role == "tool":
                tm = messages[i]
                blocks.append(
                    {"type": "tool_result", "tool_use_id": tm.tool_call_id or "", "content": tm.content}
                )
                i += 1
            out.append({"role": "user", "content": blocks})
        else:
            i += 1
    return "\n\n".join(system_parts), out
