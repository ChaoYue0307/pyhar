"""OpenAI (and OpenAI-compatible) backend.

Lazy-imports the ``openai`` SDK (`pip install "pyhar-agents[openai]"`). Pass
``base_url`` to point at any OpenAI-compatible server — vLLM, Together, LM Studio,
Ollama's ``/v1`` endpoint, etc. — which is the common way to run OSS models.

This is a non-Anthropic provider adapter by design; pyhar is model-agnostic.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.model import Response
from ..core.state import Message, ToolCall, Usage
from ..core.tool import Tool
from .pricing import cost_of


class OpenAIModel:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any | None = None,
        pricing: tuple[float, float] | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.pricing = pricing
        self._client = client or _build_client(api_key, base_url)

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        resp = self._client.chat.completions.create(**kwargs)
        return self._to_response(resp)

    def _to_response(self, resp: Any) -> Response:
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        it = getattr(usage, "prompt_tokens", 0) if usage else 0
        ot = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = (
            (it * self.pricing[0] + ot * self.pricing[1]) / 1_000_000
            if self.pricing
            else cost_of(self.model, it, ot)
        )
        return Response(
            text=getattr(msg, "content", None),
            tool_calls=tool_calls,
            usage=Usage(input_tokens=it, output_tokens=ot, cost=cost),
            stop_reason=getattr(resp.choices[0], "finish_reason", None),  # e.g. stop/tool_calls/length
            raw=resp,
        )


def OpenAICompatibleModel(base_url: str, model: str, *, api_key: str = "not-needed", **kw) -> OpenAIModel:
    """Convenience for local / OSS OpenAI-compatible servers (vLLM, LM Studio, …)."""
    return OpenAIModel(model, base_url=base_url, api_key=api_key, **kw)


def _build_client(api_key: str | None, base_url: str | None) -> Any:
    try:
        import openai
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "OpenAIModel needs the openai SDK — `pip install 'pyhar-agents[openai]'`. "
            "Or pass client=... to inject your own."
        ) from e
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _to_openai_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema or {"type": "object", "properties": {}},
        },
    }


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content})
        else:  # system / user / plain assistant
            out.append({"role": m.role, "content": m.content})
    return out
