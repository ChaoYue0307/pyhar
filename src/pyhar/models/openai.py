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
        resp = self._client.chat.completions.create(**self._request_kwargs(messages, tools))
        return self._to_response(resp)

    def stream(self, messages: list[Message], tools: list[Tool], *, on_delta) -> Response:
        """Streaming call: forwards content deltas to ``on_delta``, accumulates
        tool calls split across chunks, and returns the complete ``Response``.

        ``stream_options={"include_usage": True}`` is requested so the final
        chunk carries usage; some OpenAI-compatible servers ignore or reject it,
        in which case usage falls back to zeros.
        """
        kwargs = self._request_kwargs(messages, tools)
        kwargs["stream"] = True
        try:
            chunks = self._client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except TypeError:  # pragma: no cover - pre-stream_options SDKs
            chunks = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # servers that 400 on the unknown field
            if type(e).__name__ not in ("BadRequestError", "UnprocessableEntityError"):
                raise
            chunks = self._client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        finish_reason: str | None = None
        usage_obj: Any = None
        # tool calls assembled across chunks; dict preserves arrival order.
        # keyed by index when present, else by id, else a running counter.
        acc: dict[Any, dict[str, Any]] = {}
        last_key: Any = None
        fallback_seq = 0
        for chunk in chunks:
            if getattr(chunk, "usage", None) is not None:
                usage_obj = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                on_delta(content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", None)
                tc_id = getattr(tc, "id", None)
                if isinstance(idx, int):
                    key: Any = ("i", idx)
                elif tc_id:
                    key = ("id", tc_id)
                elif last_key is not None:
                    key = last_key  # continuation of the previous slot
                else:
                    key = ("n", fallback_seq)
                    fallback_seq += 1
                last_key = key
                slot = acc.setdefault(key, {"id": "", "name": "", "arguments": ""})
                if tc_id:
                    slot["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        tool_calls: list[ToolCall] = []
        for seq, slot in enumerate(acc.values()):  # arrival order
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": slot["arguments"]}
            tool_calls.append(
                ToolCall(id=slot["id"] or f"call_{seq}", name=slot["name"], arguments=args)
            )

        it = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
        ot = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
        cost = (
            (it * self.pricing[0] + ot * self.pricing[1]) / 1_000_000
            if self.pricing
            else cost_of(self.model, it, ot)
        )
        return Response(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=Usage(input_tokens=it, output_tokens=ot, cost=cost),
            stop_reason=finish_reason,
            raw=None,
        )

    def _request_kwargs(self, messages: list[Message], tools: list[Tool]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        return kwargs

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
