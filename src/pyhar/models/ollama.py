"""Ollama backend for local / open-source models — zero dependencies.

Talks to a local Ollama server's native chat API over stdlib ``urllib`` (no SDK
required), so `OllamaModel("llama3.1")` works out of the box once you've run
`ollama pull llama3.1`. Tool calling requires a tool-capable model. Local =
free, so ``Usage.cost`` stays 0.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..core.model import Response
from ..core.state import Message, ToolCall, Usage
from ..core.tool import Tool


class OllamaModel:
    def __init__(
        self,
        model: str = "llama3.1",
        *,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        options: dict[str, Any] | None = None,
        transport: Any | None = None,   # injectable callable(url, payload)->dict, for tests
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.options = options or {}
        self._transport = transport or self._http_post

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": False,
        }
        if self.options:
            payload["options"] = self.options
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]

        data = self._transport(f"{self.host}/api/chat", payload)
        return _to_response(data)

    def _http_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"could not reach Ollama at {self.host} ({e}); is `ollama serve` running?"
            ) from e


def _to_ollama_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema or {"type": "object", "properties": {}},
        },
    }


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_response(data: dict[str, Any]) -> Response:
    msg = data.get("message", {}) or {}
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        tool_calls.append(ToolCall(id=f"ollama_{i}", name=fn.get("name", ""), arguments=args))
    return Response(
        text=msg.get("content") or None,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            cost=0.0,
        ),
        raw=data,
    )
