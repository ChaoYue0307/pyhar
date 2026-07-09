"""Ollama backend for local / open-source models — zero dependencies.

Talks to a local Ollama server's native chat API over stdlib ``urllib`` (no SDK
required), so `OllamaModel("llama3.1")` works out of the box once you've run
`ollama pull llama3.1`. Tool calling requires a tool-capable model. Local =
free, so ``Usage.cost`` stays 0.
"""
from __future__ import annotations

import itertools
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
        transport: Any | None = None,          # callable(url, payload)->dict, for tests
        stream_transport: Any | None = None,   # callable(url, payload)->Iterator[dict], for tests
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.options = options or {}
        self._transport = transport or self._http_post
        self._stream_transport = stream_transport or self._http_post_lines

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        data = self._transport(f"{self.host}/api/chat", self._payload(messages, tools, stream=False))
        return _to_response(data)

    def stream(self, messages: list[Message], tools: list[Tool], *, on_delta) -> Response:
        """Streaming call over Ollama's NDJSON chat API: forwards content deltas
        to ``on_delta`` and assembles the complete ``Response``."""
        text_parts: list[str] = []
        tool_call_chunks: list[Any] = []
        final: dict[str, Any] = {}
        for data in self._stream_transport(
            f"{self.host}/api/chat", self._payload(messages, tools, stream=True)
        ):
            msg = data.get("message", {}) or {}
            content = msg.get("content")
            if content:
                text_parts.append(content)
                on_delta(content)
            if msg.get("tool_calls"):
                tool_call_chunks.extend(msg["tool_calls"])
            if data.get("done"):
                final = data
        assembled = dict(final)
        assembled["message"] = {
            "content": "".join(text_parts),
            "tool_calls": tool_call_chunks,
        }
        return _to_response(assembled)

    def _payload(self, messages: list[Message], tools: list[Tool], *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": stream,
        }
        if self.options:
            payload["options"] = self.options
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]
        return payload

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

    def _http_post_lines(self, url: str, payload: dict[str, Any]):  # pragma: no cover - network
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if line:
                        yield json.loads(line.decode("utf-8"))
        except urllib.error.URLError as e:
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


_call_ids = itertools.count()  # process-unique ids: Ollama's API has none, and
# reusing e.g. "ollama_0" across turns would collide (ToolOutputBudget keys its
# sandbox by call id, and OpenAI-compatible replays require unique ids)


def _to_response(data: dict[str, Any]) -> Response:
    msg = data.get("message", {}) or {}
    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        tool_calls.append(
            ToolCall(id=f"ollama_{next(_call_ids)}", name=fn.get("name", ""), arguments=args)
        )
    return Response(
        text=msg.get("content") or None,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            cost=0.0,
        ),
        stop_reason=data.get("done_reason"),  # e.g. stop/length
        raw=data,
    )
