"""EXPERIMENTAL — import MCP server tools as pyhar ``Tool`` objects.

MCP already won the tool-interface layer, so pyhar sits *on top of* it rather
than reinventing tools. Point this at a live MCP session's tool list and a
``call_tool`` callable and get back ``Tool`` objects you can hand to a ``Harness``
(and then budget with ``ToolOutputBudget``).

The pure adapter (``tools_from_mcp``) takes already-fetched descriptors, so it's
fully testable without a server. ``tools_from_mcp_session`` is an async
convenience for the official ``mcp`` SDK's ``ClientSession``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .core.tool import Tool

# call_tool(name, arguments) -> result text/obj
CallTool = Callable[[str, dict[str, Any]], Any]


def tool_from_mcp(descriptor: Any, call_tool: CallTool) -> Tool:
    """Wrap one MCP tool descriptor (``.name``/``.description``/``.inputSchema``)."""
    name = _attr(descriptor, "name")
    description = _attr(descriptor, "description", "") or ""
    schema = _attr(descriptor, "inputSchema", None) or _attr(descriptor, "input_schema", None) or {}

    def fn(**kwargs: Any) -> Any:
        return call_tool(name, kwargs)

    return Tool(name=name, fn=fn, description=description, schema=schema)


def tools_from_mcp(descriptors: Iterable[Any], call_tool: CallTool) -> list[Tool]:
    """Convert a list of MCP tool descriptors into pyhar tools."""
    return [tool_from_mcp(d, call_tool) for d in descriptors]


async def tools_from_mcp_session(session: Any) -> list[Tool]:  # pragma: no cover - needs mcp + server
    """Async convenience for the official ``mcp`` SDK ``ClientSession``.

        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await pyhar.mcp.tools_from_mcp_session(session)
    """
    listing = await session.list_tools()

    async def _acall(name: str, arguments: dict[str, Any]) -> Any:
        return await session.call_tool(name, arguments)

    # NOTE: pyhar's default loop is sync; wrap _acall for your async runtime,
    # or use tools_from_mcp with a sync bridge. Returned tools call the async fn.
    return tools_from_mcp(listing.tools, _acall)  # type: ignore[arg-type]


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
