"""Async agent: async model + async/sync tools, parallel tool execution.

`AsyncHarness` awaits async models and tools; sync ones are offloaded to a
thread so they never block the event loop. Every existing component works
unchanged. With `parallel_tools=True`, parallel tool calls run concurrently.

Run:  python examples/async_agent.py
"""
import asyncio
import time

from pyhar import AsyncHarness, Response, ScriptedModel, ToolCall, Tracer, tool


async def fetch_page(url: str) -> str:
    """An async tool — e.g. an aiohttp call in real life."""
    await asyncio.sleep(0.2)  # simulated network latency
    return f"<html>content of {url}</html>"


@tool
def word_count(text: str) -> str:
    """A plain sync tool — auto-offloaded to a thread."""
    return f"{len(text.split())} words"


fetch = tool(fetch_page, name="fetch_page")

# One turn issues TWO fetches in parallel, then a sync tool, then answers.
parallel_turn = Response(tool_calls=[
    ToolCall(id="a", name="fetch_page", arguments={"url": "https://a.example"}),
    ToolCall(id="b", name="fetch_page", arguments={"url": "https://b.example"}),
])
script = ScriptedModel([
    parallel_turn,
    ("tool", "word_count", {"text": "the quick brown fox"}),
    "Fetched both pages (in parallel) and counted 4 words.",
])


async def model(messages, tools):
    """An async Model — e.g. an async SDK client in real life."""
    return script(messages, tools)


async def main() -> None:
    harness = AsyncHarness(
        model,
        components=[Tracer(sink=lambda e: print("  trace:", e["event"], e.get("name", "")))],
        tools=[fetch, word_count],
        parallel_tools=True,
    )
    t0 = time.monotonic()
    state = await harness.arun("Fetch a.example and b.example, then count words.")
    print(f"\nresult:  {state.result}")
    print(f"elapsed: {time.monotonic() - t0:.2f}s  (two 0.2s fetches ran concurrently)")


if __name__ == "__main__":
    asyncio.run(main())
