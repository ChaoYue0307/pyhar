"""Integration tests: pyhar components inside a REAL LangChain 1.x agent.

These run only when langchain is installed (`pip install 'pyhar-agents[langgraph]'`);
core CI stays zero-dependency. They exercise create_agent end-to-end with a fake
chat model, proving the middleware hook mapping against the pinned API.
"""
import pytest

langchain = pytest.importorskip("langchain", reason="langgraph extra not installed")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool as lc_tool  # noqa: E402

from pyhar import Permissions, ToolOutputBudget, Tracer  # noqa: E402
from pyhar.adapters import to_langgraph_middleware  # noqa: E402


@lc_tool
def read_file(path: str) -> str:
    """Read a file (returns a deliberately huge blob)."""
    return "x" * 8000


@lc_tool
def delete_everything(path: str) -> str:
    """Destructive tool that must never run."""
    raise AssertionError("delete_everything executed — Permissions failed")


class ToolCallingFakeModel(GenericFakeChatModel):
    """GenericFakeChatModel that tolerates create_agent's bind_tools call."""

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_model(script):
    return ToolCallingFakeModel(messages=iter(script))


def test_tool_output_budget_shrinks_inside_create_agent():
    mw = to_langgraph_middleware([ToolOutputBudget(max_tokens=50), Tracer()])
    model = _fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "big.txt"}, "id": "c1"},
        ]),
        AIMessage(content="summarized the file"),
    ])
    agent = create_agent(model, tools=[read_file], middleware=[mw])
    result = agent.invoke({"messages": [("user", "read big.txt")]})

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "no ToolMessage in the transcript"
    assert len(tool_msgs[0].content) < 8000          # shrunk by ToolOutputBudget
    assert mw.pyhar_state.memory["_tool_savings"] > 0
    assert mw.pyhar_state.memory["_sandbox"]         # full output preserved
    events = [e["event"] for e in mw.pyhar_state.memory["_trace"]]
    assert "tool_call" in events and "tool_result" in events


def test_permissions_deny_blocks_execution_inside_create_agent():
    mw = to_langgraph_middleware([Permissions(deny=["delete_everything"])])
    model = _fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "delete_everything", "args": {"path": "/"}, "id": "c1"},
        ]),
        AIMessage(content="okay, I won't delete anything"),
    ])
    agent = create_agent(model, tools=[delete_everything], middleware=[mw])
    result = agent.invoke({"messages": [("user", "clean up")]})

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and "permission denied" in tool_msgs[0].content
    assert mw.pyhar_state.memory["_denied"][0]["tool"] == "delete_everything"
    # and the lc_tool body never ran (it raises AssertionError if executed)


def test_middleware_fires_on_start_and_on_end_per_invoke():
    tracer = Tracer()
    mw = to_langgraph_middleware([tracer])
    model = _fake_model([AIMessage(content="done, no tools")])
    agent = create_agent(model, tools=[read_file], middleware=[mw])
    agent.invoke({"messages": [("user", "hi")]})

    events = [e["event"] for e in mw.pyhar_state.memory["_trace"]]
    assert events[0] == "start"    # before_agent -> on_start
    assert events[-1] == "end"     # after_agent -> on_end


def test_async_ainvoke_with_tool_calls_works():
    # regression: overriding only sync wrap_tool_call made ainvoke raise
    # NotImplementedError from the base awrap_tool_call
    import asyncio

    mw = to_langgraph_middleware([ToolOutputBudget(max_tokens=50)])
    model = _fake_model([
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "big.txt"}, "id": "c1"},
        ]),
        AIMessage(content="done"),
    ])
    agent = create_agent(model, tools=[read_file], middleware=[mw])
    result = asyncio.run(agent.ainvoke({"messages": [("user", "read it")]}))

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and len(tool_msgs[0].content) < 8000
    assert mw.pyhar_state.memory["_tool_savings"] > 0


def test_message_channel_components_rejected_up_front():
    from pyhar import Compactor, Memory, Verifier

    for comp in (Compactor(target_tokens=10), Memory(), Verifier(lambda s: (True, ""))):
        with pytest.raises(ValueError, match="message channel"):
            to_langgraph_middleware([comp])


def test_two_middleware_instances_coexist():
    mw1 = to_langgraph_middleware([Tracer()])
    mw2 = to_langgraph_middleware([Permissions(deny=["nothing"])])
    model = _fake_model([AIMessage(content="hi")])
    agent = create_agent(model, tools=[read_file], middleware=[mw1, mw2])
    agent.invoke({"messages": [("user", "hello")]})  # no duplicate-name error


def test_loop_guard_counters_reset_across_invokes():
    from pyhar import LoopGuard

    guard = LoopGuard(max_repeats=2, max_total_repeats=3)
    mw = to_langgraph_middleware([guard])

    def two_identical_calls_then_answer():
        return _fake_model([
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "a"}, "id": "c1"},
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "a"}, "id": "c2"},
            ]),
            AIMessage(content="done"),
        ])

    for _ in range(2):  # two invokes; counters must reset between them
        agent = create_agent(
            two_identical_calls_then_answer(), tools=[read_file], middleware=[mw]
        )
        agent.invoke({"messages": [("user", "go")]})

    # without the before_agent -> on_start reset, the second invoke would trip
    # max_total_repeats and record a denial
    assert "_loop_guard" not in mw.pyhar_state.memory
