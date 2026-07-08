from types import SimpleNamespace

from pyhar import Component, HarnessState, ToolCall
from pyhar.adapters import component_hooks
from pyhar.mcp import tools_from_mcp

# -- adapters: component_hooks --------------------------------------------

class _Recorder(Component):
    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def before_model(self, state):
        self.calls.append("before_model")

    def after_tool(self, state, call, result):
        self.calls.append("after_tool")
        return f"{result}+{self.tag}"

    def should_stop(self, state):
        return self.tag == "stopper"


def test_component_hooks_run_in_order_and_chain():
    a, b = _Recorder("a"), _Recorder("b")
    hooks = component_hooks([a, b])
    state = HarnessState()

    hooks["before_model"](state)
    assert a.calls == ["before_model"] and b.calls == ["before_model"]

    result = hooks["after_tool"](state, ToolCall(id="1", name="x"), "base")
    assert result == "base+a+b"  # chained through both, in order

    votes = hooks["should_stop"](state)
    assert votes == [False, False]


def test_component_hooks_should_stop_collects_votes():
    hooks = component_hooks([_Recorder("stopper"), _Recorder("b")])
    assert hooks["should_stop"](HarnessState()) == [True, False]


# -- mcp: tools_from_mcp ---------------------------------------------------

def test_tools_from_mcp_wraps_descriptors_and_dispatches():
    dispatched = {}

    def call_tool(name, arguments):
        dispatched[name] = arguments
        return f"ran {name}"

    # one object-style descriptor, one dict-style
    descriptors = [
        SimpleNamespace(name="search", description="web search", inputSchema={"type": "object"}),
        {"name": "read", "description": "read a file", "inputSchema": {"type": "object"}},
    ]
    tools = tools_from_mcp(descriptors, call_tool)

    assert [t.name for t in tools] == ["search", "read"]
    assert tools[0].schema == {"type": "object"}
    # calling the wrapped tool dispatches through call_tool with kwargs
    assert tools[1](path="x") == "ran read"
    assert dispatched["read"] == {"path": "x"}
