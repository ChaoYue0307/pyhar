from pyhar import Harness, ScriptedModel, tool


def test_harness_runs_to_completion():
    model = ScriptedModel(["all done"])
    state = Harness(model).run("hi")
    assert state.done is True
    assert state.result == "all done"
    assert state.turn == 1


def test_tool_dispatch_and_loop():
    calls = []

    @tool
    def ping(x: int) -> str:
        calls.append(x)
        return f"pong{x}"

    model = ScriptedModel([("tool", "ping", {"x": 7}), "finished"])
    state = Harness(model, tools=[ping]).run("go")

    assert calls == [7]
    assert state.done and state.result == "finished"
    tool_msgs = [m for m in state.messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].content == "pong7"


def test_unknown_tool_is_reported_not_raised():
    model = ScriptedModel([("tool", "nope", {}), "ok"])
    state = Harness(model).run("go")
    tool_msg = [m for m in state.messages if m.role == "tool"][0]
    assert "unknown tool" in tool_msg.content
    assert state.done


def test_max_turns_stops_the_loop():
    # a model that always calls a tool would loop forever without a cap
    model = ScriptedModel([("tool", "spin", {})] * 100)

    @tool
    def spin() -> str:
        return "again"

    state = Harness(model, tools=[spin], max_turns=3).run("go")
    assert state.turn == 3
    assert state.memory.get("_stop_reason") == "max_turns"
