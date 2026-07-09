"""Tests for the 0.3.0 round: AsyncHarness, combinators, LoopGuard, checks,
parallel tools, bench trials, stop_reason, registry auto-registration."""
import asyncio
import time

import pytest

from pyhar import (
    AsyncHarness,
    Harness,
    LoopGuard,
    Message,
    Permissions,
    Response,
    ScriptedModel,
    Usage,
    Verifier,
    bench,
    registry,
    tool,
)
from pyhar.checks import contains_check, json_schema_check, parse_json_result, regex_check
from pyhar.core.state import HarnessState
from pyhar.models import FallbackModel, RetryModel, RouterModel
from pyhar.presets import coding_agent

# -- AsyncHarness ------------------------------------------------------------

def test_async_harness_with_async_model_and_mixed_tools():
    calls = []

    @tool
    def sync_tool(x: int) -> str:
        calls.append(("sync", x))
        return f"sync:{x}"

    async def async_fn(y: str) -> str:
        await asyncio.sleep(0)
        calls.append(("async", y))
        return f"async:{y}"

    async_tool = tool(async_fn, name="async_tool")

    script = ScriptedModel([
        ("tool", "sync_tool", {"x": 1}),
        ("tool", "async_tool", {"y": "hi"}),
        "both ran",
    ])

    async def model(messages, tools):  # an async Model
        return script(messages, tools)

    state = asyncio.run(AsyncHarness(model, tools=[sync_tool, async_tool]).arun("go"))
    assert state.result == "both ran"
    assert ("sync", 1) in calls and ("async", "hi") in calls


def test_async_harness_runs_sync_model_and_components():
    # sync ScriptedModel + a Verifier — same semantics as the sync loop
    model = ScriptedModel(["nope", "the answer is 42"])

    def check(state):
        return ("42" in (state.result or ""), "must contain 42")

    state = asyncio.run(AsyncHarness(model, components=[Verifier(check)]).arun("q"))
    assert state.memory["_verified"] is True and state.result == "the answer is 42"


def test_async_parallel_tools_run_concurrently():
    started, order = [], []

    async def slow(tag: str) -> str:
        started.append(tag)
        await asyncio.sleep(0.05)
        order.append(tag)
        return tag

    t = tool(slow, name="slow")
    resp = Response(tool_calls=[
        __import__("pyhar").ToolCall(id="a", name="slow", arguments={"tag": "a"}),
        __import__("pyhar").ToolCall(id="b", name="slow", arguments={"tag": "b"}),
    ])
    model = ScriptedModel([resp, "done"])

    t0 = time.monotonic()
    state = asyncio.run(AsyncHarness(model, tools=[t], parallel_tools=True).arun("go"))
    elapsed = time.monotonic() - t0

    assert state.done
    assert set(started) == {"a", "b"}
    assert elapsed < 0.15  # concurrent, not 2 * 0.05 + big overhead
    # results appear in original call order regardless of completion order
    tool_msgs = [m for m in state.messages if m.role == "tool"]
    assert [m.content for m in tool_msgs] == ["a", "b"]


# -- sync parallel tools -------------------------------------------------------

def test_sync_parallel_tools_preserve_order_and_gating():
    @tool
    def work(tag: str) -> str:
        time.sleep(0.02)
        return f"ran:{tag}"

    @tool
    def blocked() -> str:
        raise AssertionError("must not run")

    from pyhar import ToolCall
    resp = Response(tool_calls=[
        ToolCall(id="1", name="work", arguments={"tag": "x"}),
        ToolCall(id="2", name="blocked", arguments={}),
        ToolCall(id="3", name="work", arguments={"tag": "y"}),
    ])
    model = ScriptedModel([resp, "done"])
    state = Harness(
        model,
        components=[Permissions(deny=["blocked"])],
        tools=[work, blocked],
        parallel_tools=True,
    ).run("go")

    tool_msgs = [m.content for m in state.messages if m.role == "tool"]
    assert tool_msgs[0] == "ran:x"
    assert "permission denied" in tool_msgs[1]
    assert tool_msgs[2] == "ran:y"


# -- combinators ---------------------------------------------------------------

class _Flaky:
    """Fails n times, then delegates to a ScriptedModel."""

    def __init__(self, fails: int, then):
        self.fails = fails
        self.then = then
        self.attempts = 0

    def __call__(self, messages, tools):
        self.attempts += 1
        if self.attempts <= self.fails:
            raise ConnectionError("boom")
        return self.then(messages, tools)


def test_retry_model_retries_then_succeeds():
    naps = []
    flaky = _Flaky(2, ScriptedModel(["ok"]))
    model = RetryModel(flaky, max_retries=3, base_delay=1.0, sleep=naps.append)
    resp = model([Message(role="user", content="hi")], [])
    assert resp.text == "ok" and flaky.attempts == 3
    assert naps == [1.0, 2.0]  # exponential backoff, no real sleeping


def test_retry_model_raises_after_exhaustion():
    flaky = _Flaky(10, ScriptedModel(["never"]))
    model = RetryModel(flaky, max_retries=2, sleep=lambda s: None)
    with pytest.raises(ConnectionError):
        model([Message(role="user", content="hi")], [])
    assert flaky.attempts == 3  # 1 try + 2 retries


def test_fallback_model_fails_over_in_order():
    primary = _Flaky(99, ScriptedModel(["never"]))
    backup = ScriptedModel(["served by backup"])
    model = FallbackModel([primary, backup])
    resp = model([Message(role="user", content="hi")], [])
    assert resp.text == "served by backup" and model.last_served == 1


def test_fallback_model_respects_should_fallback():
    primary = _Flaky(99, ScriptedModel(["never"]))
    model = FallbackModel([primary, ScriptedModel(["x"])], should_fallback=lambda e: False)
    with pytest.raises(ConnectionError):
        model([Message(role="user", content="hi")], [])


def test_router_model_routes_and_defaults():
    tier = {"key": "strong"}
    model = RouterModel(
        {
            "strong": ScriptedModel(["from strong", "from strong again"]),
            "cheap": ScriptedModel(["from cheap"]),
        },
        route=lambda msgs, tools: tier["key"],
        default="strong",
    )
    assert model([Message(role="user", content="a")], []).text == "from strong"
    tier["key"] = "cheap"
    assert model([Message(role="user", content="b")], []).text == "from cheap"
    assert model.last_key == "cheap"
    tier["key"] = "no-such-model"  # unknown key -> default
    assert model([Message(role="user", content="c")], []).text == "from strong again"
    assert model.last_key == "strong"


# -- LoopGuard -------------------------------------------------------------

def test_loop_guard_breaks_identical_call_streak():
    @tool
    def probe(q: str) -> str:
        return "same result"

    same = ("tool", "probe", {"q": "x"})
    model = ScriptedModel([same, same, same, same, same, "gave up"])
    state = Harness(model, components=[LoopGuard(max_repeats=3)], tools=[probe]).run("go")

    tool_msgs = [m.content for m in state.messages if m.role == "tool"]
    assert tool_msgs[:3] == ["same result"] * 3
    assert "loop guard" in tool_msgs[3]
    assert state.memory["_loop_guard"][0]["tool"] == "probe"


def test_loop_guard_resets_on_different_arguments():
    guard = LoopGuard(max_repeats=2)
    state = HarnessState()
    from pyhar import ToolCall

    a = ToolCall(id="1", name="t", arguments={"q": "a"})
    b = ToolCall(id="2", name="t", arguments={"q": "b"})
    assert guard.before_tool(state, a) is None
    assert guard.before_tool(state, a) is None
    assert guard.before_tool(state, b) is None  # different args reset the streak
    assert guard.before_tool(state, a) is None  # streak restarted at 1


def test_coding_agent_preset_includes_loop_guard():
    assert any(isinstance(c, LoopGuard) for c in coding_agent(ScriptedModel(["x"])).components)


# -- checks ------------------------------------------------------------------

def _state_with_result(text: str) -> HarnessState:
    s = HarnessState()
    s.result = text
    return s


def test_contains_and_regex_checks():
    ok, _ = contains_check("Answer", "42")(_state_with_result("the ANSWER is 42"))
    assert ok
    bad, msg = contains_check("missing")(_state_with_result("nothing here"))
    assert not bad and "missing" in msg
    assert regex_check(r"\b\d{2}\b")(_state_with_result("it is 42"))[0]


def test_json_schema_check_accepts_valid_and_fenced():
    schema = {
        "type": "object",
        "required": ["answer", "confidence"],
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    check = json_schema_check(schema)
    ok, _ = check(_state_with_result('{"answer": "yes", "confidence": 0.9, "tags": ["a"]}'))
    assert ok
    ok2, _ = check(_state_with_result('```json\n{"answer": "x", "confidence": 1}\n```'))
    assert ok2


def test_json_schema_check_reports_specific_failures():
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}, "n": {"type": "integer"}},
        "additionalProperties": False,
    }
    check = json_schema_check(schema)
    ok, msg = check(_state_with_result("not json at all"))
    assert not ok and "valid JSON" in msg
    ok, msg = check(_state_with_result('{"n": 1}'))
    assert not ok and "answer is required" in msg
    ok, msg = check(_state_with_result('{"answer": 5}'))
    assert not ok and "must be string" in msg
    ok, msg = check(_state_with_result('{"answer": "x", "extra": 1}'))
    assert not ok and "not an allowed property" in msg
    ok, msg = check(_state_with_result('{"answer": "x", "n": true}'))  # bool is not integer
    assert not ok and "must be integer" in msg


def test_json_schema_check_drives_verifier_retry():
    model = ScriptedModel(['not json', '{"answer": "done"}'])
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
    state = Harness(model, components=[Verifier(json_schema_check(schema))]).run("q")
    assert state.memory["_verified"] is True
    assert parse_json_result(state) == {"answer": "done"}


# -- bench trials ---------------------------------------------------------------

def test_bench_trials_aggregate():
    def factory():
        return Harness(ScriptedModel(["ok"]))

    report = bench("t", {"cfg": factory}, trials=3)
    run = report.runs[0]
    assert run.trials == 3 and run.success_rate == 1.0 and run.turns == 1.0
    assert "trials" in report.table()


def test_bench_single_trial_backward_compatible():
    report = bench("t", {"cfg": lambda: Harness(ScriptedModel(["ok"]))})
    run = report.runs[0]
    assert run.trials == 1 and run.success is True
    assert "trials" not in report.table()


# -- stop_reason + registry -----------------------------------------------------

def test_response_stop_reason_defaults_none_and_is_settable():
    assert Response(text="x").stop_reason is None
    r = Response(text="x", stop_reason="max_tokens", usage=Usage())
    assert r.stop_reason == "max_tokens"


def test_registry_has_builtin_components():
    from pyhar import Compactor, Tracer
    assert registry.get("compactor") is Compactor
    assert registry.get("tracer") is Tracer
    names = registry.available()
    assert {"compactor", "loop_guard", "permissions", "verifier"} <= set(names)


# -- regression tests for the 0.3.0 review findings -----------------------------

def test_async_dispatch_awaits_coroutine_from_sync_closure():
    # MCP-style tool: a SYNC closure that returns a coroutine (finding #1)
    async def remote_call(name, arguments):
        await asyncio.sleep(0)
        return f"mcp:{arguments['q']}"

    from pyhar import Tool
    mcp_like = Tool(name="search", fn=lambda **kw: remote_call("search", kw))

    model = ScriptedModel([("tool", "search", {"q": "x"}), "done"])
    state = asyncio.run(AsyncHarness(model, tools=[mcp_like]).arun("go"))
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content == "mcp:x"  # awaited, not repr(<coroutine ...>)
    assert "coroutine" not in tool_msg.content


def test_async_model_via_functools_partial_is_awaited():
    import functools

    async def base(tag, messages, tools):
        return Response(text=f"from {tag}")

    model = functools.partial(base, "partial")  # defeats iscoroutinefunction on the wrapper
    state = asyncio.run(AsyncHarness(model).arun("go"))
    assert state.result == "from partial"


def test_json_schema_accepts_list_form_type():
    schema = {
        "type": "object",
        "required": ["note"],
        "properties": {"note": {"type": ["string", "null"]}},
    }
    check = json_schema_check(schema)
    assert check(_state_with_result('{"note": null}'))[0]
    assert check(_state_with_result('{"note": "hi"}'))[0]
    ok, msg = check(_state_with_result('{"note": 5}'))
    assert not ok and "string or null" in msg


def test_parse_json_valid_json_with_backticks_not_mangled():
    payload = '{"snippet": "use ```python\\nprint(1)\\n``` here", "answer": 42}'
    assert parse_json_result(_state_with_result(payload))["answer"] == 42


def test_parse_json_prefers_last_valid_fence():
    text = 'Bad:\n```json\n{broken\n```\nGood:\n```json\n{"a": 1}\n```'
    assert parse_json_result(_state_with_result(text)) == {"a": 1}


def test_checks_do_not_fall_back_past_empty_final_answer():
    s = HarnessState()
    s.add_message(Message(role="assistant", content='{"answer": "stale"}',
                          tool_calls=[__import__("pyhar").ToolCall(id="1", name="t")]))
    s.result = ""  # the current candidate is EMPTY — must fail, not pass on stale text
    ok, _ = contains_check("stale")(s)
    assert not ok


def test_enum_rejects_bool_int_cross_match():
    check = json_schema_check({"type": "object", "properties": {"n": {"enum": [1, 2]}}})
    assert check(_state_with_result('{"n": 1}'))[0]
    ok, _ = check(_state_with_result('{"n": true}'))
    assert not ok  # True == 1 in Python, but not in JSON


def test_json_schema_check_rejects_unknown_type_names_at_construction():
    with pytest.raises(ValueError, match="unknown type 'str'"):
        json_schema_check({"type": "object", "properties": {"x": {"type": "str"}}})


def test_max_turns_zero_means_zero_turns():
    from pyhar import Budget
    calls = []

    def counting_model(messages, tools):
        calls.append(1)
        return Response(text="never")

    state = Harness(counting_model, budget=Budget(max_turns=0)).run("go")
    assert calls == [] and state.turn == 0
    assert state.memory["_stop_reason"] == "max_turns"


def test_shared_budget_not_mutated_by_constructor():
    from pyhar import Budget
    shared = Budget(max_cost=1.0)
    Harness(ScriptedModel(["a"]), budget=shared, max_turns=3)
    worker = Harness(ScriptedModel(["b"]), budget=shared, max_turns=50)
    assert shared.max_turns is None          # caller's object untouched
    assert worker.budget.max_turns == 50     # second harness got its own default


def test_reused_harness_resets_verifier_and_loop_guard():
    # Verifier: retry budget restored on each run
    def never_pass(state):
        return (False, "nope")

    h = Harness(ScriptedModel(["a", "b", "c", "x", "y", "z"]),
                components=[Verifier(never_pass, max_retries=2)])
    s1 = h.run("one")
    s2 = h.run("two")
    assert s1.turn == 3 and s2.turn == 3  # full retry budget both times

    # LoopGuard: streak/totals cleared between runs
    @tool
    def probe(q: str) -> str:
        return "r"

    same = ("tool", "probe", {"q": "x"})
    g = Harness(ScriptedModel([same, same, "end", same, same, "end"]),
                components=[LoopGuard(max_repeats=2, max_total_repeats=3)], tools=[probe])
    g.run("one")
    s2 = g.run("two")
    tool_msgs = [m.content for m in s2.messages if m.role == "tool"]
    assert all("loop guard" not in c for c in tool_msgs)  # counters did not leak


def test_loop_guard_canonicalizes_nested_arguments():
    guard = LoopGuard(max_repeats=1)
    state = HarnessState()
    from pyhar import ToolCall
    a = ToolCall(id="1", name="t", arguments={"cfg": {"a": 1, "b": 2}})
    b = ToolCall(id="2", name="t", arguments={"cfg": {"b": 2, "a": 1}})  # same, reordered
    assert guard.before_tool(state, a) is None
    assert guard.before_tool(state, b) is not None  # recognized as identical


def test_router_last_key_none_after_failure():
    class Boom:
        def __call__(self, messages, tools):
            raise RuntimeError("x")

    model = RouterModel({"a": Boom()}, route=lambda m, t: "a", default="a")
    with pytest.raises(RuntimeError):
        model([Message(role="user", content="hi")], [])
    assert model.last_key is None


def test_on_end_runs_even_when_budget_exceeded():
    from pyhar import Budget, BudgetExceeded, Component

    class EndRecorder(Component):
        def __init__(self):
            self.ended = 0

        def on_end(self, state):
            self.ended += 1

    rec = EndRecorder()
    h = Harness(ScriptedModel([("tool", "x", {})] * 5, output_tokens=10**9),
                components=[rec], budget=Budget(max_total_tokens=1))
    with pytest.raises(BudgetExceeded):
        h.run("go")
    assert rec.ended == 1
