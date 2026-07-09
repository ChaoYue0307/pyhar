"""Tests for the 0.4.0 round: streaming, registry ecosystem, config-driven
harnesses, and Tracer deltas. (LangGraph integration lives in
test_langgraph_integration.py.)"""
import asyncio
from types import SimpleNamespace

import pytest

from pyhar import (
    AsyncHarness,
    Component,
    Harness,
    Message,
    ScriptedModel,
    Tracer,
    harness_from_config,
    registry,
    tool,
)
from pyhar.models import AnthropicModel, OllamaModel, OpenAIModel


class DeltaCollector(Component):
    def __init__(self):
        self.deltas = []

    def on_delta(self, state, delta):
        self.deltas.append(delta)


# -- streaming: harness fanout ------------------------------------------------

def test_stream_fans_deltas_and_final_result_matches():
    collector = DeltaCollector()
    model = ScriptedModel(["the final answer"])
    state = Harness(model, components=[collector], stream=True).run("q")

    assert state.result == "the final answer"
    assert "".join(collector.deltas) == "the final answer"
    assert len(collector.deltas) == 3  # word-sized chunks
    assert "_stream_fallback" not in state.memory


def test_stream_falls_back_when_model_cannot_stream():
    collector = DeltaCollector()

    def plain_model(messages, tools):  # no .stream attribute
        from pyhar import Response
        return Response(text="done")

    state = Harness(plain_model, components=[collector], stream=True).run("q")
    assert state.result == "done"
    assert collector.deltas == []
    assert state.memory["_stream_fallback"] is True


def test_stream_off_by_default_no_deltas():
    collector = DeltaCollector()
    state = Harness(ScriptedModel(["hi"]), components=[collector]).run("q")
    assert collector.deltas == [] and state.result == "hi"


def test_async_harness_streams_via_sync_stream_offthread():
    collector = DeltaCollector()
    model = ScriptedModel(["async streamed answer"])
    state = asyncio.run(
        AsyncHarness(model, components=[collector], stream=True).arun("q")
    )
    assert state.result == "async streamed answer"
    assert "".join(collector.deltas) == "async streamed answer"


def test_async_harness_prefers_astream():
    collector = DeltaCollector()

    class AStreamModel:
        def __call__(self, messages, tools):  # pragma: no cover - not used
            raise AssertionError("should use astream")

        async def astream(self, messages, tools, *, on_delta):
            from pyhar import Response
            for chunk in ("a", "b", "c"):
                on_delta(chunk)
            return Response(text="abc")

    state = asyncio.run(
        AsyncHarness(AStreamModel(), components=[collector], stream=True).arun("q")
    )
    assert state.result == "abc" and collector.deltas == ["a", "b", "c"]


def test_tracer_records_deltas_when_opted_in():
    tracer = Tracer(include_deltas=True)
    state = Harness(ScriptedModel(["one two"]), components=[tracer], stream=True).run("q")
    kinds = [e["event"] for e in state.memory["_trace"]]
    assert kinds.count("delta") == 2
    assert not any(e["event"] == "delta" for e in
                   Harness(ScriptedModel(["x"]), components=[Tracer()], stream=True)
                   .run("q").memory["_trace"])


# -- streaming: provider backends (fake clients) -------------------------------

def test_anthropic_stream_via_fake_client():
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello world")],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        stop_reason="end_turn",
    )

    class FakeStreamCtx:
        def __init__(self):
            self.text_stream = iter(["hello ", "world"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return final

    fake = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: final, stream=lambda **kw: FakeStreamCtx()
    ))
    deltas = []
    resp = AnthropicModel(client=fake).stream(
        [Message(role="user", content="hi")], [], on_delta=deltas.append
    )
    assert deltas == ["hello ", "world"]
    assert resp.text == "hello world" and resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 5


def test_openai_stream_accumulates_text_and_tool_calls():
    def chunk(content=None, tool_calls=None, finish=None, usage=None):
        choice = SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    tc_part1 = SimpleNamespace(index=0, id="c1",
                               function=SimpleNamespace(name="grep", arguments='{"q"'))
    tc_part2 = SimpleNamespace(index=0, id=None,
                               function=SimpleNamespace(name=None, arguments=': "x"}'))
    chunks = [
        chunk(content="thinking "),
        chunk(tool_calls=[tc_part1]),
        chunk(tool_calls=[tc_part2], finish="tool_calls"),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    deltas = []
    resp = OpenAIModel("gpt-4o-mini", client=fake).stream(
        [Message(role="user", content="hi")], [], on_delta=deltas.append
    )
    assert deltas == ["thinking "]
    assert resp.tool_calls[0].name == "grep"
    assert resp.tool_calls[0].arguments == {"q": "x"}   # assembled across chunks
    assert resp.stop_reason == "tool_calls"
    assert resp.usage.input_tokens == 7


def test_ollama_stream_via_injected_line_transport():
    lines = [
        {"message": {"content": "hel"}},
        {"message": {"content": "lo"}},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 4, "eval_count": 2, "done_reason": "stop"},
    ]
    deltas = []
    model = OllamaModel("llama3.1", stream_transport=lambda url, payload: iter(lines))
    resp = model.stream([Message(role="user", content="hi")], [], on_delta=deltas.append)
    assert deltas == ["hel", "lo"]
    assert resp.text == "hello" and resp.stop_reason == "stop"
    assert resp.usage.input_tokens == 4 and resp.usage.output_tokens == 2


# -- registry ecosystem ---------------------------------------------------------

def test_registry_create_and_build():
    from pyhar import Compactor, LoopGuard
    c = registry.create("compactor", target_tokens=123)
    assert isinstance(c, Compactor) and c.target_tokens == 123

    built = registry.build([
        "loop_guard",
        {"name": "compactor", "args": {"target_tokens": 55}},
    ])
    assert isinstance(built[0], LoopGuard)
    assert built[1].target_tokens == 55


def test_registry_helpful_errors():
    with pytest.raises(KeyError, match="no component registered as 'nope'"):
        registry.get("nope")
    with pytest.raises(ValueError, match="bad component spec"):
        registry.build([42])


def test_registry_load_entrypoints(monkeypatch):
    class GoodEP:
        name = "third_party_widget"

        def load(self):
            class Widget(Component):
                name = "third_party_widget"
            return Widget

    class BrokenEP:
        name = "broken_widget"

        def load(self):
            raise ImportError("bad package")

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group=None: [GoodEP(), BrokenEP()])
    loaded = registry.load_entrypoints()
    assert loaded == ["third_party_widget", "!broken_widget"]
    assert registry.get("third_party_widget").name == "third_party_widget"


# -- config-driven harnesses ------------------------------------------------------

def test_harness_from_config_full():
    config = {
        "system": "be terse",
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": 50}},
            "tracer",
        ],
        "budget": {"max_context_tokens": 500},
        "max_turns": 7,
        "parallel_tools": True,
    }

    @tool
    def read(path: str) -> str:
        return "y" * 4000

    model = ScriptedModel([("tool", "read", {"path": "f"}), "done"])
    h = harness_from_config(config, model=model, tools=[read])
    assert h.budget.max_context_tokens == 500 and h.budget.max_turns == 7
    assert h.parallel_tools is True

    state = h.run("go")
    assert state.done
    assert state.memory["_tool_savings"] > 0      # budget component came from config
    assert state.memory["_trace"][0]["event"] == "start"
    assert state.messages[0].content == "be terse"


def test_harness_from_config_rejects_unknown_keys_and_applies_overrides():
    with pytest.raises(ValueError, match="unknown config keys"):
        harness_from_config({"componnets": []}, model=ScriptedModel(["x"]))

    h = harness_from_config({"max_turns": 3}, model=ScriptedModel(["x"]), max_turns=9)
    assert h.budget.max_turns == 9  # kwarg override wins


def test_harness_from_config_async_cls():
    h = harness_from_config(
        {"components": ["tracer"]}, model=ScriptedModel(["ok"]), harness_cls=AsyncHarness
    )
    state = asyncio.run(h.arun("q"))
    assert state.result == "ok" and state.memory["_trace"]


def test_budget_from_config_is_isolated():
    config = {"budget": {"max_turns": 2}}
    h1 = harness_from_config(config, model=ScriptedModel(["a"]))
    h2 = harness_from_config(config, model=ScriptedModel(["b"]))
    h1.budget.max_turns = 99
    assert h2.budget.max_turns == 2  # no shared mutable Budget between harnesses


# -- regression tests for the 0.4.0 review findings -----------------------------

def test_combinators_forward_streaming():
    from pyhar.models import FallbackModel, RetryModel, RouterModel

    inner = ScriptedModel(["streamed through wrappers"])
    model = RetryModel(FallbackModel([inner]))
    state = Harness(model, components=[], stream=True).run("q")
    # harness found .stream on the combinator chain — no fallback breadcrumb
    assert "_stream_fallback" not in state.memory

    collector = DeltaCollector()
    inner2 = ScriptedModel(["router streams too"])
    router = RouterModel({"only": inner2}, route=lambda m, t: "only", default="only")
    state2 = Harness(router, components=[collector], stream=True).run("q")
    assert "".join(collector.deltas) == "router streams too"
    assert state2.result == "router streams too"


def test_retry_model_streams_after_failed_attempt():
    class FlakyStreamer:
        def __init__(self):
            self.attempts = 0

        def __call__(self, messages, tools):  # pragma: no cover
            raise AssertionError("stream path should be used")

        def stream(self, messages, tools, *, on_delta):
            from pyhar import Response
            self.attempts += 1
            if self.attempts == 1:
                on_delta("partial ")  # dies mid-stream
                raise ConnectionError("dropped")
            on_delta("full answer")
            return Response(text="full answer")

    from pyhar.models import RetryModel
    deltas = []
    flaky = FlakyStreamer()
    model = RetryModel(flaky, max_retries=2, sleep=lambda s: None)
    resp = model.stream([Message(role="user", content="q")], [], on_delta=deltas.append)
    assert resp.text == "full answer" and flaky.attempts == 2
    assert deltas == ["partial ", "full answer"]  # documented re-emission on retry


def test_openai_stream_falls_back_on_bad_request_error():
    class BadRequestError(Exception):
        pass

    final_chunks = [SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None),
                                 finish_reason="stop")],
        usage=None,
    )]

    class FakeCompletions:
        def create(self, **kwargs):
            if "stream_options" in kwargs:
                raise BadRequestError("unknown field: stream_options")
            return iter(final_chunks)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    deltas = []
    resp = OpenAIModel("m", client=fake).stream(
        [Message(role="user", content="hi")], [], on_delta=deltas.append
    )
    assert resp.text == "ok" and deltas == ["ok"]  # retried without stream_options


def test_openai_stream_handles_missing_index():
    tc = SimpleNamespace(index=None, id="c9",
                         function=SimpleNamespace(name="grep", arguments='{"q": "z"}'))
    chunks = [SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]),
                                 finish_reason="tool_calls")],
        usage=None,
    )]

    class FakeCompletions:
        def create(self, **kwargs):
            return iter(chunks)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    resp = OpenAIModel("m", client=fake).stream(
        [Message(role="user", content="hi")], [], on_delta=lambda d: None
    )
    assert resp.tool_calls[0].id == "c9" and resp.tool_calls[0].arguments == {"q": "z"}


def test_ollama_tool_call_ids_unique_across_responses():
    data = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "read", "arguments": {}}}]}}
    model = OllamaModel("m", transport=lambda url, payload: data)
    r1 = model([Message(role="user", content="a")], [])
    r2 = model([Message(role="user", content="b")], [])
    assert r1.tool_calls[0].id != r2.tool_calls[0].id


def test_config_budget_override_as_dict_and_max_turns_precedence():
    h = harness_from_config({}, model=ScriptedModel(["x"]), budget={"max_turns": 5})
    assert h.budget.max_turns == 5  # dict override coerced, no TypeError

    h2 = harness_from_config(
        {"budget": {"max_turns": 2, "max_context_tokens": 100}},
        model=ScriptedModel(["x"]),
        max_turns=9,
    )
    assert h2.budget.max_turns == 9              # explicit max_turns wins
    assert h2.budget.max_context_tokens == 100   # other budget fields kept


def test_registry_entrypoint_collision_is_skipped_with_warning(monkeypatch):
    import warnings as w

    class HijackEP:
        name = "compactor"  # tries to shadow the built-in
        value = "evil_pkg:FakeCompactor"

        def load(self):
            class FakeCompactor(Component):
                name = "compactor"
            return FakeCompactor

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [HijackEP()])
    from pyhar import Compactor
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        loaded = registry.load_entrypoints()
    assert loaded == ["!compactor"]
    assert registry.get("compactor") is Compactor      # built-in untouched
    assert any("collides" in str(c.message) for c in caught)
