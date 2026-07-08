"""Model backends are tested via injected fake clients — no SDKs, no network."""
from types import SimpleNamespace

from pyhar import Message, ToolCall
from pyhar.models import AnthropicModel, EchoModel, OllamaModel, OpenAIModel


def _conversation():
    return [
        Message(role="system", content="be terse"),
        Message(role="user", content="inspect db"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="t1", name="read", arguments={})]),
        Message(role="tool", content="big output", tool_call_id="t1", name="read"),
    ]


# -- EchoModel -------------------------------------------------------------

def test_echo_model_echoes_last_user():
    resp = EchoModel()([Message(role="user", content="hello")], [])
    assert resp.text == "ok: hello"
    assert resp.tool_calls == []


# -- AnthropicModel --------------------------------------------------------

class _FakeMessages:
    def __init__(self, resp):
        self.resp = resp
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


class _FakeAnthropic:
    def __init__(self, resp):
        self.messages = _FakeMessages(resp)


def test_anthropic_conversion_and_mapping():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="found it"),
            SimpleNamespace(type="tool_use", id="tu1", name="grep", input={"q": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        stop_reason="tool_use",
    )
    fake = _FakeAnthropic(resp)
    model = AnthropicModel(client=fake, system="global sys")
    out = model([Message(role=m.role, content=m.content, tool_calls=m.tool_calls,
                         tool_call_id=m.tool_call_id) for m in _conversation()], [])

    # response mapping
    assert out.text == "found it"
    assert out.tool_calls[0].name == "grep" and out.tool_calls[0].arguments == {"q": "x"}
    assert out.usage.input_tokens == 1000
    assert out.usage.cost > 0  # opus-4-8 pricing applied

    # request conversion: system hoisted, tool_result present, no temperature sent
    kw = fake.messages.kwargs
    assert "global sys" in kw["system"] and "be terse" in kw["system"]
    assert "temperature" not in kw
    roles = [msg["role"] for msg in kw["messages"]]
    assert roles == ["user", "assistant", "user"]  # system hoisted out
    assert kw["messages"][2]["content"][0]["type"] == "tool_result"


def test_anthropic_effort_and_thinking_passthrough():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                           stop_reason="end_turn")
    fake = _FakeAnthropic(resp)
    AnthropicModel(client=fake, thinking={"type": "adaptive"}, effort="high")(
        [Message(role="user", content="hi")], []
    )
    assert fake.messages.kwargs["thinking"] == {"type": "adaptive"}
    assert fake.messages.kwargs["output_config"] == {"effort": "high"}


# -- OpenAIModel -----------------------------------------------------------

class _FakeCompletions:
    def __init__(self, resp):
        self.resp = resp
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


class _FakeOpenAI:
    def __init__(self, resp):
        self.chat = SimpleNamespace(completions=_FakeCompletions(resp))


def test_openai_conversion_and_mapping():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="done",
            tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="grep", arguments='{"q": "x"}'))],
        ))],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=4),
    )
    fake = _FakeOpenAI(resp)
    out = OpenAIModel("gpt-4o-mini", client=fake)(_conversation(), [])

    assert out.text == "done"
    assert out.tool_calls[0].name == "grep" and out.tool_calls[0].arguments == {"q": "x"}
    assert out.usage.input_tokens == 8
    # tool message maps to role "tool"; assistant tool_calls serialize arguments as JSON string
    sent = fake.chat.completions.kwargs["messages"]
    assert any(m["role"] == "tool" for m in sent)
    asst = next(m for m in sent if m["role"] == "assistant" and m.get("tool_calls"))
    assert isinstance(asst["tool_calls"][0]["function"]["arguments"], str)


# -- OllamaModel -----------------------------------------------------------

def test_anthropic_coalesces_parallel_tool_results_and_skips_empty_assistant():
    from pyhar.models.anthropic import _to_anthropic_messages

    msgs = [
        Message(role="assistant", content="", tool_calls=[ToolCall(id="a", name="x"), ToolCall(id="b", name="y")]),
        Message(role="tool", content="ra", tool_call_id="a", name="x"),
        Message(role="tool", content="rb", tool_call_id="b", name="y"),
    ]
    _system, out = _to_anthropic_messages(msgs, None)
    # assistant kept (has tool_calls); the two tool results are ONE user message, two blocks
    assert [m["role"] for m in out] == ["assistant", "user"]
    assert len(out[1]["content"]) == 2
    assert all(b["type"] == "tool_result" for b in out[1]["content"])

    # an empty assistant turn (no text, no tool_calls) is dropped, not sent as empty text
    _s2, out2 = _to_anthropic_messages([Message(role="assistant", content="")], None)
    assert out2 == []


def test_ollama_via_injected_transport():
    calls = {}

    def transport(url, payload):
        calls["url"] = url
        calls["payload"] = payload
        return {
            "message": {"content": "hi", "tool_calls": [{"function": {"name": "grep", "arguments": {"q": "x"}}}]},
            "prompt_eval_count": 7,
            "eval_count": 3,
        }

    out = OllamaModel("llama3.1", transport=transport)(_conversation(), [])
    assert out.text == "hi"
    assert out.tool_calls[0].name == "grep" and out.tool_calls[0].arguments == {"q": "x"}
    assert out.usage.input_tokens == 7 and out.usage.cost == 0.0
    assert calls["url"].endswith("/api/chat")
