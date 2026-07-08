# Model backends & tools

pyhar never imports a provider SDK. A **model** is anything that maps a message
list plus tool specs to a `Response`. That one-method boundary is the whole
contract — wrap Anthropic, OpenAI, a local server, or a stub of your own, and
everything else (the [Harness](concepts.md), [Components](components.md),
[adapters and MCP](adapters-and-mcp.md)) works unchanged.

> **Install.** The import name is `pyhar`; the PyPI distribution is
> `pyhar-agents`.
>
> ```bash
> pip install pyhar-agents        # core, zero runtime dependencies
> ```
> ```python
> import pyhar                     # this line never needs a provider SDK
> ```

## The `Model` protocol

```python
from typing import Any, Protocol
from pyhar import Message, Response

class Model(Protocol):
    def __call__(self, messages: list[Message], tools: list[Any]) -> Response: ...
```

A `Response` is a small dataclass:

```python
from dataclasses import dataclass, field
from pyhar import Response, ToolCall, Usage

@dataclass
class Response:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None            # the untouched provider payload, if any
```

`text` is the assistant's message, `tool_calls` is the list of calls the model
wants the harness to run this turn, `usage` carries `input_tokens`,
`output_tokens`, and `cost`, and `raw` holds the original provider object for
escape hatches. That's all a backend has to produce.

## `ScriptedModel` — deterministic, key-free

Every runnable snippet on this page uses `ScriptedModel`, so you can paste and
run it with no API key. It returns queued responses in order. Each script item
is either:

- a **string** → a final text answer, or
- a `("tool", name, {args})` tuple → a single tool call.

```python
from pyhar import Harness, ScriptedModel, tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

model = ScriptedModel([
    ("tool", "add", {"a": 2, "b": 3}),   # turn 1: call the tool
    "The sum is 5.",                       # turn 2: final answer
])

state = Harness(model, tools=[add]).run("What is 2 + 3?")
print(state.result)                        # -> "The sum is 5."
print(state.usage.total_tokens)
```

`ScriptedModel` attributes input tokens from the live context size at call time,
so context-shrinking components (see the [Compactor](components.md)) show
visibly lower usage — useful when demonstrating or benchmarking a harness.

## `EchoModel` — a one-line template

`EchoModel` echoes the last user message and never calls tools. It is the
smallest possible `Model` and a good template for writing your own.

```python
from pyhar import Harness
from pyhar.models import EchoModel

state = Harness(EchoModel(prefix="ok: ")).run("hello")
print(state.result)     # -> "ok: hello"
```

## `AnthropicModel` — Claude via the official SDK

```bash
pip install "pyhar-agents[anthropic]"
```

The `anthropic` SDK is lazy-imported: the `ImportError` fires only when you
actually construct an `AnthropicModel` without the extra installed (or without
injecting a `client=`).

```python
from pyhar import Harness
from pyhar.models import AnthropicModel

model = AnthropicModel(
    "claude-opus-4-8",                # default model
    max_tokens=4096,
    system="You are a terse assistant.",
    thinking={"type": "adaptive"},    # adaptive extended thinking
    effort="high",                    # "low" | "medium" | "high" | "xhigh" | "max"
)
state = Harness(model).run("Explain adaptive thinking in one sentence.")
```

Constructor arguments:

| Argument | Purpose |
| --- | --- |
| `model` | model id, default `claude-opus-4-8` |
| `max_tokens` | output cap, default `4096` |
| `system` | system prompt hoisted into the top-level Anthropic `system` field |
| `thinking` | e.g. `{"type": "adaptive"}` — passed straight through |
| `effort` | reasoning depth via `output_config.effort` |
| `api_key` | overrides `ANTHROPIC_API_KEY` |
| `client` | inject a client (or a fake for tests) |
| `pricing` | `(input, output)` USD-per-million override |

This backend follows the current Messages API: adaptive thinking via
`thinking={"type": "adaptive"}` (never `budget_tokens`), depth via
`effort`, and **no** `temperature` / `top_p` / `top_k` (those are rejected on
Opus 4.8 / Sonnet 5 / Fable 5). pyhar messages are converted for you: `system`
messages are hoisted into the top-level system string, assistant tool calls
become `tool_use` blocks, and `tool` results are grouped into `tool_result`
blocks.

## `OpenAIModel` / `OpenAICompatibleModel`

```bash
pip install "pyhar-agents[openai]"
```

```python
from pyhar.models import OpenAIModel

model = OpenAIModel("gpt-4o-mini", max_tokens=1024)
```

| Argument | Purpose |
| --- | --- |
| `model` | model id, default `gpt-4o-mini` |
| `base_url` | point at any OpenAI-compatible server |
| `api_key` | overrides `OPENAI_API_KEY` |
| `max_tokens` | output cap, default `1024` |
| `client` | inject a client (or a fake) |
| `pricing` | `(input, output)` USD-per-million override |

Pass `base_url` to talk to any OpenAI-compatible server — vLLM, Together,
LM Studio, or Ollama's `/v1` endpoint — which is the common way to run OSS
models. `OpenAICompatibleModel` is a thin convenience wrapper that requires
`base_url` and defaults `api_key` to `"not-needed"`:

```python
from pyhar.models import OpenAICompatibleModel

# a local vLLM / LM Studio server
model = OpenAICompatibleModel(
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3.1-8B-Instruct",
)
```

## `OllamaModel` — local, zero dependencies

```python
from pyhar.models import OllamaModel

model = OllamaModel("llama3.1")      # after: ollama pull llama3.1
```

Talks to a local Ollama server's native chat API over stdlib `urllib` — **no SDK
required**, so no install extra. Tool calling requires a tool-capable model.
Because local inference is free, `Usage.cost` stays `0.0`.

| Argument | Purpose |
| --- | --- |
| `model` | model id, default `llama3.1` |
| `host` | Ollama server, default `http://localhost:11434` |
| `timeout` | request timeout in seconds, default `120.0` |
| `options` | extra Ollama options (temperature, etc.) |
| `transport` | injectable `callable(url, payload) -> dict` for tests |

## Write your own `Model`

Because the protocol is a single call, a backend is just a callable. Here is a
complete, working model that always answers a fixed string:

```python
from pyhar import Harness, Response, Usage

class ConstantModel:
    def __init__(self, answer: str):
        self.answer = answer

    def __call__(self, messages, tools):
        return Response(
            text=self.answer,
            usage=Usage(input_tokens=10, output_tokens=5),
        )

state = Harness(ConstantModel("always 42")).run("anything")
print(state.result)     # -> "always 42"
```

To make a real backend: translate `messages` into the provider's format, call
the API, and map the reply back into a `Response` — populating `tool_calls`
with `ToolCall(id=..., name=..., arguments={...})` when the model requests tools,
and filling `usage`. The stock backends in `pyhar.models` are readable examples
to copy.

## Tools

Wrap any function with `@tool`. The `input_schema` is generated from the
function's type hints via `schema_from_signature`, so real models actually see
the parameters:

```python
from pyhar import tool

@tool
def read_file(path: str, max_bytes: int = 4096) -> str:
    """Read a file."""
    ...

read_file.schema
# {'type': 'object',
#  'properties': {'path': {'type': 'string'},
#                 'max_bytes': {'type': 'integer'}},
#  'additionalProperties': False,
#  'required': ['path']}
```

Parameters without a default become `required`. Python hints map to JSON-schema
types: `str`→`string`, `int`→`integer`, `float`→`number`, `bool`→`boolean`,
`list`/`tuple`/`set`→`array`, `dict`→`object`; `Optional[X]` is unwrapped to
`X`, and anything unrecognized falls back to `string`.

`@tool` works bare or with arguments (`name=`, `description=`, `schema=`). A
`Tool` is a dataclass with `name`, `fn`, `description`, and `schema`, and it is
directly callable (`read_file(path="x")`).

### Override the schema

Pass `schema=` to bypass generation entirely — useful for enums, nested objects,
or anything richer than a flat signature:

```python
from pyhar import tool

@tool(schema={
    "type": "object",
    "properties": {"unit": {"type": "string", "enum": ["c", "f"]}},
    "required": ["unit"],
})
def temperature(unit: str) -> str:
    """Return the current temperature."""
    ...
```

A full end-to-end run with a scripted tool call:

```python
from pyhar import Harness, ScriptedModel, tool

@tool
def word_count(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())

model = ScriptedModel([
    ("tool", "word_count", {"text": "one two three"}),
    "There are 3 words.",
])
state = Harness(model, tools=[word_count]).run("Count the words in 'one two three'.")
print(state.result)     # -> "There are 3 words."
```

You can also import tools from an MCP server rather than defining them by hand —
see [Adapters, MCP & subagents](adapters-and-mcp.md).

## How cost is computed

Every backend fills `Usage.cost`. Real backends compute it from token counts and
a per-model price table (`pyhar.models.pricing`), quoted in USD per 1M tokens
`(input, output)`:

```python
from pyhar.models.pricing import cost_of, price_for

price_for("claude-opus-4-8")     # -> (5.0, 25.0)
cost_of("claude-opus-4-8", 1_000_000, 100_000)   # -> 7.5
```

`price_for` tries an exact match, then a prefix match (so dated snapshots like
`gpt-4o-2024-…` resolve), then falls back to `(0.0, 0.0)`. To override pricing
for an unlisted or custom model, pass `pricing=(input, output)` to the model
constructor:

```python
from pyhar.models import OpenAIModel

model = OpenAIModel("my-finetune", pricing=(0.5, 1.5))
```

`OllamaModel` is local and always reports `cost=0.0`. `ScriptedModel` and
`EchoModel` don't compute cost.

## Testing with an injected client

Both `AnthropicModel` and `OpenAIModel` accept a `client=` argument, so you can
drive them with a fake in tests — no network, no key. The fake just needs to
mimic the SDK's response shape:

```python
from types import SimpleNamespace
from pyhar.models import AnthropicModel

class FakeAnthropic:
    class messages:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hi from the fake")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            )

model = AnthropicModel("claude-opus-4-8", client=FakeAnthropic())
resp = model([], [])
print(resp.text)            # -> "hi from the fake"
print(resp.usage.cost)      # priced from the table: 12*5 + 4*25 per 1M
```

`OllamaModel` has the same seam via `transport=` (an injectable
`callable(url, payload) -> dict`), so it too is testable without a running
server.

---

**See also:** [Concepts](concepts.md) · [Components](components.md) ·
[Adapters, MCP & subagents](adapters-and-mcp.md) · [Cookbook](cookbook.md)
