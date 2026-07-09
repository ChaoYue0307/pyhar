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
    stop_reason: str | None = None   # provider stop/finish reason, normalized as-is
    raw: Any = None                  # the untouched provider payload, if any
```

`text` is the assistant's message, `tool_calls` is the list of calls the model
wants the harness to run this turn, `usage` carries `input_tokens`,
`output_tokens`, and `cost`, and `raw` holds the original provider object for
escape hatches. That's all a backend has to produce.

### `stop_reason`

New in 0.3.0: `Response.stop_reason` carries the provider's stop/finish reason,
passed through as-is. The stock backends fill it — Anthropic reports values like
`end_turn` / `tool_use` / `max_tokens` / `refusal`; OpenAI reports `stop` /
`tool_calls` / `length`; Ollama reports its `done_reason` (e.g. `stop` /
`length`). It defaults to `None`, so custom backends and `ScriptedModel` are
unaffected if they never set it.

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

## Combinators — retry, fallback, routing

New in 0.3.0, `pyhar.models` ships three **model combinators**: wrappers that
take model(s) and return a `Model`. Because the result satisfies the same
one-method protocol, combinators nest freely and drop into any `Harness`
unchanged:

```python
from pyhar.models import AnthropicModel, FallbackModel, OllamaModel, RetryModel

model = RetryModel(FallbackModel([AnthropicModel(), OllamaModel()]))
```

That one line reads: try Claude, fail over to the local model, and retry the
whole chain with backoff if even that fails.

### `RetryModel`

Retries the wrapped model on exception, with exponential backoff
(`min(base_delay * 2**attempt, max_delay)` between attempts). Provider SDKs
already retry HTTP 429/5xx internally — this wrapper is for everything above
that: network flaps, local servers restarting, or models with no built-in
retry. After the final attempt, the last exception is re-raised.

| Argument | Purpose |
| --- | --- |
| `model` | the wrapped `Model` (positional) |
| `max_retries` | extra attempts after the first, default `3` |
| `base_delay` | first backoff in seconds, default `1.0` |
| `max_delay` | backoff cap in seconds, default `30.0` |
| `retry_on` | exception types to retry, default `(Exception,)` — anything else propagates immediately |
| `sleep` | injectable sleeper, default `time.sleep` — pass `lambda s: None` in tests |

### `FallbackModel`

Tries each model in order; if one raises, moves to the next. If every model
fails, the last exception is raised.

| Argument | Purpose |
| --- | --- |
| `models` | sequence of models, tried in order (positional; must be non-empty) |
| `should_fallback` | `callable(exc) -> bool` deciding whether an exception triggers failover, default: any `Exception`. Returning `False` re-raises immediately |

The index of the model that served the last call is available as
`.last_served`. It is reset at the start of every call, so **after a failed
call it reads `None`** — never a stale index from an earlier success.

A runnable demo of the nesting pattern, with a dead primary and a
`ScriptedModel` backup:

```python
from pyhar import Harness, ScriptedModel
from pyhar.models import FallbackModel, RetryModel

class DownModel:
    def __call__(self, messages, tools):
        raise ConnectionError("provider is down")

fallback = FallbackModel([DownModel(), ScriptedModel(["served by the backup"])])
model = RetryModel(fallback, max_retries=2, sleep=lambda s: None)

state = Harness(model).run("hello")
print(state.result)             # -> "served by the backup"
print(fallback.last_served)     # -> 1 (index of the model that answered)
```

### `RouterModel`

Routes each call to one of several **named** models via a policy callback
`route(messages, tools) -> key`. Unknown keys fall back to `default`;
construction raises `ValueError` if `default` is not one of the model keys.

| Argument | Purpose |
| --- | --- |
| `models` | `dict[str, Model]` of named models (positional) |
| `route` | `callable(messages, tools) -> str` picking the key per call (keyword-only, required) |
| `default` | key used when `route` returns an unknown key (keyword-only, required) |

The key that served the last call is available as `.last_key` — set **only
after the routed model actually returned**, so after a failed call it reads
`None`.

### Recipe: cheap/strong tiering with `BudgetPolicy`

`RouterModel` is the cheap/strong "frontier + sidekick" tiering primitive. Pair
it with `BudgetPolicy(on_over_soft=...)`: the router closure reads a flag, and
the soft-budget callback flips it. With real backends that looks like:

```python
from pyhar import BudgetPolicy
from pyhar.models import AnthropicModel, RouterModel

tier = {"key": "strong"}
router = RouterModel(
    {"strong": AnthropicModel("claude-opus-4-8"),
     "cheap": AnthropicModel("claude-haiku-4-5")},
    route=lambda messages, tools: tier["key"],
    default="strong",
)
budget = BudgetPolicy(
    max_total_tokens=200_000,
    on_over_soft=lambda state: tier.update(key="cheap"),
)
```

And here is the same wiring as a key-free runnable demo — `soft_fraction=0.0`
makes the soft warning fire after the first turn, so the second turn is served
by the cheap tier:

```python
from pyhar import BudgetPolicy, Harness, ScriptedModel, tool
from pyhar.models import RouterModel

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tier = {"key": "strong"}
router = RouterModel(
    {"strong": ScriptedModel([("tool", "add", {"a": 2, "b": 3})]),
     "cheap": ScriptedModel(["The sum is 5."])},
    route=lambda messages, tools: tier["key"],
    default="strong",
)
budget = BudgetPolicy(
    max_total_tokens=200_000,
    soft_fraction=0.0,          # fire the soft warning immediately (demo only)
    on_over_soft=lambda state: tier.update(key="cheap"),
)

state = Harness(router, components=[budget], tools=[add]).run("What is 2 + 3?")
print(state.result)         # -> "The sum is 5."
print(router.last_key)      # -> "cheap" (the tier that served the final turn)
```

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
