"""Model combinators — resilience and routing as composable ``Model`` wrappers.

A combinator takes model(s) and returns a ``Model``, so they nest freely and
drop into any ``Harness`` unchanged:

    model = RetryModel(FallbackModel([AnthropicModel(...), OllamaModel(...)]))

- ``RetryModel``    — retry a failing model with exponential backoff.
- ``FallbackModel`` — try models in order; on failure, move to the next.
- ``RouterModel``   — route each call to a named model via your policy
  (the cheap/strong "frontier + sidekick" tiering pattern).

All three also implement ``stream(...)``, forwarding to the wrapped model's
``stream`` when it has one (so ``Harness(stream=True)`` keeps streaming through
a combinator). A wrapped model without ``stream`` is called normally — the turn
completes, just without deltas. NOTE: if an attempt fails mid-stream and is
retried (RetryModel) or failed over (FallbackModel), the next attempt re-emits
its text from the start — downstream UIs should treat a new turn's deltas as
replacing, not strictly appending, on provider errors.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..core.model import Model, OnDelta, Response
from ..core.state import Message
from ..core.tool import Tool

Sleeper = Callable[[float], None]


def _invoke(model: Model, messages: list[Message], tools: list[Tool],
            on_delta: OnDelta | None) -> Response:
    """Call ``model.stream`` when streaming is requested and available,
    else the plain call (no deltas)."""
    if on_delta is not None:
        stream_fn = getattr(model, "stream", None)
        if callable(stream_fn):
            return stream_fn(messages, tools, on_delta=on_delta)
    return model(messages, tools)


class RetryModel:
    """Retry the wrapped model on exception, with exponential backoff.

    Note: provider SDKs already retry HTTP 429/5xx internally — this wrapper is
    for everything above that (network flaps, local servers restarting, or
    models with no built-in retry). ``retry_on`` filters which exceptions are
    retried; anything else propagates immediately.
    """

    def __init__(
        self,
        model: Model,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        sleep: Sleeper = time.sleep,
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on
        self._sleep = sleep

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        return self._run(messages, tools, None)

    def stream(self, messages: list[Message], tools: list[Tool], *, on_delta: OnDelta) -> Response:
        """Streaming with the same retry semantics. A retried attempt re-emits
        its deltas from the start (see module docstring)."""
        return self._run(messages, tools, on_delta)

    def _run(self, messages: list[Message], tools: list[Tool], on_delta: OnDelta | None) -> Response:
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return _invoke(self.model, messages, tools, on_delta)
            except self.retry_on as e:  # noqa: PERF203 - retry loop by design
                last = e
                if attempt == self.max_retries:
                    break
                self._sleep(min(self.base_delay * (2**attempt), self.max_delay))
        assert last is not None
        raise last


class FallbackModel:
    """Try each model in order; if one raises, move to the next.

    ``should_fallback(exc)`` decides whether an exception triggers failover
    (default: any Exception). If every model fails, the last exception is
    raised. The index of the model that served each call is available as
    ``.last_served``.
    """

    def __init__(
        self,
        models: Sequence[Model],
        *,
        should_fallback: Callable[[BaseException], bool] = lambda e: True,
    ):
        if not models:
            raise ValueError("FallbackModel needs at least one model")
        self.models = list(models)
        self.should_fallback = should_fallback
        self.last_served: int | None = None

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        return self._run(messages, tools, None)

    def stream(self, messages: list[Message], tools: list[Tool], *, on_delta: OnDelta) -> Response:
        """Streaming with the same failover semantics. A failed-over attempt
        re-emits its deltas from the start (see module docstring)."""
        return self._run(messages, tools, on_delta)

    def _run(self, messages: list[Message], tools: list[Tool], on_delta: OnDelta | None) -> Response:
        self.last_served = None  # a failed call reads as None, never stale
        last: BaseException | None = None
        for i, model in enumerate(self.models):
            try:
                resp = _invoke(model, messages, tools, on_delta)
                self.last_served = i
                return resp
            except Exception as e:
                last = e
                if not self.should_fallback(e):
                    raise
        assert last is not None
        raise last


class RouterModel:
    """Route each call to one of several named models via a policy callback.

    ``route(messages, tools) -> key`` picks the model per call; unknown keys
    fall back to ``default``. This is the cheap/strong tiering primitive — pair
    it with ``BudgetPolicy(on_over_soft=...)`` by flipping a flag the router
    closure reads:

        tier = {"key": "strong"}
        router = RouterModel(
            {"strong": AnthropicModel("claude-opus-4-8"),
             "cheap": AnthropicModel("claude-haiku-4-5")},
            route=lambda msgs, tools: tier["key"],
            default="strong",
        )
        budget = BudgetPolicy(
            max_total_tokens=200_000,
            on_over_soft=lambda state: tier.update(key="cheap"),
        )

    The key that served each call is available as ``.last_key``.
    """

    def __init__(
        self,
        models: dict[str, Model],
        *,
        route: Callable[[list[Message], list[Tool]], str],
        default: str,
    ):
        if default not in models:
            raise ValueError(f"default {default!r} is not one of {sorted(models)}")
        self.models = dict(models)
        self.route = route
        self.default = default
        self.last_key: str | None = None

    def __call__(self, messages: list[Message], tools: list[Tool]) -> Response:
        return self._run(messages, tools, None)

    def stream(self, messages: list[Message], tools: list[Tool], *, on_delta: OnDelta) -> Response:
        """Streaming through whichever model the policy routes to."""
        return self._run(messages, tools, on_delta)

    def _run(self, messages: list[Message], tools: list[Tool], on_delta: OnDelta | None) -> Response:
        self.last_key = None  # a failed call reads as None, never stale
        key = self.route(messages, tools)
        if key not in self.models:
            key = self.default
        resp = _invoke(self.models[key], messages, tools, on_delta)
        self.last_key = key  # set only after the model actually served the call
        return resp
