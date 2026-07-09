"""Resilient + cost-aware models: retry, failover, and cheap/strong routing.

Model combinators are just Models, so they nest and drop into any Harness:

    RetryModel(FallbackModel([primary, backup]))

`RouterModel` + `BudgetPolicy` implements the frontier/sidekick pattern: start
on the strong model, and once the soft token budget trips, route the remaining
turns to the cheap one.

Run:  python examples/model_routing.py
"""
from pyhar import BudgetPolicy, Harness, ScriptedModel, tool
from pyhar.models import FallbackModel, RetryModel, RouterModel


class FlakyModel:
    """Simulates a provider that fails twice before recovering."""

    def __init__(self, inner):
        self.inner, self.attempts = inner, 0

    def __call__(self, messages, tools):
        self.attempts += 1
        if self.attempts <= 2:
            raise ConnectionError("simulated outage")
        return self.inner(messages, tools)


@tool
def lookup(q: str) -> str:
    return f"result for {q}"


def demo_retry_and_fallback() -> None:
    flaky = FlakyModel(ScriptedModel(["recovered and answered"]))
    model = RetryModel(flaky, max_retries=3, base_delay=0.01)
    state = Harness(model).run("q")
    print(f"retry:    {state.result!r} after {flaky.attempts} attempts")

    dead = FlakyModel(ScriptedModel(["never"]))
    dead.attempts = -10**9  # always failing
    backup = ScriptedModel(["served by the backup model"])
    fb = FallbackModel([dead, backup])
    state = Harness(fb).run("q")
    print(f"fallback: {state.result!r} (served index {fb.last_served})")


def demo_budget_tiering() -> None:
    tier = {"key": "strong"}
    router = RouterModel(
        {
            "strong": ScriptedModel([("tool", "lookup", {"q": "deep question"}),
                                     "strong model finished the hard part"]),
            "cheap": ScriptedModel(["cheap model wrapped up the rest"]),
        },
        route=lambda msgs, tools: tier["key"],
        default="strong",
    )
    budget = BudgetPolicy(
        max_total_tokens=100_000,
        soft_fraction=0.0000001,  # trip immediately for the demo
        on_over_soft=lambda state: tier.update(key="cheap"),
    )
    state = Harness(router, components=[budget], tools=[lookup]).run("hard task")
    print(f"tiering:  {state.result!r} (last served: {router.last_key})")


if __name__ == "__main__":
    demo_retry_and_fallback()
    demo_budget_tiering()
