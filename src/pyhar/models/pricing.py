"""Rough per-model pricing so ``Usage.cost`` is populated.

USD per 1M tokens (input, output). Approximate and provider-published rates
change — override via ``Model(pricing=(in, out))`` or edit here. Local models
(Ollama) are free.
"""
from __future__ import annotations

# (input $/1M, output $/1M)
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    # OpenAI (approximate)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
}


def price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    # prefix match (e.g. dated snapshots) then give up to (0, 0)
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return (0.0, 0.0)


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000
