"""Streaming: watch the answer arrive token-by-token via the on_delta hook.

Any component can implement `on_delta(state, delta)`; the harness fans out text
chunks as the model produces them when constructed with `stream=True`. Models
that can't stream degrade gracefully (full response, no deltas).

Run:  python examples/streaming.py
"""
from pyhar import Component, Harness, ScriptedModel, Tracer, tool


class LivePrinter(Component):
    """Prints deltas as they arrive — the '// streaming UI' of this demo."""

    def on_delta(self, state, delta: str) -> None:
        print(delta, end="", flush=True)


@tool
def look_up(topic: str) -> str:
    """Pretend knowledge-base lookup."""
    return f"notes about {topic}"


def main() -> None:
    model = ScriptedModel([
        ("tool", "look_up", {"topic": "harness design"}),
        "Streaming answers feel alive: the reader sees progress "
        "word by word instead of waiting for the whole reply.",
    ])

    harness = Harness(
        model,
        components=[LivePrinter(), Tracer(include_deltas=True)],
        tools=[look_up],
        stream=True,
    )
    print("assistant> ", end="")
    state = harness.run("Explain why streaming matters.")
    print()  # newline after the streamed text

    deltas = [e for e in state.memory["_trace"] if e["event"] == "delta"]
    print(f"\n({len(deltas)} deltas streamed; final result identical to state.result)")

    # swap in a real streaming backend with no other changes:
    #   from pyhar.models import AnthropicModel
    #   harness = Harness(AnthropicModel("claude-opus-4-8"), components=[LivePrinter()], stream=True)


if __name__ == "__main__":
    main()
