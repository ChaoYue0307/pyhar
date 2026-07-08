"""End-to-end demo: a coding-style harness with a scripted model (no API key).

Run:  python examples/react_agent.py
"""
from pyhar import ScriptedModel, tool
from pyhar.presets import coding_agent


@tool
def read_file(path: str) -> str:
    """Return the contents of a file (here: a big fake blob)."""
    return "decision: use SQLite\n" + ("some verbose log line\n" * 400) + "TODO: add index"


def main() -> None:
    # The model: read a file, then answer. In real use, wrap Anthropic/OpenAI.
    model = ScriptedModel([
        ("tool", "read_file", {"path": "db.py"}),
        "Done — the store uses SQLite; I added the index. The answer is 42.",
    ])

    # Verify the final answer actually contains the expected token.
    def check(state):
        text = (state.result or state.messages[-1].content).lower()
        return ("42" in text, "answer must contain '42'")

    harness = coding_agent(model, tools=[read_file], check=check, context_tokens=300)
    state = harness.run("Inspect db.py and tell me the answer.")

    print("result:  ", state.result)
    print("turns:   ", state.turn)
    print("verified:", state.memory.get("_verified"))
    print("tool tokens saved:", state.memory.get("_tool_savings", 0))
    print("compactions:      ", state.memory.get("_compactions", []))
    print("usage:   ", state.usage)


if __name__ == "__main__":
    main()
