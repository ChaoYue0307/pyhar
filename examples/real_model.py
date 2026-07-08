"""Run the coding-agent harness against a REAL model.

Pick a backend by env: set ANTHROPIC_API_KEY (uses claude-opus-4-8), or
OPENAI_API_KEY, or run a local Ollama server. Falls back to the key-free
ScriptedModel so the file always runs.

    ANTHROPIC_API_KEY=... python examples/real_model.py
    OPENAI_API_KEY=...    python examples/real_model.py
    OLLAMA=1              python examples/real_model.py   # needs `ollama serve`
"""
import os

from pyhar import ScriptedModel, tool
from pyhar.presets import coding_agent


@tool
def read_file(path: str) -> str:
    """Read a file (demo returns a fixed blob)."""
    return "decision: use SQLite\n" + "log line\n" * 200 + "TODO: add an index"


def pick_model():
    if os.getenv("ANTHROPIC_API_KEY"):
        from pyhar.models import AnthropicModel
        return AnthropicModel("claude-opus-4-8", max_tokens=1024), "AnthropicModel"
    if os.getenv("OPENAI_API_KEY"):
        from pyhar.models import OpenAIModel
        return OpenAIModel("gpt-4o-mini"), "OpenAIModel"
    if os.getenv("OLLAMA"):
        from pyhar.models import OllamaModel
        return OllamaModel(os.getenv("OLLAMA_MODEL", "llama3.1")), "OllamaModel"
    # key-free fallback so the example always runs
    return ScriptedModel([("tool", "read_file", {"path": "db.py"}), "Done — the answer is 42."]), "ScriptedModel"


def main() -> None:
    model, name = pick_model()
    print(f"backend: {name}")

    harness = coding_agent(model, tools=[read_file], context_tokens=1000)
    state = harness.run("Inspect db.py and tell me what storage it uses, then the answer.")

    print("result:", state.result)
    print("turns: ", state.turn)
    print("usage: ", state.usage)
    print("tool tokens saved:", state.memory.get("_tool_savings", 0))


if __name__ == "__main__":
    main()
