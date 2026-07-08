"""A safe, observable agent: gate destructive tools + trace every step.

`Permissions` denies dangerous tools before they run; `Tracer` streams a
structured event log. Both are just components — drop them into any harness.

Run:  python examples/permissions_and_tracing.py
"""
from pyhar import Harness, Permissions, ScriptedModel, ToolOutputBudget, Tracer, tool


@tool
def read_file(path: str) -> str:
    """Read a file (schema auto-generated from the type hint)."""
    return "contents of " + path


@tool
def delete_everything(path: str) -> str:
    """A destructive tool we do NOT want the agent to actually call."""
    raise RuntimeError("this should never run — it's denied")


def main() -> None:
    # the model tries the destructive tool first, then a safe read, then answers
    model = ScriptedModel([
        ("tool", "delete_everything", {"path": "/"}),
        ("tool", "read_file", {"path": "README.md"}),
        "Done — I read README.md and left everything else alone.",
    ])

    harness = Harness(
        model,
        components=[
            Permissions(deny=["delete_everything"]),   # gate: block destructive tools
            ToolOutputBudget(max_tokens=200),           # keep tool output small
            Tracer(sink=lambda e: print("  trace:", e)),  # live event log
        ],
        tools=[read_file, delete_everything],
    )
    state = harness.run("Clean up the repo.")

    print("\nresult: ", state.result)
    print("denied: ", state.memory.get("_denied"))
    print("read_file schema:", read_file.schema)


if __name__ == "__main__":
    main()
