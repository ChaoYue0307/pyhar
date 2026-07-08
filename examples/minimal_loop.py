"""Portability proof: the SAME components, applied by hand in a plain loop.

pyhar's own `Harness.run` is a convenience. Because components are just
objects with lifecycle hooks, you can drive them from your own while-loop (or
another runtime). This is what "composes WITH your framework, not against it"
means in practice.

Run:  python examples/minimal_loop.py
"""
from pyhar import Message, ScriptedModel, ToolOutputBudget
from pyhar.core.state import HarnessState


def main() -> None:
    model = ScriptedModel([
        ("tool", "grep", {"q": "TODO"}),
        "Found the TODOs; wrapping up.",
    ])
    tool_budget = ToolOutputBudget(max_tokens=30)      # a pyhar component
    components = [tool_budget]

    state = HarnessState()
    state.add_message(Message(role="user", content="find the TODOs"))

    # --- your own loop; pyhar hooks are called by hand ---
    for _ in range(5):
        for c in components:
            c.before_model(state)

        resp = model(state.messages, [])
        state.add_message(Message(role="assistant", content=resp.text or "",
                                  tool_calls=list(resp.tool_calls)))
        state.last_turn_had_tool_calls = bool(resp.tool_calls)

        if resp.tool_calls:
            for call in resp.tool_calls:
                raw = "match at line 0\n" + ("x" * 2000)   # a big tool result
                result = raw
                for c in components:                            # <- component runs here
                    result = c.after_tool(state, call, result)
                state.add_message(Message(role="tool", content=result,
                                          tool_call_id=call.id, name=call.name))
        else:
            state.result = resp.text
            break

    print("result:", state.result)
    print("tokens saved by ToolOutputBudget:", state.memory.get("_tool_savings", 0))
    print("full output preserved in sandbox:", bool(state.memory.get("_sandbox")))


if __name__ == "__main__":
    main()
