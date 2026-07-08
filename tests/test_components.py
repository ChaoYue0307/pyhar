from pyhar import (
    Budget,
    Compactor,
    Harness,
    Message,
    ScriptedModel,
    ToolOutputBudget,
    Verifier,
    bench,
    tool,
)
from pyhar.core.state import HarnessState

# -- ToolOutputBudget ------------------------------------------------------

def test_tool_output_budget_shrinks_and_sandboxes():
    tob = ToolOutputBudget(max_tokens=20)
    state = HarnessState()
    from pyhar import ToolCall

    big = "x" * 4000
    out = tob.after_tool(state, ToolCall(id="c1", name="read", arguments={}), big)

    assert len(out) < len(big)
    assert state.memory["_sandbox"]["c1"] == big     # full fidelity preserved
    assert state.memory["_tool_savings"] > 0


def test_tool_output_budget_leaves_small_results_alone():
    tob = ToolOutputBudget(max_tokens=100)
    state = HarnessState()
    from pyhar import ToolCall

    out = tob.after_tool(state, ToolCall(id="c1", name="read", arguments={}), "tiny")
    assert out == "tiny"
    assert "_sandbox" not in state.memory


# -- Compactor -------------------------------------------------------------

def test_compactor_reduces_tokens_and_preserves_decisions():
    state = HarnessState(budget=Budget(max_context_tokens=100))
    state.add_message(Message(role="system", content="you are a coding agent"))
    for i in range(8):
        state.add_message(Message(role="assistant", content=f"decision: use approach {i}"))
        state.add_message(Message(role="tool", content="verbose log " * 200, name="run"))
    before = state.count_tokens()

    Compactor(target_tokens=100).before_model(state)
    after = state.count_tokens()

    assert after < before
    # system message survives; a compaction happened
    assert state.messages[0].role == "system"
    assert state.memory.get("_compactions")
    # at least one preserved decision made it into the synopsis
    synopsis = " ".join(m.content for m in state.messages if m.meta.get("compacted"))
    assert "decision" in synopsis


# -- Verifier --------------------------------------------------------------

def test_verifier_retries_until_check_passes():
    model = ScriptedModel(["wrong answer", "the answer is 42"])

    def check(state):
        text = (state.result or state.messages[-1].content).lower()
        return ("42" in text, "must contain 42")

    state = Harness(model, components=[Verifier(check, max_retries=2)]).run("q")

    assert state.memory["_verified"] is True
    assert "42" in (state.result or "")
    assert state.turn == 2  # one failed candidate, one successful


def test_verifier_check_can_read_state_result_directly():
    # the candidate answer is exposed as state.result BEFORE after_turn, so a
    # check that reads only state.result works (no need to dig into messages)
    model = ScriptedModel(["nope", "the answer is 42"])

    def check(state):
        return ("42" in (state.result or ""), "must contain 42")

    state = Harness(model, components=[Verifier(check, max_retries=2)]).run("q")
    assert state.memory["_verified"] is True
    assert state.result == "the answer is 42"
    assert state.turn == 2


def test_verifier_gives_up_after_max_retries():
    model = ScriptedModel(["nope", "still nope", "nope again", "and again"])

    def check(state):
        return (False, "never passes")

    state = Harness(model, components=[Verifier(check, max_retries=2)]).run("q")

    assert state.memory["_verified"] is False
    assert state.done  # stops after retries are exhausted, does not loop forever


# -- Bench -----------------------------------------------------------------

def test_bench_shows_measurable_tool_budget_win():
    @tool
    def read_file(path: str) -> str:
        return "y" * 8000

    def build(with_budget: bool):
        def factory():
            comps = [ToolOutputBudget(max_tokens=50)] if with_budget else []
            model = ScriptedModel([("tool", "read_file", {"path": "big"}), "done"])
            return Harness(model, components=comps, tools=[read_file])
        return factory

    report = bench(
        "read a big file then answer",
        {"baseline": build(False), "with_tool_budget": build(True)},
        success=lambda s: s.done,
    )

    by_name = {r.name: r for r in report.runs}
    assert by_name["baseline"].success and by_name["with_tool_budget"].success
    # the second model call sees a much smaller tool message -> fewer input tokens
    assert by_name["with_tool_budget"].input_tokens < by_name["baseline"].input_tokens
