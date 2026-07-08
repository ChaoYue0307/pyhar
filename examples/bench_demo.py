"""Measure the win: A/B a bare harness vs a tuned one on the same task.

`bench` runs each config from a fresh factory and reports tokens/cost/turns, so
"tool-output budgeting saves tokens" is a number you can reproduce, not a claim.

Run:  python examples/bench_demo.py
"""
from pyhar import Compactor, Harness, ScriptedModel, ToolOutputBudget, bench, tool


@tool
def read_log(path: str) -> str:
    """Return a large log blob (the kind of output that blows up context)."""
    return "decision: use SQLite\n" + ("verbose log line\n" * 500) + "TODO: add index"


def make_config(tuned: bool):
    def factory() -> Harness:
        # identical task/model; the only difference is whether we add the primitives
        model = ScriptedModel([
            ("tool", "read_log", {"path": "app.log"}),
            "Analysis complete — the store is SQLite; add the index.",
        ])
        components = []
        if tuned:
            components = [ToolOutputBudget(max_tokens=200), Compactor(target_tokens=1500)]
        return Harness(model, components=components, tools=[read_log])
    return factory


def main() -> None:
    report = bench(
        "read a big log then summarize",
        {
            "baseline": make_config(tuned=False),
            "tuned (budget+compact)": make_config(tuned=True),
        },
        success=lambda s: s.done,
    )
    print(report.table())

    base = next(r for r in report.runs if r.name == "baseline")
    tuned = next(r for r in report.runs if r.name.startswith("tuned"))
    saved = base.input_tokens - tuned.input_tokens
    pct = 100 * saved / base.input_tokens if base.input_tokens else 0
    print(f"\ninput tokens saved by the primitives: {saved} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
