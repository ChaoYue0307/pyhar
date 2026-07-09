"""Tune a harness config from run traces — the "autograd from traces" seed.

The tuner runs candidate configs on your tasks, reads the trace signals the
components leave behind (budget never fired, turn cap hit, ...), proposes
directional config changes, and keeps only those that measurably improve the
objective. Everything is seeded and reproducible; the winning config is plain
JSON you can check in and load with `harness_from_config`.

Run:  python examples/tune_harness.py
"""
import json

from pyhar import Choice, Range, ScriptedModel, harness_from_config, tool, tune


@tool
def read_log(path: str) -> str:
    """Return a big log blob (the kind that bloats context)."""
    return "decision: use SQLite\n" + ("verbose log line\n" * 500) + "TODO: add index"


def make_model():
    # fresh ScriptedModel per run (it is consumed); swap for a real backend factory
    return ScriptedModel([
        ("tool", "read_log", {"path": "app.log"}),
        "Analysis complete — the store is SQLite; add the index.",
    ])


def main() -> None:
    # the search space: a config template where some values are knobs
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
            {"name": "compactor", "args": {"target_tokens": Choice(500, 1500, 3000)},
             "optional": True},
            "loop_guard",
        ],
        "max_turns": Range(2, 10, step=2),
    }

    report = tune(
        space,
        model_factory=make_model,
        tasks=["read the log and summarize the findings"],
        tools=[read_log],
        budget_runs=14,
        seed=7,
    )

    print(report.table())
    print("\nwhy the winning config won:")
    print(report.explain())

    print("\nbest config (plain JSON — check it in):")
    print(json.dumps(report.best_config, indent=2))

    baseline, best = report.steps[0], report.best_summary
    saved = baseline.summary.mean_total_tokens - best.mean_total_tokens
    pct = 100 * saved / baseline.summary.mean_total_tokens
    print(f"\ntokens: {baseline.summary.mean_total_tokens:.0f} -> "
          f"{best.mean_total_tokens:.0f}  ({pct:.0f}% saved, success "
          f"{best.success_rate:.0%}, {report.runs_used} runs spent)")

    # the tuned config is immediately usable
    harness = harness_from_config(report.best_config, model=make_model(), tools=[read_log])
    state = harness.run("read the log and summarize the findings")
    print("re-run with tuned config:", state.result)


if __name__ == "__main__":
    main()
