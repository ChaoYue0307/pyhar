"""Tests for pyhar.optimize — trace-guided harness-config search."""
import json

import pytest

from pyhar import Choice, Objective, Range, ScriptedModel, harness_from_config, tool, tune
from pyhar.optimize import EvalSummary, hints_from, resolve

# -- space primitives ----------------------------------------------------------

def test_choice_and_range_validation():
    with pytest.raises(ValueError):
        Choice("only-one")
    with pytest.raises(ValueError):
        Range(5, 5)
    with pytest.raises(ValueError):
        Range(0, 10, step=0)


def test_range_defaults_sampling_and_shift():
    import random
    r = Range(100, 900, step=200)
    assert r.default() == 500
    rng = random.Random(0)
    for _ in range(20):
        v = r.sample(rng)
        assert 100 <= v <= 900
    assert r.shift(500, -1, rng) == 300
    assert r.shift(100, -1, rng) == 100  # clamped at the edge
    assert isinstance(r.shift(500, 1, rng), int)


def test_choice_shift_is_directional_for_numeric():
    import random
    c = Choice(500, 1000, 2000)
    rng = random.Random(0)
    assert c.shift(1000, -1, rng) == 500
    assert c.shift(1000, +1, rng) == 2000
    assert c.shift(2000, +1, rng) == 2000  # edge


def test_resolve_defaults_and_json_clean():
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 900, step=200)}},
            {"name": "compactor", "args": {"target_tokens": Choice(500, 1000)}, "optional": True},
            "loop_guard",
        ],
        "budget": {"max_context_tokens": Choice(1000, 2000)},
        "max_turns": Range(2, 10, step=2),
    }
    config = resolve(space)
    json.dumps(config)  # JSON-able, no marker objects left
    assert config["components"][0]["args"]["max_tokens"] == 500
    assert config["components"][1] == {"name": "compactor", "args": {"target_tokens": 500}}
    assert config["components"][2] == "loop_guard"
    assert config["max_turns"] == 6
    assert "optional" not in json.dumps(config)
    # the default config is buildable
    harness_from_config(config, model=ScriptedModel(["ok"]))


# -- the "gradient": hints ---------------------------------------------------------

def _summary(memories, success_rate=1.0):
    return EvalSummary(success_rate=success_rate, mean_total_tokens=100,
                       mean_input_tokens=80, mean_cost=0.0, mean_turns=1.0,
                       runs=len(memories), memories=memories)


def test_hint_tighten_unfired_tool_budget():
    config = {"components": [{"name": "tool_output_budget", "args": {"max_tokens": 500}}]}
    hints = hints_from(config, _summary([{"_tool_savings": 0}, {}]))
    assert any(h.path == ("components", "tool_output_budget", "args", "max_tokens")
               and h.direction == -1 for h in hints)


def test_hint_raise_turn_cap_on_failures():
    config = {"max_turns": 2, "components": []}
    hints = hints_from(config, _summary([{"_stop_reason": "max_turns"}], success_rate=0.0))
    assert any(h.path == ("max_turns",) and h.direction == +1 for h in hints)


def test_hint_loosen_budget_when_failing_with_truncation():
    config = {"components": [{"name": "tool_output_budget", "args": {"max_tokens": 100}}]}
    hints = hints_from(config, _summary([{"_tool_savings": 900}], success_rate=0.5))
    assert any(h.direction == +1 for h in hints)


# -- end-to-end: the tuner discovers better configs -------------------------------

BIG = "x" * 4000  # ~1000 tokens of tool output


@tool
def read_log(path: str) -> str:
    """Return a big log blob."""
    return BIG


def _log_model():
    return ScriptedModel([("tool", "read_log", {"path": "app.log"}), "done"])


def test_tune_tightens_unfired_tool_budget():
    # default max_tokens=1500 > output size -> budget never fires -> hint says
    # tighten -> lower thresholds shrink the big output -> fewer input tokens.
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
        ],
    }
    report = tune(
        space,
        model_factory=_log_model,
        tasks=["summarize the log"],
        tools=[read_log],
        budget_runs=12,
        init_samples=1,   # defaults only, so improvement must come from hints
        seed=7,
    )
    start = report.steps[0]
    assert start.config["components"][0]["args"]["max_tokens"] == 1500
    best_max = report.best_config["components"][0]["args"]["max_tokens"]
    assert best_max < 1500                       # tightened
    assert report.best_score > start.score      # and measurably better
    assert "never fired" in report.explain()    # the trace signal is named
    assert report.runs_used <= 12


def test_tune_raises_turn_cap_to_reach_success():
    # the task needs 3 turns (two tool calls then the answer); default cap is 2
    def model_factory():
        return ScriptedModel([
            ("tool", "read_log", {"path": "a"}),
            ("tool", "read_log", {"path": "b"}),
            "finished",
        ])

    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": 200}},  # fixed, not a knob
        ],
        "max_turns": Range(1, 4, step=1),  # default = 2 -> fails
    }
    report = tune(
        space,
        model_factory=model_factory,
        tasks=["do the two-step task"],
        tools=[read_log],
        budget_runs=10,
        init_samples=1,
        seed=3,
    )
    assert report.steps[0].summary.success_rate == 0.0
    assert report.best_summary.success_rate == 1.0
    assert report.best_config["max_turns"] >= 3
    assert "turn cap" in report.explain()


def test_tune_is_deterministic_for_a_seed():
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
            {"name": "compactor", "args": {"target_tokens": Choice(500, 1500)}, "optional": True},
        ],
    }

    def run():
        return tune(space, model_factory=_log_model, tasks=["t"], tools=[read_log],
                    budget_runs=10, seed=42)

    a, b = run(), run()
    assert [s.config for s in a.steps] == [s.config for s in b.steps]
    assert a.best_config == b.best_config and a.best_score == b.best_score


def test_objective_token_penalty_breaks_ties():
    # both configs succeed; the tighter budget wins only because tokens are penalized
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Choice(2900, 100)}},
        ],
    }
    report = tune(
        space,
        model_factory=_log_model,
        tasks=["summarize"],
        tools=[read_log],
        objective=Objective(token_penalty_per_1k=0.05),
        budget_runs=8,
        seed=1,
    )
    assert report.best_summary.success_rate == 1.0
    assert report.best_config["components"][0]["args"]["max_tokens"] == 100


def test_tune_budget_respected_and_validated():
    space = {"components": [{"name": "tool_output_budget",
                             "args": {"max_tokens": Range(100, 900, step=200)}}]}
    with pytest.raises(ValueError, match="cannot cover one evaluation"):
        tune(space, model_factory=_log_model, tasks=["a", "b", "c"],
             tools=[read_log], budget_runs=2)

    report = tune(space, model_factory=_log_model, tasks=["a", "b"],
                  tools=[read_log], budget_runs=7, seed=0)
    assert report.runs_used <= 7 - (7 % 2) or report.runs_used <= 7


def test_best_config_round_trips_through_harness_from_config():
    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 900, step=400)}},
            "tracer",
        ],
        "max_turns": Choice(5, 10),
    }
    report = tune(space, model_factory=_log_model, tasks=["go"], tools=[read_log],
                  budget_runs=6, seed=0)
    json.dumps(report.best_config)  # shareable data
    h = harness_from_config(report.best_config, model=_log_model(), tools=[read_log])
    state = h.run("go")
    assert state.done and state.memory["_trace"]


def test_duplicate_component_names_rejected():
    space = {"components": [
        {"name": "tracer"},
        {"name": "tracer", "optional": True},
    ]}
    with pytest.raises(ValueError, match="duplicate component"):
        tune(space, model_factory=_log_model, tasks=["x"], budget_runs=5)


def test_report_table_and_explain_render():
    space = {"components": [{"name": "tool_output_budget",
                             "args": {"max_tokens": Range(100, 2900, step=700)}}]}
    report = tune(space, model_factory=_log_model, tasks=["t"], tools=[read_log],
                  budget_runs=8, seed=7)
    out = report.table()
    assert "score" in out and "defaults" in out
    assert isinstance(report.explain(), str)


# -- regression tests for the 0.5.0 review findings ------------------------------

def test_aborted_runs_charged_real_usage_not_zero():
    # HIGH: a config that aborts on the hard budget must NOT score as free
    from pyhar import Budget, BudgetExceeded, Harness

    # BudgetExceeded now carries the state with real usage
    h = Harness(ScriptedModel([("tool", "read_log", {"path": "x"})] * 9, output_tokens=10**6),
                tools=[read_log], budget=Budget(max_total_tokens=10))
    try:
        h.run("go")
        raise AssertionError("expected BudgetExceeded")
    except BudgetExceeded as e:
        assert e.state is not None and e.state.usage.total_tokens > 10

    # and tune() must prefer the honest config over the aborting one
    space = {"budget": {"max_total_tokens": Choice(5_000_000, 10)}}

    def spendy_model():
        return ScriptedModel(["done"], output_tokens=400_000)

    report = tune(space, model_factory=spendy_model, tasks=["t"],
                  objective=Objective(token_penalty_per_1k=0.003),
                  budget_runs=6, seed=0)
    assert report.best_config["budget"]["max_total_tokens"] == 5_000_000
    assert report.best_summary.success_rate == 1.0


def test_unknown_space_keys_rejected():
    with pytest.raises(ValueError, match="unknown space keys"):
        tune({"maxturns": Range(2, 10)}, model_factory=_log_model,
             tasks=["t"], budget_runs=5)
    with pytest.raises(ValueError, match="unknown space keys"):
        resolve({"componnets": []})


def test_int_range_rejects_fractional_step():
    with pytest.raises(ValueError, match="whole number"):
        Range(1, 10, step=0.4)
    assert Range(1, 10, step=2.0).step == 2  # integral float is fine


def test_range_sample_can_reach_hi():
    import random
    r = Range(0.0, 1.0, step=0.25)
    rng = random.Random(0)
    assert any(r.sample(rng) == 1.0 for _ in range(200))


def test_choice_with_none_is_tunable():
    # a None VALUE must not be confused with "component absent"
    space = {"system": Choice(None, "be terse"), "components": ["tracer"]}
    report = tune(space, model_factory=_log_model, tasks=["t"], tools=[read_log],
                  budget_runs=6, init_samples=1, seed=0)
    assert any(s.config.get("system") == "be terse" for s in report.steps)


def test_success_fn_exception_is_failure_not_crash():
    def bad_check(state):
        return json.loads(state.result)["ok"]  # prose answer -> JSONDecodeError

    report = tune({"components": ["tracer"], "max_turns": Choice(5, 6)},
                  model_factory=_log_model, tasks=[("t", bad_check)],
                  tools=[read_log], budget_runs=4, seed=0)
    assert report.best_summary.success_rate == 0.0  # failed, not crashed
    assert any("_success_fn_error" in m for m in report.best_summary.memories)


def test_callable_component_args_survive_dedupe():
    from pyhar.optimize import _key
    check = lambda s: (True, "")  # noqa: E731
    cfg = {"components": [{"name": "verifier", "args": {"check": check, "max_retries": 2}}]}
    assert _key(cfg) == _key(cfg)  # no TypeError, identity-stable


def test_verifier_failure_counts_as_unsuccessful_by_default():
    # default success must respect a Verifier that exhausted retries
    def never_pass(state):
        return (False, "nope")

    space = {"components": [
        {"name": "verifier", "args": {"check": never_pass, "max_retries": 1}},
    ]}
    report = tune(space, model_factory=lambda: ScriptedModel(["a", "b", "c"]),
                  tasks=["t"], budget_runs=3, seed=0)
    assert report.best_summary.success_rate == 0.0  # _verified False -> not a success


def test_trials_zero_rejected():
    with pytest.raises(ValueError, match="trials must be >= 1"):
        tune({"components": ["tracer"]}, model_factory=_log_model,
             tasks=["t"], budget_runs=5, trials=0)
