# Tuning — trace-guided config search

`pyhar.optimize` closes the loop the rest of the library sets up: harness
compositions are data ([`harness_from_config`](cookbook.md#14-ship-a-harness-as-shareable-config)),
runs are measurable ([`bench`](cookbook.md#5-keep-tool-output-small--and-prove-it)),
and components leave **trace signals** in `state.memory`. `tune` searches a
config space using those signals as its "gradient".

Honest framing up front: this is **seeded, greedy local search** — mutations
are *proposed* from run evidence and *accepted* only when the objective score
measurably improves. Every accepted change is attributable to a named signal
(`report.explain()`). It is not a prompt optimizer and makes no
global-optimality claims.

```bash
pip install pyhar-agents   # zero new dependencies
```

## The loop

```
resolve defaults ──► evaluate on your tasks ──► read trace signals
      ▲                                              │
      │                                              ▼
keep only if score improves ◄── mutate a knob ◄── directional hints
```

1. **Forward pass** — build the candidate with `harness_from_config`, run it on
   every task (`len(tasks) × trials` runs per candidate).
2. **"Gradient"** — `hints_from(config, summary)` maps evidence to directions:

   | Trace signal | Hint |
   | --- | --- |
   | `_tool_savings == 0` with a `tool_output_budget` present | tighten `max_tokens` — the budget never fired |
   | failures + `_tool_savings > 0` | loosen `max_tokens` — possible over-truncation |
   | `_compactions` empty with a `compactor` present | lower `target_tokens`, or drop the component |
   | failures + compactions happened | raise `target_tokens` — possible over-compaction |
   | failures + `_stop_reason == "max_turns"` | raise the turn cap |
   | failures + `_verified is False` | raise `verifier.max_retries` |
   | a run aborted on the hard token budget | raise `budget.max_total_tokens` |

3. **Step** — apply one hinted mutation (80% of the time, else explore),
   re-measure, keep it only if the score strictly improves.

## Defining a space

A space is a normal config template where some values are knobs:

```python
from pyhar.optimize import Choice, Range

space = {
    "components": [
        {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
        {"name": "compactor", "args": {"target_tokens": Choice(500, 1500, 3000)},
         "optional": True},                      # the tuner may include or drop it
        "loop_guard",                             # fixed — not a knob
    ],
    "max_turns": Range(2, 10, step=2),
    "budget": {"max_context_tokens": Choice(1000, 2000)},
}
```

- `Range(lo, hi, step=...)` — numeric; defaults to the midpoint; mutations move
  one `step`, clamped at the edges. Integer in, integer out (integer ranges
  require whole-number steps; use float bounds for a float knob).
- `Choice(a, b, c)` — categorical; defaults to the first option; numeric
  choices shift directionally, others resample.
- `"optional": True` — the component itself becomes a knob (include/drop).
- Each component name may appear once — knobs are addressed by name.

## Running the tuner

```python
from pyhar import Range, ScriptedModel, tool, tune

@tool
def read_log(path: str) -> str:
    """A big log blob."""
    return "log line\n" * 500

def make_model():   # FRESH model per run — ScriptedModel is consumed by a run
    return ScriptedModel([("tool", "read_log", {"path": "app.log"}), "done"])

space = {"components": [
    {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 2900, step=700)}},
]}

report = tune(
    space,
    model_factory=make_model,
    tasks=["summarize the log"],       # or (task, success_fn) tuples
    tools=[read_log],
    budget_runs=14,                    # hard cap on TOTAL harness runs
    seed=7,                            # fully deterministic for a fixed seed
)

print(report.table())                  # every candidate: score, success, tokens
print(report.explain())                # which signal drove each accepted change
report.best_config                     # plain JSON -> harness_from_config
```

Key arguments:

| Argument | Meaning |
| --- | --- |
| `model_factory` | called once **per run**; must return a fresh model when the model is stateful (`ScriptedModel`!) |
| `tasks` | `str` tasks or `(task, success_fn)` — default success is `state.done` *and* not vetoed by a `Verifier` (`_verified is False` fails) |
| `objective` | an `Objective(...)` or any `EvalSummary -> float` (higher is better). Default: success dominates, tokens tie-break |
| `budget_runs` | total harness runs allowed; each candidate costs `len(tasks) × trials` |
| `init_samples` | defaults + N−1 random samples before hinted search begins |
| `trials` | runs per task per candidate (raise for noisy real models) |
| `harness_cls` | `AsyncHarness` builds the async twin (evaluation itself stays sync) |

## The objective

```python
from pyhar import Objective

Objective(
    success_weight=1.0,          # success_rate ∈ [0, 1]
    token_penalty_per_1k=0.001,  # mean total tokens, per thousand
    cost_penalty=0.0,
    turn_penalty=0.0,
)
```

With the defaults, one percentage point of success is worth ~10k tokens — set
the weights to *your* exchange rate, or pass a callable for anything fancier.
Runs that abort on a hard `BudgetExceeded` count as failures **and are charged
the tokens they actually consumed** (the exception carries the run state), so
"abort early" can never look cheaper than an honest completion. A raising
`success_fn` also counts as a failure (recorded as `_success_fn_error` in the
run's memory) rather than crashing the search.

## Reading the result

From `examples/tune_harness.py` (offline, deterministic):

```text
#   score     ok     tokens   turns  kept  note
0   0.9984    100%   1601     2.0    yes   defaults
2   0.9998    100%   193      2.0    yes   random sample
3   0.9998    100%   193      2.0    -     compactor.target_tokens 3000 -> 1500 (compactor never fired)

tokens: 1601 -> 193  (88% saved, success 100%, 5 runs spent)
```

The winning config is plain JSON — check it into your repo, load it with
`harness_from_config`, and re-verify it any time with `bench`.

## Practical notes

- **Real models:** raise `trials` (noise), keep `budget_runs` honest about cost,
  and prefer tasks with a real `success_fn` (e.g. [`checks`](components.md)-style
  assertions) — `state.done` alone is a weak signal.
- **Determinism:** fixed seed + fixed space + deterministic model = identical
  report. Real models break determinism; the search still works, the exact
  trajectory just varies.
- **Scope:** knobs cover component args, `budget` fields, `max_turns`,
  `parallel_tools`, `stream`, `system`, and optional-component inclusion —
  anything `harness_from_config` accepts.

See also: [Concepts](concepts.md) · [Components](components.md) ·
[Cookbook](cookbook.md) · [Model backends](models.md)
