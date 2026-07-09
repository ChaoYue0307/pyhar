"""Trace-guided harness-config search — the seed of "autograd from traces".

A harness composition is data (see ``harness_from_config``), and every run
leaves trace signals in ``state.memory`` (``_tool_savings``, ``_compactions``,
``_stop_reason``, ``_verified``, ``_loop_guard``). ``tune`` closes the loop:

1. **Forward pass** — run a candidate config on your tasks.
2. **"Gradient"** — map the trace signals to *directional* config adjustments
   (e.g. "tool-output budget never fired -> tighten it", "runs hit the turn cap
   and failed -> raise ``max_turns``").
3. **Step** — apply a hinted mutation and re-measure; keep it only if the
   ``Objective`` score improves. Greedy, seeded, reproducible.

This is honest local search guided by run evidence — every accepted change is
measured, and ``report.explain()`` names the signal behind it. It is not a
prompt optimizer (that's DSPy's territory) and it makes no global-optimality
claims.

    from pyhar.optimize import Choice, Range, tune

    space = {
        "components": [
            {"name": "tool_output_budget", "args": {"max_tokens": Range(100, 800)}},
            {"name": "compactor", "args": {"target_tokens": Choice(500, 1000, 2000)},
             "optional": True},
            "loop_guard",
        ],
        "max_turns": Range(2, 10),
    }
    report = tune(space, model_factory=make_model, tasks=[task1, task2], budget_runs=30)
    report.best_config          # plain JSON-able dict -> harness_from_config
    print(report.table())
    print(report.explain())
"""
from __future__ import annotations

import copy
import json
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import harness_from_config
from .core.harness import BudgetExceeded, Harness
from .core.model import Model
from .core.state import HarnessState
from .core.tool import Tool

SuccessFn = Callable[[HarnessState], bool]
TaskSpec = Any  # str | tuple[str, SuccessFn]

_SIGNAL_KEYS = (
    "_tool_savings",
    "_compactions",
    "_verified",
    "_loop_guard",
    "_stop_reason",
    "_denied",
    "_stream_fallback",
)

_SCALAR_KEYS = ("system", "max_turns", "parallel_tools", "stream")


# -- search-space markers ----------------------------------------------------


class Choice:
    """A categorical knob: one of the given options."""

    def __init__(self, *options: Any):
        if len(options) < 2:
            raise ValueError("Choice needs at least two options")
        self.options = list(options)
        self._numeric = all(isinstance(o, (int, float)) and not isinstance(o, bool)
                            for o in options)
        self._sorted = sorted(options) if self._numeric else list(options)

    def default(self) -> Any:
        return self.options[0]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)

    def shift(self, current: Any, direction: int, rng: random.Random) -> Any:
        if self._numeric and current in self._sorted:
            i = self._sorted.index(current) + direction
            if 0 <= i < len(self._sorted):
                return self._sorted[i]
            return current  # already at the edge
        others = [o for o in self.options if o != current]
        return rng.choice(others) if others else current


class Range:
    """A numeric knob in ``[lo, hi]``, moved in ``step`` increments."""

    def __init__(self, lo: float, hi: float, *, step: float | None = None):
        if not lo < hi:
            raise ValueError(f"Range needs lo < hi, got {lo} >= {hi}")
        self.lo, self.hi = lo, hi
        self.is_int = isinstance(lo, int) and isinstance(hi, int)
        if step is None:
            step = max(1, (hi - lo) // 8) if self.is_int else (hi - lo) / 8
        if step <= 0:
            raise ValueError("Range step must be positive")
        if self.is_int:
            # a fractional step on an int range would make shift() a permanent
            # no-op (int(round(x + 0.4)) == x) — reject it up front
            if float(step) != int(step):
                raise ValueError(
                    f"Range({lo}, {hi}) is an integer range; step must be a whole "
                    f"number (got {step}). Use float bounds for a float knob."
                )
            step = int(step)
        self.step = step

    def _cast(self, v: float) -> float:
        v = min(max(v, self.lo), self.hi)
        return int(round(v)) if self.is_int else v

    def default(self) -> Any:
        return self._cast((self.lo + self.hi) / 2)

    def sample(self, rng: random.Random) -> Any:
        # float-tolerant grid size so an exactly-divisible hi is sampleable
        n = int((self.hi - self.lo) / self.step + 1e-9)
        return self._cast(self.lo + rng.randint(0, max(n, 1)) * self.step)

    def shift(self, current: Any, direction: int, rng: random.Random) -> Any:
        return self._cast(current + direction * self.step)


def _is_knob(v: Any) -> bool:
    return isinstance(v, (Choice, Range))


def _resolve_value(v: Any, rng: random.Random | None) -> Any:
    if isinstance(v, (Choice, Range)):
        return v.sample(rng) if rng is not None else v.default()
    return v


# -- space -> config resolution ------------------------------------------------


def _validate_space(space: dict[str, Any]) -> None:
    known = {"components", "budget"} | set(_SCALAR_KEYS)
    unknown = set(space) - known
    if unknown:
        raise ValueError(
            f"unknown space keys: {sorted(unknown)} (known: {sorted(known)}) — "
            f"a typo here would otherwise be silently ignored"
        )


def _key(config: dict[str, Any]) -> str:
    """Dedupe key tolerant of non-JSON values (e.g. a Verifier's callable
    ``check`` arg) — callables are identity-stable within one tune call."""
    return json.dumps(config, sort_keys=True, default=repr)


def _component_specs(space: dict[str, Any]) -> list[dict[str, Any] | str]:
    return list(space.get("components", []))


def _spec_name(spec: Any) -> str:
    return spec if isinstance(spec, str) else spec["name"]


def resolve(space: dict[str, Any], rng: random.Random | None = None,
            include: set[str] | None = None) -> dict[str, Any]:
    """Resolve a search space into a concrete, JSON-able config.

    ``rng=None`` resolves defaults (Choice -> first option, Range -> midpoint,
    optional components included). With an ``rng``, knobs are sampled and each
    optional component is included with probability 0.5 — unless ``include``
    pins the exact component-name set.
    """
    _validate_space(space)
    config: dict[str, Any] = {}
    comps: list[Any] = []
    for spec in _component_specs(space):
        if isinstance(spec, str):
            comps.append(spec)
            continue
        name = spec["name"]
        included = True
        if spec.get("optional"):
            if include is not None:
                included = name in include
            elif rng is not None:
                included = rng.random() < 0.5
        if not included:
            continue
        out: dict[str, Any] = {"name": name}
        if spec.get("args"):
            out["args"] = {k: _resolve_value(v, rng) for k, v in spec["args"].items()}
        comps.append(out)
    if "components" in space:
        config["components"] = comps
    if "budget" in space:
        config["budget"] = {k: _resolve_value(v, rng) for k, v in space["budget"].items()}
    for key in _SCALAR_KEYS:
        if key in space:
            config[key] = _resolve_value(space[key], rng)
    return config


# -- knob addressing over a resolved config -------------------------------------


@dataclass(frozen=True)
class _Knob:
    kind: str            # "value" | "component"
    path: tuple[str, ...]
    spec: Any            # Choice/Range for values; the component spec dict for toggles


def _knobs(space: dict[str, Any]) -> list[_Knob]:
    knobs: list[_Knob] = []
    seen_names: set[str] = set()
    for spec in _component_specs(space):
        name = _spec_name(spec)
        if name in seen_names:
            raise ValueError(f"duplicate component {name!r} in space — tune addresses "
                             f"components by name, so each may appear once")
        seen_names.add(name)
        if isinstance(spec, str):
            continue
        if spec.get("optional"):
            knobs.append(_Knob("component", ("components", name), spec))
        for arg, v in (spec.get("args") or {}).items():
            if _is_knob(v):
                knobs.append(_Knob("value", ("components", name, "args", arg), v))
    for k, v in (space.get("budget") or {}).items():
        if _is_knob(v):
            knobs.append(_Knob("value", ("budget", k), v))
    for key in _SCALAR_KEYS:
        if _is_knob(space.get(key)):
            knobs.append(_Knob("value", (key,), space[key]))
    return knobs


def _find_component(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    for spec in config.get("components", []):
        if isinstance(spec, dict) and spec.get("name") == name:
            return spec
    return None


def _has_component(config: dict[str, Any], name: str) -> bool:
    return any(_spec_name(s) == name for s in config.get("components", []))


def _get(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    if path[0] == "components":
        comp = _find_component(config, path[1])
        if comp is None:
            return None
        return (comp.get("args") or {}).get(path[3])
    if path[0] == "budget":
        return (config.get("budget") or {}).get(path[1])
    return config.get(path[0])


def _set(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if path[0] == "components":
        comp = _find_component(config, path[1])
        if comp is not None:
            comp.setdefault("args", {})[path[3]] = value
    elif path[0] == "budget":
        config.setdefault("budget", {})[path[1]] = value
    else:
        config[path[0]] = value


def _toggle_component(config: dict[str, Any], space: dict[str, Any], name: str) -> bool:
    """Add or remove an optional component, preserving the space's ordering.
    Returns True when the component is present AFTER the toggle."""
    if _has_component(config, name):
        config["components"] = [s for s in config.get("components", [])
                                if _spec_name(s) != name]
        return False
    current = {_spec_name(s): s for s in config.get("components", [])}
    rebuilt: list[Any] = []
    for spec in _component_specs(space):
        sname = _spec_name(spec)
        if sname in current:
            rebuilt.append(current[sname])
        elif sname == name:  # (re)insert at its space position, with default args
            if isinstance(spec, str):
                rebuilt.append(spec)
            else:
                out: dict[str, Any] = {"name": sname}
                if spec.get("args"):
                    out["args"] = {k: _resolve_value(v, None) for k, v in spec["args"].items()}
                rebuilt.append(out)
    config["components"] = rebuilt
    return True


# -- objective & evaluation -----------------------------------------------------


@dataclass
class Objective:
    """Scalarize a run summary. Success dominates by default; tokens/cost/turns
    act as tie-breakers. Or pass ``tune(..., objective=my_callable)`` taking an
    ``EvalSummary`` and returning a float (higher is better)."""

    success_weight: float = 1.0
    token_penalty_per_1k: float = 0.001
    cost_penalty: float = 0.0
    turn_penalty: float = 0.0

    def __call__(self, s: EvalSummary) -> float:
        return (
            self.success_weight * s.success_rate
            - self.token_penalty_per_1k * (s.mean_total_tokens / 1000.0)
            - self.cost_penalty * s.mean_cost
            - self.turn_penalty * s.mean_turns
        )


@dataclass
class EvalSummary:
    success_rate: float
    mean_total_tokens: float
    mean_input_tokens: float
    mean_cost: float
    mean_turns: float
    runs: int
    memories: list[dict[str, Any]] = field(default_factory=list)


def _normalize_tasks(tasks: Sequence[TaskSpec]) -> list[tuple[str, SuccessFn | None]]:
    out: list[tuple[str, SuccessFn | None]] = []
    for t in tasks:
        if isinstance(t, str):
            out.append((t, None))
        elif isinstance(t, tuple) and len(t) == 2:
            out.append((t[0], t[1]))
        else:
            raise ValueError(f"bad task spec {t!r} — expected 'task' or (task, success_fn)")
    if not out:
        raise ValueError("tune needs at least one task")
    return out


def _evaluate(
    config: dict[str, Any],
    *,
    model_factory: Callable[[], Model],
    tasks: list[tuple[str, SuccessFn | None]],
    tools: Iterable[Tool],
    trials: int,
    harness_cls: type[Harness],
) -> EvalSummary:
    oks: list[bool] = []
    totals: list[int] = []
    inputs: list[int] = []
    costs: list[float] = []
    turns: list[int] = []
    memories: list[dict[str, Any]] = []
    tools = list(tools)
    for task, success_fn in tasks:
        for _ in range(trials):
            harness = harness_from_config(
                config, model=model_factory(), tools=tools, harness_cls=harness_cls
            )
            try:
                state = harness.run(task)
            except BudgetExceeded as e:
                # charge the aborted run its REAL consumption — recording zeros
                # would make "abort early" outscore honest completions
                st = e.state
                oks.append(False)
                totals.append(st.usage.total_tokens if st else 0)
                inputs.append(st.usage.input_tokens if st else 0)
                costs.append(st.usage.cost if st else 0.0)
                turns.append(st.turn if st else 0)
                mem = (
                    {k: st.memory[k] for k in _SIGNAL_KEYS if k in st.memory} if st else {}
                )
                mem["_budget_exceeded"] = True
                memories.append(mem)
                continue
            if success_fn is not None:
                try:
                    ok = bool(success_fn(state))
                except Exception as check_error:
                    # a raising check is a failed check, not a crashed search
                    ok = False
                    state.memory["_success_fn_error"] = repr(check_error)
            else:
                # default: done, AND not vetoed by a Verifier that gave up
                ok = bool(state.done) and state.memory.get("_verified", True) is not False
            oks.append(ok)
            totals.append(state.usage.total_tokens)
            inputs.append(state.usage.input_tokens)
            costs.append(state.usage.cost)
            turns.append(state.turn)
            mem = {k: state.memory[k] for k in _SIGNAL_KEYS if k in state.memory}
            if "_success_fn_error" in state.memory:
                mem["_success_fn_error"] = state.memory["_success_fn_error"]
            memories.append(mem)
    return EvalSummary(
        success_rate=sum(oks) / len(oks),
        mean_total_tokens=statistics.fmean(totals),
        mean_input_tokens=statistics.fmean(inputs),
        mean_cost=statistics.fmean(costs),
        mean_turns=statistics.fmean(turns),
        runs=len(oks),
        memories=memories,
    )


# -- the "gradient": trace signals -> directional hints ---------------------------


@dataclass(frozen=True)
class Hint:
    path: tuple[str, ...]
    direction: int      # +1 loosen/raise, -1 tighten/lower (or drop, for components)
    reason: str


def hints_from(config: dict[str, Any], ev: EvalSummary) -> list[Hint]:
    """Map run evidence to directional config adjustments. Pure and inspectable —
    this is the whole 'gradient'."""
    hints: list[Hint] = []
    mems = ev.memories
    failing = ev.success_rate < 1.0

    if _has_component(config, "tool_output_budget"):
        savings = sum(m.get("_tool_savings", 0) for m in mems)
        if savings == 0:
            hints.append(Hint(("components", "tool_output_budget", "args", "max_tokens"), -1,
                              "tool-output budget never fired (no savings) — tighten it"))
        elif failing:
            hints.append(Hint(("components", "tool_output_budget", "args", "max_tokens"), +1,
                              "failures alongside truncated tool output — loosen it"))

    if _has_component(config, "compactor"):
        fired = any(m.get("_compactions") for m in mems)
        if not fired:
            hints.append(Hint(("components", "compactor", "args", "target_tokens"), -1,
                              "compactor never fired — lower the target"))
            hints.append(Hint(("components", "compactor"), -1,
                              "compactor never fired — try removing it"))
        elif failing:
            hints.append(Hint(("components", "compactor", "args", "target_tokens"), +1,
                              "failures alongside compaction — possible over-compaction"))

    if failing:
        if any(m.get("_stop_reason") == "max_turns" for m in mems):
            hints.append(Hint(("max_turns",), +1, "runs hit the turn cap before finishing"))
            hints.append(Hint(("budget", "max_turns"), +1,
                              "runs hit the turn cap before finishing"))
        if _has_component(config, "verifier") and any(m.get("_verified") is False for m in mems):
            hints.append(Hint(("components", "verifier", "args", "max_retries"), +1,
                              "verification still failing after retries"))
        if any(m.get("_budget_exceeded") for m in mems):
            hints.append(Hint(("budget", "max_total_tokens"), +1,
                              "runs aborted on the hard token budget"))
    return hints


# -- report ----------------------------------------------------------------------


@dataclass
class TuneStep:
    index: int
    config: dict[str, Any]
    score: float
    summary: EvalSummary
    note: str
    accepted: bool


@dataclass
class TuneReport:
    best_config: dict[str, Any]
    best_score: float
    best_summary: EvalSummary
    steps: list[TuneStep] = field(default_factory=list)
    runs_used: int = 0

    def table(self) -> str:
        header = f"{'#':<4}{'score':<10}{'ok':<7}{'tokens':<9}{'turns':<7}{'kept':<6}note"
        lines = [header, "-" * (len(header) + 24)]
        for s in self.steps:
            lines.append(
                f"{s.index:<4}{s.score:<10.4f}{s.summary.success_rate:<7.0%}"
                f"{s.summary.mean_total_tokens:<9.0f}{s.summary.mean_turns:<7.1f}"
                f"{'yes' if s.accepted else '-':<6}{s.note}"
            )
        return "\n".join(lines)

    def explain(self) -> str:
        """The accepted-step chain — what changed, and which signal drove it."""
        lines = []
        for s in self.steps:
            if s.accepted and s.index > 0:
                lines.append(f"step {s.index}: {s.note} -> score {s.score:.4f}")
        if not lines:
            return "no mutation beat the initial candidates — best is an initial sample"
        return "\n".join(lines)


# -- the optimizer -----------------------------------------------------------------


def tune(
    space: dict[str, Any],
    *,
    model_factory: Callable[[], Model],
    tasks: Sequence[TaskSpec],
    tools: Iterable[Tool] = (),
    objective: Callable[[EvalSummary], float] | None = None,
    budget_runs: int = 25,
    init_samples: int = 3,
    trials: int = 1,
    seed: int = 0,
    harness_cls: type[Harness] = Harness,
) -> TuneReport:
    """Greedy, trace-hinted local search over a harness-config space.

    ``budget_runs`` caps the TOTAL number of harness runs (each candidate costs
    ``len(tasks) * trials`` runs). ``model_factory`` must return a fresh model
    per run when the model is stateful (``ScriptedModel``!). Deterministic for a
    fixed seed, space, tasks, and model behavior.
    """
    _validate_space(space)
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if init_samples < 1:
        raise ValueError(f"init_samples must be >= 1, got {init_samples}")
    rng = random.Random(seed)
    score_of = objective if objective is not None else Objective()
    task_list = _normalize_tasks(tasks)
    knobs = _knobs(space)
    knob_paths = {k.path for k in knobs}
    runs_per_eval = len(task_list) * trials
    if budget_runs < runs_per_eval:
        raise ValueError(
            f"budget_runs={budget_runs} cannot cover one evaluation "
            f"({len(task_list)} tasks x {trials} trials = {runs_per_eval} runs)"
        )

    tools = list(tools)
    report_steps: list[TuneStep] = []
    seen: set[str] = set()
    runs_used = 0
    index = 0
    best: TuneStep | None = None

    def evaluate(config: dict[str, Any], note: str) -> TuneStep:
        nonlocal runs_used, index, best
        summary = _evaluate(config, model_factory=model_factory, tasks=task_list,
                            tools=tools, trials=trials, harness_cls=harness_cls)
        runs_used += summary.runs
        score = float(score_of(summary))
        accepted = best is None or score > best.score
        step = TuneStep(index=index, config=copy.deepcopy(config), score=score,
                        summary=summary, note=note, accepted=accepted)
        report_steps.append(step)
        if accepted:
            best = step
        index += 1
        return step

    # -- initial candidates: defaults first, then random samples
    default_cfg = resolve(space)
    seen.add(_key(default_cfg))
    evaluate(default_cfg, "defaults")
    attempts = 0
    while (index < init_samples and runs_used + runs_per_eval <= budget_runs
           and attempts < init_samples * 10):
        attempts += 1
        candidate = resolve(space, rng)
        key = _key(candidate)
        if key in seen:
            continue
        seen.add(key)
        evaluate(candidate, "random sample")

    # -- greedy hinted local search from the best candidate
    while runs_used + runs_per_eval <= budget_runs:
        assert best is not None
        mutation = _propose(best.config, space, knobs, knob_paths,
                            hints_from(best.config, best.summary), rng, seen)
        if mutation is None:
            break  # neighborhood exhausted
        candidate, note = mutation
        seen.add(_key(candidate))
        evaluate(candidate, note)

    assert best is not None
    return TuneReport(
        best_config=copy.deepcopy(best.config),
        best_score=best.score,
        best_summary=best.summary,
        steps=report_steps,
        runs_used=runs_used,
    )


def _propose(
    config: dict[str, Any],
    space: dict[str, Any],
    knobs: list[_Knob],
    knob_paths: set[tuple[str, ...]],
    hints: list[Hint],
    rng: random.Random,
    seen: set[str],
) -> tuple[dict[str, Any], str] | None:
    """One unseen neighbor of ``config`` — hinted when evidence exists.

    A hint that turns out to be a no-op (knob at its edge, component absent,
    neighbor already seen) is pruned so it can't starve the attempt budget.
    """
    actionable = [h for h in hints if h.path in knob_paths]
    knob_by_path = {k.path: k for k in knobs}
    for _ in range(40):
        hint: Hint | None = None
        if actionable and rng.random() < 0.8:
            hint = rng.choice(actionable)
            knob = knob_by_path[hint.path]
            direction = hint.direction
            reason = hint.reason
        elif knobs:
            knob = rng.choice(knobs)
            direction = rng.choice([-1, 1])
            reason = "exploration"
        else:
            return None

        def _prune(h: Hint | None = hint) -> None:
            if h is not None and h in actionable:
                actionable.remove(h)

        candidate = copy.deepcopy(config)
        if knob.kind == "component":
            name = knob.path[1]
            if direction < 0 and not _has_component(candidate, name):
                _prune()  # hint says drop, but it's already absent
                continue
            present = _toggle_component(candidate, space, name)
            note = f"{'added' if present else 'removed'} component {name!r} ({reason})"
        else:
            # a missing OWNING COMPONENT skips the knob; a None VALUE is real
            # and still shiftable for categorical knobs (e.g. Choice(None, ...))
            if knob.path[0] == "components" and _find_component(config, knob.path[1]) is None:
                _prune()
                continue
            current = _get(config, knob.path)
            if current is None and isinstance(knob.spec, Range):
                _prune()  # can't shift a non-numeric current numerically
                continue
            new = knob.spec.shift(current, direction, rng)
            if new == current:
                _prune()  # at the edge of the range — this hint is spent
                continue
            _set(candidate, knob.path, new)
            note = f"{'.'.join(knob.path)} {current} -> {new} ({reason})"

        if _key(candidate) not in seen:
            return candidate, note
        _prune()  # this hint's neighbor was already tried
    return None
