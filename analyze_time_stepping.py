"""Heuristic, data-driven tuning of the Coso adaptive time stepper.

Reads one or more run folders (each holding run_config.json, solver_statistics.json,
times.json and the numbered ``<file_name>_NNNNNN.pvd`` files) and

1. builds a per-attempt table: every nonlinear solve attempt with its start time, dt,
   schedule interval, outcome, the iteration count the time-step controller saw
   (``outer``), and the Newton iterations the attempt cost;
2. fits a per-run cost model ``wall = a * newton + b * accepted_steps`` from the pvd
   mtimes, giving an objective in "Newton-equivalents";
3. prints rule-based diagnostics R1-R5, each with a suggested parameter value;
4. ranks a small grid of candidate parameter sets by replaying porepy's actual
   ``TimeScheduler`` (built by :func:`time_stepping.time_steppers_coso`, so the replay
   follows any future change to the algorithm) against a surrogate of the physics
   fitted to each run: an attempt succeeds iff its dt is at most a critical step
   size ``dt_crit(interval, tau)``, with ``tau`` the time since the interval start.

solver_statistics.json does not store the count the controller sees. Each entry's
``num_iterations`` is the Newton count of the *last subsolver call* of the attempt;
using it instead of the real count makes every rule and replay wrong. For the
sequential scheme the controller reads ``SequentialNonlinearSolver.
number_of_iterations()``, the number of outer iterations. Each outer iteration calls
each of the SEQUENTIAL_SUBSOLVERS subsolvers once (mechanics, then pressure and
temperature; see nl_params.py), and every call advances the statistics index. So the
count is the attempt's number of subsolver calls divided by SEQUENTIAL_SUBSOLVERS. For
the monolithic scheme it is the Newton count itself.

The only structural assumption of the surrogate is that, at a given time, a smaller
step is easier than a larger one. dt_crit is *not* assumed monotone in tau: in the
production-start interval of the reference run it drops tenfold some 10^4 s in and
recovers later. Every failure brackets dt_crit between the failed dt and the retry
that succeeded from the same start time; their geometric mean is a knot, and knots are
interpolated in log-log. dt_crit is never taken below the dt of the observed accepted
step covering tau, so the surrogate classifies every observed attempt correctly and
replaying a run's own settings reproduces that run (the "self-check"). After the last
failure of an interval, and in intervals without failures, dt_crit is unbounded
(capped only by dt_max).

Replay outcomes that no nearby observation backs (a success at a dt larger than any
accepted near that tau, a failure without a nearby failed attempt at or below that
dt, or an iteration count predicted far from any observed step) are counted as
"extrapolated". Candidates that rely on untried territory, such as a larger dt_start
than any run has used, are therefore flagged rather than silently favoured. The
surrogate is fitted per run because dt_crit depends on the mesh. Runs that share a
configuration (same schedule, mesh sizes and fault set) are additionally
cross-checked: each run's settings are replayed on the other run's data and compared
with what actually happened. As new test runs arrive, this is the real validation of
the predictions.

Usage:
    python analyze_time_stepping.py RUN_DIR [RUN_DIR ...] [--top N] [--csv OUT]
        [--plot] [--no-replay]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time as _time
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
from porepy.time_stepper.scheduler import CannotRecomputeTimeStep

from time_stepping import time_steppers_coso

REPO_ROOT = Path(__file__).resolve().parent

# Time-stepper settings run_example_3.py used before run_config.json recorded them
# (up to and including commit ffec246). Runs whose config lacks "time_stepper_params"
# are assumed to have used these, and the report marks them as "assumed".
LEGACY_TIME_STEPPER_PARAMS: dict = {
    "iter_optimal_range": (10, 15),
    "iter_relax_factors": (0.6, 2.5),
    "recomp_factor": 0.2,
    "max_attempts": 10,
    "dt_start_transition": 10.0,
    "dt_max_divisor": 2.0,
    "dt_min": 1e-2,
}
TIME_STEPPER_KEYS = tuple(LEGACY_TIME_STEPPER_PARAMS)
# Subsolvers of nl_params.set_nonlinear_solver(scheme="sequential"): mechanics and
# pressure-temperature. Used to turn subsolver calls into outer iterations.
SEQUENTIAL_SUBSOLVERS = 2

# R1: growing after an accepted step with `outer` iterations is "safe" if the next,
# grown attempt failed in at most P_SAFE of the observed cases; holding is preferred
# over shrinking while that probability stays below P_HOLD. Counts seen fewer than
# MIN_COUNT times are ignored.
P_SAFE = 0.15
P_HOLD = 0.5
MIN_COUNT = 3
# R3: a larger retry factor is only suggested if at most this share of failure
# sequences needed more than one retry.
MAX_MULTI_RETRY_SHARE = 0.2
# Candidate growth factors and retry factors always tried by the replay, in addition
# to the current values and the rule-based suggestions.
GROWTH_FACTORS = (1.5, 2.0, 2.5)
RETRY_FACTORS = (0.2, 0.5)
DT_MAX_DIVISORS = (2.0, 1.0)
EXTRA_BANDS: tuple[tuple[int, int], ...] = ()
# Relative error above which a replay is flagged as not reproducing a run. Runs that
# fail their self-check are left out of the cross-run ranking.
SELF_CHECK_TOL = 0.25
# A replay is abandoned (and its candidate ranked last) once it needs more than
# max(STEP_CAP_MIN, STEP_CAP_FACTOR * attempts of the observed run) attempts, which
# happens when a hold band freezes dt far below what the interval needs.
STEP_CAP_FACTOR = 20
STEP_CAP_MIN = 2000
# Observations within this distance in log(tau + TAU_OFFSET) count as "near" when
# deciding whether a replayed outcome is backed by data, and iteration counts are
# only taken from observed steps within OUTER_RADIUS in (log tau, log dt).
TAU_WINDOW = 0.1
OUTER_RADIUS = 0.7
# Offset (s) so that log(tau + TAU_OFFSET) is finite at the start of an interval.
TAU_OFFSET = 1.0
T_TOL = 1e-6


def _same_time(t1: float, t2: float) -> bool:
    return abs(t1 - t2) <= 1e-9 * max(1.0, abs(t2))


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class Attempt:
    """One nonlinear solve attempt, i.e. one entry of solver_statistics.json."""

    t: float
    """Simulation time at the start of the attempted step."""
    dt: float
    interval: int
    """Index i of the schedule interval [schedule[i], schedule[i+1]) containing t."""
    is_transition: bool
    """Whether that interval starts at a real BC change (see create_schedule)."""
    tau: float
    """Time since the start of the interval."""
    ok: bool
    outer: int
    """Iteration count the time-step controller sees (see the module docstring)."""
    newton: int
    """Newton iterations spent on the attempt, summed over all subsolver calls."""


@dataclass
class Run:
    path: Path
    label: str
    signature: str
    """Schedule, mesh sizes and fault set; runs sharing it are cross-checked."""
    schedule: np.ndarray
    is_transition: np.ndarray
    dt: float
    """Uniform dt passed to time_steppers_coso (only used without is_transition)."""
    params: dict
    params_assumed: bool
    solver_scheme: str
    attempts: list[Attempt]
    wall: np.ndarray
    """Wall-clock seconds per accepted step, from the numbered pvd mtimes."""
    newton_per_step: np.ndarray
    """Newton iterations per accepted step, including failed attempts before it."""
    total_newton_reported: int
    t_reached: float

    @property
    def accepted(self) -> list[Attempt]:
        return [a for a in self.attempts if a.ok]

    @property
    def failed(self) -> list[Attempt]:
        return [a for a in self.attempts if not a.ok]

    @property
    def finished(self) -> bool:
        return self.t_reached >= self.schedule[-1] - T_TOL

    def status(self) -> str:
        if self.finished:
            return "finished"
        trailing = 0
        for a in reversed(self.attempts):
            if a.ok:
                break
            trailing += 1
        msg = f"NOT finished: reached t={self.t_reached:.6g} s of {self.schedule[-1]:.6g}"
        if trailing:
            msg += f", ended with {trailing} consecutive failed attempt(s)"
        return msg


def _read_json(path: Path, retries: int = 5, wait: float = 2.0) -> dict:
    """Read a JSON file, retrying if a running simulation is caught mid-write."""
    for i in range(retries):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            if i == retries - 1:
                raise
            _time.sleep(wait)
    raise AssertionError("unreachable")


def interval_index(schedule: np.ndarray, t: float) -> int:
    """Index of the schedule interval [schedule[i], schedule[i+1]) containing t."""
    i = int(np.searchsorted(schedule, t + T_TOL, side="right")) - 1
    return min(max(i, 0), len(schedule) - 2)


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_run(path: Path | str) -> Run:
    """Build the per-attempt table of one run folder.

    solver_statistics.json holds one entry per solve attempt, keyed by the statistics
    index at the end of the attempt. That index advances once per subsolver call, and
    ``global.num_iterations_history[k]`` holds the Newton iterations of call k. So an
    attempt's subsolver calls are the history slice between its key and the previous
    one. That slice gives both the Newton cost and, for the sequential scheme, the
    controller's iteration count (see the module docstring).
    (analyze_solver_statistics.load_attempts is not reused because it drops the keys.)
    Entries still in progress when the file was written are skipped.
    """
    path = Path(path)
    cfg = _read_json(path / "run_config.json")
    stats = _read_json(path / "solver_statistics.json")
    times = _read_json(path / "times.json")["time"]

    schedule = np.asarray(cfg["schedule"], dtype=float)
    is_transition = np.asarray(
        cfg.get("is_transition", [True] * len(schedule)), dtype=bool
    )
    params = dict(LEGACY_TIME_STEPPER_PARAMS)
    recorded = cfg.get("time_stepper_params")
    if recorded:
        params.update(recorded)
    params["iter_optimal_range"] = tuple(int(x) for x in params["iter_optimal_range"])
    params["iter_relax_factors"] = tuple(float(x) for x in params["iter_relax_factors"])
    scheme = cfg.get("solver_scheme") or (
        "monolithic" if "Case_III_monolithic" in str(path.resolve()) else "sequential"
    )
    model_params = cfg.get("model_params", {})
    signature = json.dumps(
        [
            cfg["schedule"],
            model_params.get("meshing_arguments"),
            model_params.get("exclude_faults"),
        ],
        sort_keys=True,
    )

    history = stats["global"]["num_iterations_history"]
    keys = sorted((k for k in stats if k != "global"), key=int)
    attempts: list[Attempt] = []
    newton_per_step: list[int] = []
    prev, pending = -1, 0
    for key in keys:
        entry, idx = stats[key], int(key)
        calls = history[prev + 1 : idx + 1]
        prev = idx
        status = entry.get("simulation_status")
        if status not in ("successful", "failed"):
            continue
        if scheme == "sequential":
            outer = math.ceil(len(calls) / SEQUENTIAL_SUBSOLVERS)
        else:
            outer = int(entry["num_iterations"])
        i = interval_index(schedule, entry["time"])
        attempt = Attempt(
            t=float(entry["time"]),
            dt=float(entry["dt"]),
            interval=i,
            is_transition=bool(is_transition[i]),
            tau=float(entry["time"] - schedule[i]),
            ok=status == "successful",
            outer=outer,
            newton=int(sum(calls)),
        )
        attempts.append(attempt)
        pending += attempt.newton
        if attempt.ok:
            newton_per_step.append(pending)
            pending = 0

    file_name = model_params.get("file_name", "example_3")
    mtimes = []
    for j in range(len(times)):
        pvd = path / f"{file_name}_{j:06d}.pvd"
        if not pvd.exists():
            break
        mtimes.append(pvd.stat().st_mtime)

    return Run(
        path=path,
        label=_label(path),
        signature=signature,
        schedule=schedule,
        is_transition=is_transition,
        dt=float(cfg.get("dt", 1e3)),
        params=params,
        params_assumed=not recorded,
        solver_scheme=scheme,
        attempts=attempts,
        wall=np.diff(mtimes),
        newton_per_step=np.asarray(newton_per_step, dtype=float),
        total_newton_reported=int(stats["global"].get("total_num_iterations", -1)),
        t_reached=float(times[-1]) if times else float(schedule[0]),
    )


# --------------------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------------------


@dataclass
class CostModel:
    a: float
    """Wall seconds per Newton iteration."""
    b: float
    """Wall seconds of overhead per accepted step (export, assembly, ...)."""
    r2: float
    n: int

    @property
    def overhead(self) -> float:
        """Per-step overhead in Newton-equivalents."""
        return self.b / self.a


def fit_cost_model(run: Run) -> CostModel | None:
    """Least-squares fit of wall time per accepted step against its Newton iterations."""
    n = min(len(run.wall), len(run.newton_per_step))
    if n < 5:
        return None
    x, y = run.newton_per_step[:n], run.wall[:n]
    A = np.column_stack([x, np.ones(n)])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    if a <= 0:
        return None
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - A @ [a, b]) ** 2).sum()) / total if total > 0 else math.nan
    return CostModel(a=float(a), b=max(float(b), 0.0), r2=r2, n=n)


# --------------------------------------------------------------------------------------
# Rule-based diagnostics
# --------------------------------------------------------------------------------------


def _is_next_step(a: Attempt, b: Attempt) -> bool:
    """Whether attempt b starts where the accepted attempt a ended, in the same interval."""
    return b.interval == a.interval and _same_time(b.t, a.t + a.dt)


def precursor_table(runs: list[Run]) -> dict[int, tuple[int, int]]:
    """outer count k of an accepted step -> (number of grown next attempts, failures)."""
    table: dict[int, tuple[int, int]] = {}
    for run in runs:
        for a, b in zip(run.attempts, run.attempts[1:]):
            if a.ok and _is_next_step(a, b) and b.dt > 1.01 * a.dt:
                n, f = table.get(a.outer, (0, 0))
                table[a.outer] = (n + 1, f + (not b.ok))
    return table


def suggest_band(table: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
    """R1: grow only after counts where growth was safe; hold while it is tolerable.

    TargetNonlinearIterations grows dt if outer < iter_min, shrinks it if
    outer > iter_max and holds it otherwise. With k_safe the largest count such that
    growth after it (and after every smaller count) failed in at most P_SAFE of the
    cases, iter_min = k_safe + 1: grow below it, hold at it. iter_max then extends
    the hold over the following counts whose failure probability stays below P_HOLD;
    counts beyond iter_max shrink dt.
    """
    ks = sorted(k for k, (n, _) in table.items() if n >= MIN_COUNT)
    if not ks:
        return None
    p = {k: table[k][1] / table[k][0] for k in ks}
    k_safe = None
    for k in ks:
        if p[k] > P_SAFE:
            break
        k_safe = k
    iter_min = ks[0] if k_safe is None else k_safe + 1
    iter_max = iter_min
    while p.get(iter_max + 1, 1.0) < P_HOLD:
        iter_max += 1
    return iter_min, iter_max


def refailures(run: Run) -> int:
    """Failed attempts at a dt >= one that already failed in the same interval."""
    seen: dict[int, list[float]] = defaultdict(list)
    count = 0
    for a in run.failed:
        if any(a.dt >= 0.99 * d for d in seen[a.interval]):
            count += 1
        seen[a.interval].append(a.dt)
    return count


def cycle_length(growth: float, retry: float) -> float:
    """Accepted steps after a failure before growth re-reaches the failed dt."""
    if growth <= 1:
        return math.inf
    return math.ceil(math.log(1 / retry) / math.log(growth) - 1e-12)


def retry_evidence(run: Run) -> tuple[list[float], int, int]:
    """R3: per failure sequence (consecutive failures from one start time), the ratio
    of the largest dt shown to work there to the first failed dt.

    Evidence is the accepted step the failed attempt grew from and the retry that
    finally succeeded from the same start time. Returns the ratios, the number of
    sequences, and how many of them needed more than one retry.
    """
    ratios, n_seq, n_multi = [], 0, 0
    atts = run.attempts
    for idx, a in enumerate(atts):
        if a.ok:
            continue
        prev = atts[idx - 1] if idx > 0 else None
        if prev is not None and not prev.ok and _same_time(prev.t, a.t):
            continue  # not the first failure of its sequence
        n_seq += 1
        j = idx + 1
        while j < len(atts) and not atts[j].ok and _same_time(atts[j].t, a.t):
            j += 1
        n_multi += j > idx + 1
        evidence = [prev.dt] if prev is not None and prev.ok and _is_next_step(prev, a) else []
        if j < len(atts) and atts[j].ok and _same_time(atts[j].t, a.t):
            evidence.append(atts[j].dt)
        evidence = [d for d in evidence if d < a.dt]
        if evidence:
            ratios.append(max(evidence) / a.dt)
    return ratios, n_seq, n_multi


@dataclass
class TransitionClimb:
    interval: int
    first_dt: float
    start_failed: bool
    accepted_before_first_failure: int
    first_failure_dt: float | None


def transition_climbs(run: Run) -> list[TransitionClimb]:
    """R4: how each transition interval's step size developed from its first step."""
    out = []
    by_interval: dict[int, list[Attempt]] = defaultdict(list)
    for a in run.attempts:
        if a.is_transition:
            by_interval[a.interval].append(a)
    for i, atts in sorted(by_interval.items()):
        first = atts[0]
        if first.tau > T_TOL:
            continue  # interval entered mid-way (restart); no dt_start information
        first_fail = next((a for a in atts if not a.ok), None)
        n_ok = 0
        for a in atts:
            if not a.ok:
                break
            n_ok += 1
        out.append(
            TransitionClimb(
                interval=i,
                first_dt=first.dt,
                start_failed=not first.ok,
                accepted_before_first_failure=n_ok,
                first_failure_dt=None if first_fail is None else first_fail.dt,
            )
        )
    return out


@dataclass
class Suggestions:
    band: tuple[int, int] | None
    retry: float | None
    dt_start_transition: float | None
    dt_max_divisor: float | None


def rule_report(runs: list[Run], base: dict) -> tuple[Suggestions, list[str]]:
    """Diagnostics R1-R5 pooled over all runs, with a suggested value for each."""
    lines = []
    iter_min, iter_max = base["iter_optimal_range"]
    decrease, growth = base["iter_relax_factors"]
    retry = base["recomp_factor"]
    schemes = sorted({r.solver_scheme for r in runs})
    unit = "outer sequential iterations" if schemes == ["sequential"] else "iterations"

    # R1
    lines.append(f"R1  Iteration band {base['iter_optimal_range']} (unit: {unit})")
    counts: dict[int, int] = defaultdict(int)
    for r in runs:
        for a in r.accepted:
            counts[a.outer] += 1
    max_outer = max(counts, default=0)
    lines.append(f"    accepted steps by count: {dict(sorted(counts.items()))}")
    typical = sorted(counts, key=counts.get, reverse=True)[:2]
    if iter_min > max(typical, default=0):
        lines.append(
            f"    MOSTLY INERT: the typical counts {sorted(typical)} are below "
            f"iter_min={iter_min}, so almost every accepted step grows dt by "
            f"{growth:g}; hold/shrink only fire on rare outliers (max {max_outer})."
        )
    table = precursor_table(runs)
    lines.append(
        "    P(next grown attempt fails | count): "
        + ", ".join(
            f"{k}: {f}/{n} ({f / n:.0%})" for k, (n, f) in sorted(table.items())
        )
    )
    band = suggest_band(table)
    if band:
        lines.append(f"    -> suggested iter_optimal_range = {band}")
        if band[0] <= max(typical, default=0):
            lines.append(
                f"    caution: steps often take {band[0]}+ iterations, so holding "
                f"there can freeze dt regardless of step size; the replay shows "
                "whether it does"
            )
    else:
        lines.append("    -> not enough grow events to suggest a band")

    # R2
    n_star = cycle_length(growth, retry)
    lines.append(f"R2  Grow/fail limit cycle (g={growth:g}, r={retry:g})")
    lines.append(
        f"    after a failure growth re-reaches the failed dt in n*={n_star} accepted "
        f"steps -> about {1 / (n_star + 1):.0%} of attempts fail where dt is "
        "limited by convergence"
    )
    for r in runs:
        fails = r.failed
        if not fails:
            continue
        hot = {a.interval for a in fails}
        n_hot = sum(1 for a in r.attempts if a.interval in hot)
        wasted = sum(a.newton for a in fails) / max(1, sum(a.newton for a in r.attempts))
        lines.append(
            f"    {r.label}: {refailures(r)}/{len(fails)} failures re-tried a dt that "
            f"had already failed; failed share in affected intervals "
            f"{len(fails)}/{n_hot} ({len(fails) / n_hot:.0%}); {wasted:.0%} of Newton "
            "iterations spent in failed attempts"
        )

    # R3
    ratios, n_seq, n_multi = [], 0, 0
    for r in runs:
        rr, ns, nm = retry_evidence(r)
        ratios += rr
        n_seq += ns
        n_multi += nm
    lines.append(f"R3  Retry factor (current r={retry:g})")
    retry_suggestion = None
    if n_seq:
        lines.append(
            f"    {n_seq} failure sequences, {n_multi} ({n_multi / n_seq:.0%}) needed "
            "more than one retry"
        )
    if ratios:
        median = float(np.median(ratios))
        lines.append(
            "    largest dt shown to work / first failed dt: quantiles 10/50/90% = "
            f"{np.quantile(ratios, 0.1):.2f}/{median:.2f}/{np.quantile(ratios, 0.9):.2f}"
        )
        if n_multi / n_seq <= MAX_MULTI_RETRY_SHARE:
            # Retry just below the dt that typically still works, and below 1/g so the
            # next growth step does not re-try the failed dt at once.
            retry_suggestion = round(min(median, 0.9 / growth), 3)
            if retry_suggestion > retry:
                lines.append(f"    -> suggested recomp_factor = {retry_suggestion:g}")
            else:
                retry_suggestion = None
        else:
            lines.append("    -> the current cut is not too deep; keep it")
    else:
        lines.append("    no failures with usable evidence")

    # R4
    lines.append(
        f"R4  First step at transitions (dt_start_transition={base['dt_start_transition']:g}"
        " s; the controller updates it once before use, hence often g*dt_start)"
    )
    climbs: list[TransitionClimb] = []
    for r in runs:
        rc = transition_climbs(r)
        climbs += rc
        if len(runs) > 1 and rc:
            lines.append(f"    {r.label}:")
        for c in rc:
            ff = "-" if c.first_failure_dt is None else f"{c.first_failure_dt:.3g}"
            lines.append(
                f"    interval {c.interval:2d}: first dt={c.first_dt:.3g} "
                f"{'FAILED' if c.start_failed else 'ok'}, "
                f"{c.accepted_before_first_failure} accepted before first failure "
                f"(at dt={ff})"
            )
    dt_start_suggestion = None
    if any(c.start_failed for c in climbs):
        first_ok = [
            a.dt for r in runs for a in r.accepted if a.is_transition and a.tau <= T_TOL
        ]
        if first_ok:
            dt_start_suggestion = float(min(first_ok)) / growth
        lines.append(
            "    a first step failed -> suggest dt_start_transition so that the first "
            f"step equals the smallest first dt that worked: {dt_start_suggestion}"
        )
    elif any(c.first_failure_dt is not None for c in climbs):
        # First step two growth steps below the smallest dt that ever failed in a
        # transition interval, skipping part of the climb. This extrapolates to
        # tau=0, which the replay's extrapolated share makes visible.
        smallest = min(c.first_failure_dt for c in climbs if c.first_failure_dt)
        dt_start_suggestion = float(
            max(base["dt_start_transition"], smallest / growth**3)
        )
        if dt_start_suggestion > base["dt_start_transition"]:
            lines.append(
                "    no first step failed -> suggest dt_start_transition = "
                f"{dt_start_suggestion:.3g} s (first step two growth steps below the "
                f"smallest first-failure dt, {smallest:.3g} s)"
            )
        else:
            dt_start_suggestion = None
            lines.append("    the climb to the first failure is already short; keep it")

    # R5
    divisor = base["dt_max_divisor"]
    bin_fail = [a for r in runs for a in r.failed if not a.is_transition]
    bin_ok = [a for r in runs for a in r.accepted if not a.is_transition]
    lines.append(f"R5  dt_max divisor (current {divisor:g}) on bin-only intervals")
    lines.append(
        f"    {len(bin_ok)} accepted, {len(bin_fail)} failed attempts on bin-only "
        "intervals"
    )
    divisor_suggestion = None
    if bin_ok and not bin_fail and divisor > 1:
        divisor_suggestion = 1.0
        lines.append(
            "    -> no failures: try dt_max_divisor = 1 (one step per bin interval)"
        )

    return (
        Suggestions(band, retry_suggestion, dt_start_suggestion, divisor_suggestion),
        lines,
    )


# --------------------------------------------------------------------------------------
# Surrogate and replay
# --------------------------------------------------------------------------------------


def _log_tau(tau: float) -> float:
    return math.log(tau + TAU_OFFSET)


class Envelope:
    """Critical step size dt_crit(interval, tau) fitted to one run; see module doc."""

    def __init__(self, run: Run) -> None:
        acc: dict[int, list] = defaultdict(list)
        fail: dict[int, list] = defaultdict(list)
        for a in run.attempts:
            (acc if a.ok else fail)[a.interval].append((a.tau, a.dt))
        self.acc = {i: np.asarray(sorted(v)) for i, v in acc.items()}
        self.fail = {i: np.asarray(sorted(v)) for i, v in fail.items()}
        self.knots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, fails in self.fail.items():
            upper: dict[float, float] = {}
            for tau_f, dt_f in fails:
                upper[tau_f] = min(dt_f, upper.get(tau_f, math.inf))
            taus, crit = [], []
            for tau_f, dt_f in sorted(upper.items()):
                lo = self.covering_dt(i, tau_f)
                if not 0 < lo < dt_f:
                    lo = 0.5 * dt_f  # the run ended on this failure
                taus.append(_log_tau(tau_f))
                crit.append(math.log(math.sqrt(lo * dt_f)))
            self.knots[i] = (np.asarray(taus), np.asarray(crit))

    def covering_dt(self, i: int, tau: float) -> float:
        """dt of the observed accepted step whose [tau_a, tau_a + dt_a) contains tau."""
        arr = self.acc.get(i)
        if arr is None:
            return 0.0
        j = int(np.searchsorted(arr[:, 0], tau + T_TOL, side="right")) - 1
        if j < 0 or tau >= arr[j, 0] + arr[j, 1] - T_TOL:
            return 0.0
        return float(arr[j, 1])

    def dt_crit(self, i: int, tau: float) -> float:
        if i not in self.knots:
            return math.inf
        x, y = self.knots[i]
        xq = _log_tau(tau)
        if xq > x[-1] + 1e-12:
            return math.inf
        return max(float(np.exp(np.interp(xq, x, y))), self.covering_dt(i, tau))

    def supported(self, i: int, tau: float, dt: float, ok: bool) -> bool:
        """Whether a predicted outcome is backed by an observation near tau."""
        if ok:
            if dt <= self.covering_dt(i, tau) * (1 + 1e-9):
                return True
            arr, backs = self.acc.get(i), lambda d: d >= dt * (1 - 1e-9)
        else:
            arr, backs = self.fail.get(i), lambda d: d <= dt * (1 + 1e-9)
        if arr is None:
            return False
        near = np.abs(np.log(arr[:, 0] + TAU_OFFSET) - _log_tau(tau)) <= TAU_WINDOW
        return bool(np.any(near & backs(arr[:, 1])))


class OuterPredictor:
    """Controller-visible iteration count of an accepted step.

    On the observed trajectory it returns the observed count, so replaying a run's own
    settings reproduces the run. Elsewhere it takes the majority vote of the k nearest
    accepted steps of the run in the same interval, among those within OUTER_RADIUS in
    (log tau, log dt). If none is that close it returns the run's most common count:
    copying a rare high count to much smaller steps would make the replay shrink dt
    forever.
    """

    def __init__(self, run: Run, k: int = 3) -> None:
        pts: dict[int, list] = defaultdict(list)
        for a in run.accepted:
            pts[a.interval].append((_log_tau(a.tau), math.log(a.dt), a.outer))
        self.pts = {i: np.asarray(v) for i, v in pts.items()}
        counts = np.bincount([a.outer for a in run.accepted] or [1])
        self.default = int(counts.argmax())
        self.k = k

    def _d2(self, pts: np.ndarray, tau: float, dt: float) -> np.ndarray:
        return (pts[:, 0] - _log_tau(tau)) ** 2 + (pts[:, 1] - math.log(dt)) ** 2

    def __call__(self, i: int, tau: float, dt: float) -> int:
        pts = self.pts.get(i)
        if pts is None:
            return self.default
        d2 = self._d2(pts, tau, dt)
        order = np.argsort(d2)
        if d2[order[0]] < 1e-18:
            return int(pts[order[0], 2])
        nearest = order[: self.k]
        close = nearest[d2[nearest] <= OUTER_RADIUS**2]
        if close.size == 0:
            return self.default
        votes = np.bincount(pts[close, 2].astype(int))
        return int(np.flatnonzero(votes == votes.max())[-1])  # ties: the larger count

    def near(self, i: int, tau: float, dt: float) -> bool:
        """Whether an observed accepted step lies within OUTER_RADIUS."""
        pts = self.pts.get(i)
        return pts is not None and float(self._d2(pts, tau, dt).min()) <= OUTER_RADIUS**2


class NewtonCost:
    """Newton iterations of an attempt: per-count means for accepted steps, the mean
    over failed attempts otherwise."""

    def __init__(self, run: Run, fail_ratio_fallback: float) -> None:
        by_outer: dict[int, list[int]] = defaultdict(list)
        for a in run.accepted:
            by_outer[a.outer].append(a.newton)
        self.by_outer = {k: float(np.mean(v)) for k, v in by_outer.items()}
        n_outer = sum(a.outer for a in run.accepted)
        self.per_outer = sum(a.newton for a in run.accepted) / max(1, n_outer)
        mean_ok = float(np.mean([a.newton for a in run.accepted] or [1.0]))
        fails = run.failed
        self.fail = (
            float(np.mean([a.newton for a in fails]))
            if fails
            else fail_ratio_fallback * mean_ok
        )

    def __call__(self, ok: bool, outer: int) -> float:
        if not ok:
            return self.fail
        return self.by_outer.get(outer, self.per_outer * outer)


class _ReplayStatus:
    """Stand-in for a NonlinearSolverStatus: all TargetNonlinearIterations reads."""

    def __init__(self, converged: bool, iterations: int) -> None:
        self._converged = converged
        self._iterations = iterations

    def is_converged(self) -> bool:
        return self._converged

    def number_of_iterations(self) -> int:
        return self._iterations


@dataclass
class ReplayResult:
    attempts: list[tuple[float, float, bool]]
    """(start time, dt, success) of every replayed attempt."""
    accepted: int = 0
    failed: int = 0
    newton: float = 0.0
    extrapolated: int = 0
    aborted: str | None = None

    def objective(self, overhead: float) -> float:
        """Cost in Newton-equivalents: Newton iterations plus per-step overhead."""
        return self.newton + overhead * self.accepted

    @property
    def extrapolated_share(self) -> float:
        return self.extrapolated / max(1, len(self.attempts))


def replay(
    schedule: np.ndarray,
    is_transition: np.ndarray,
    dt: float,
    params: dict,
    dt_crit: Callable[[int, float], float],
    outer: Callable[[int, float, float], int],
    newton: Callable[[bool, int], float],
    horizon: float | None = None,
    supported: Callable[[int, float, float, bool], bool] | None = None,
    max_total_attempts: int | None = None,
) -> ReplayResult:
    """Drive the real time_steppers_coso scheduler through the schedule.

    An attempt succeeds iff its dt <= dt_crit(interval, tau); the scheduler then
    picks the next dt exactly as in a simulation. Stops at ``horizon`` (default: the
    end of the schedule), when the scheduler aborts, or after ``max_total_attempts``.
    """
    schedule = np.asarray(schedule, dtype=float)
    kwargs = {k: params[k] for k in TIME_STEPPER_KEYS if k in params}
    stepper, _ = time_steppers_coso(
        schedule, dt, is_transition=np.asarray(is_transition, dtype=bool), **kwargs
    )
    scheduler = stepper.scheduler
    horizon = schedule[-1] if horizon is None else horizon
    result = ReplayResult(attempts=[])
    while not scheduler.is_finished() and scheduler.get_time() < horizon - T_TOL:
        t = scheduler.get_time()
        i = interval_index(schedule, t)
        tau = t - schedule[i]
        if max_total_attempts is not None and len(result.attempts) >= max_total_attempts:
            result.aborted = (
                f"over {max_total_attempts} attempts: dt stalled at "
                f"{scheduler.get_dt():.3g} s in interval {i}"
            )
            return result
        for _ in range(stepper.max_attempts):
            step = scheduler.get_dt()
            ok = step <= dt_crit(i, tau)
            count = outer(i, tau, step) if ok else 0
            result.attempts.append((t, step, ok))
            result.newton += newton(ok, count)
            if ok:
                result.accepted += 1
            else:
                result.failed += 1
            if supported is not None and not supported(i, tau, step, ok):
                result.extrapolated += 1
            try:
                scheduler.compute_next_time_step(
                    success=ok,
                    context={"nonlinear_solver_status": _ReplayStatus(ok, count)},
                )
            except CannotRecomputeTimeStep as exc:
                result.aborted = str(exc)
                return result
            if ok:
                break
        else:
            result.aborted = f"max_attempts ({stepper.max_attempts}) exhausted"
            return result
    return result


def replay_run(
    run: Run, params: dict, fail_ratio: float, horizon: float | None = None
) -> ReplayResult:
    """Replay ``params`` on the surrogate fitted to ``run``, up to where it got."""
    envelope = Envelope(run)
    outer = OuterPredictor(run)

    def supported(i: int, tau: float, dt: float, ok: bool) -> bool:
        return envelope.supported(i, tau, dt, ok) and (not ok or outer.near(i, tau, dt))

    return replay(
        run.schedule,
        run.is_transition,
        run.dt,
        params,
        dt_crit=envelope.dt_crit,
        outer=outer,
        newton=NewtonCost(run, fail_ratio),
        horizon=run.t_reached if horizon is None else horizon,
        supported=supported,
        max_total_attempts=max(STEP_CAP_MIN, STEP_CAP_FACTOR * len(run.attempts)),
    )


def candidate_grid(base: dict, s: Suggestions) -> list[dict]:
    """Candidates: the current settings varied along each tuned parameter."""
    decrease, growth = base["iter_relax_factors"]

    def unique(values):
        out = []
        for v in values:
            if v is not None and v not in out:
                out.append(v)
        return out

    bands = unique([tuple(base["iter_optimal_range"]), s.band, *EXTRA_BANDS])
    growths = unique([growth, *GROWTH_FACTORS])
    retries = unique([base["recomp_factor"], s.retry, *RETRY_FACTORS])
    dt_starts = unique([base["dt_start_transition"], s.dt_start_transition])
    divisors = unique([base["dt_max_divisor"], s.dt_max_divisor, *DT_MAX_DIVISORS])
    grid = []
    for band, g, r, dt0, div in itertools.product(
        bands, growths, retries, dt_starts, divisors
    ):
        params = dict(base)
        params.update(
            iter_optimal_range=band,
            iter_relax_factors=(decrease, g),
            recomp_factor=r,
            dt_start_transition=dt0,
            dt_max_divisor=div,
        )
        grid.append(params)
    return grid


def params_key(p: dict) -> tuple:
    return (
        tuple(p["iter_optimal_range"]),
        float(p["iter_relax_factors"][1]),
        float(p["recomp_factor"]),
        float(p["dt_start_transition"]),
        float(p["dt_max_divisor"]),
    )


def format_key(key: tuple) -> str:
    band, g, r, dt0, div = key
    return f"band={band} g={g:g} r={r:g} dt0_tr={dt0:.3g} div={div:g}"


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _rel(pred: float, actual: float) -> float:
    return abs(pred - actual) / max(abs(actual), 1.0)


def _compare(res: ReplayResult, actual: list[Attempt]) -> tuple[str, float]:
    checks = {
        "accepted": (res.accepted, sum(a.ok for a in actual)),
        "failed": (res.failed, sum(not a.ok for a in actual)),
        "Newton": (res.newton, sum(a.newton for a in actual)),
    }
    worst = max(_rel(p, a) for p, a in checks.values())
    text = ", ".join(f"{k} {p:.0f}/{a:.0f}" for k, (p, a) in checks.items())
    return text, worst


def run_report(run: Run, cost: CostModel | None) -> list[str]:
    lines = [f"== {run.label}", f"   status: {run.status()}"]
    src = "ASSUMED (not in run_config.json)" if run.params_assumed else "recorded"
    lines.append(
        f"   scheme: {run.solver_scheme}; time-stepper params {src}: "
        f"{format_key(params_key(run.params))}"
    )
    total = sum(a.newton for a in run.attempts)
    lines.append(
        f"   {len(run.attempts)} attempts, {len(run.accepted)} accepted, "
        f"{len(run.failed)} failed; {total} Newton iterations "
        f"(solver_statistics.json: {run.total_newton_reported})"
    )
    if cost is None:
        lines.append("   cost model: too few timed steps to fit")
    else:
        lines.append(
            f"   cost model: wall = {cost.a:.2f} s/Newton-it + {cost.b:.1f} s/step "
            f"(R^2={cost.r2:.3f}, {cost.n} steps; overhead = {cost.overhead:.1f} "
            f"Newton-equivalents/step); wall so far {run.wall.sum() / 3600:.2f} h"
        )
    return lines


def rank(
    runs: list[Run],
    costs: list[CostModel | None],
    grid: list[dict],
    fail_ratio: float,
    top: int,
) -> list[str]:
    """Replay every candidate on every run; rank per run and across runs."""
    lines = []
    valid_costs = [c for c in costs if c is not None]
    fallback = (
        CostModel(
            a=float(np.median([c.a for c in valid_costs])),
            b=float(np.median([c.b for c in valid_costs])),
            r2=math.nan,
            n=0,
        )
        if valid_costs
        else CostModel(a=1.0, b=0.0, r2=math.nan, n=0)
    )
    ratios: dict[tuple, list[float]] = defaultdict(list)
    extrap: dict[tuple, list[float]] = defaultdict(list)
    untrusted = []
    for run, cost in zip(runs, costs):
        cost = cost or fallback
        base = replay_run(run, run.params, fail_ratio)
        base_j = base.objective(cost.overhead)
        actual = [a for a in run.attempts if a.t < run.t_reached - T_TOL]
        lines.append(f"== Replay: {run.label} (up to t={run.t_reached:.6g} s)")
        text, worst = _compare(base, actual)
        trusted = worst <= SELF_CHECK_TOL and not base.aborted
        if not trusted:
            untrusted.append(run.label)
        lines.append(
            f"   self-check, own settings replayed vs actual: {text}"
            + (
                "  ok"
                if trusted
                else f"  WARNING: off by {worst:.0%}; the surrogate does not reproduce "
                "this run (were its time-stepper settings assumed wrongly?) -- left "
                "out of the cross-run ranking"
            )
        )
        rows = []
        for params in grid:
            key = params_key(params)
            try:
                res = replay_run(run, params, fail_ratio)
            except ValueError as exc:  # e.g. dt_start outside [dt_min, dt_max]
                rows.append((math.inf, key, None, str(exc)))
                continue
            if res.aborted:
                rows.append((math.inf, key, res, res.aborted))
                continue
            j = res.objective(cost.overhead)
            rows.append((j, key, res, None))
            if trusted:
                ratios[key].append(j / base_j)
                extrap[key].append(res.extrapolated_share)
        rows.sort(key=lambda row: row[0])
        lines.append(
            f"   {'candidate':52s} {'pred wall-h':>11s} {'vs own':>7s} "
            f"{'attempts':>8s} {'failed':>6s} {'extrap':>6s}"
        )
        lines.append(
            f"   {'(own settings) ' + format_key(params_key(run.params)):52s} "
            f"{cost.a * base_j / 3600:11.2f} {1:7.2f} {len(base.attempts):8d} "
            f"{base.failed:6d} {base.extrapolated_share:6.0%}"
        )
        for j, key, res, err in rows[:top]:
            if err:
                lines.append(f"   {format_key(key):52s} {err}")
                continue
            lines.append(
                f"   {format_key(key):52s} {cost.a * j / 3600:11.2f} "
                f"{j / base_j:7.2f} {len(res.attempts):8d} {res.failed:6d} "
                f"{res.extrapolated_share:6.0%}"
            )
        bad = [row for row in rows if row[3]]
        if bad:
            reasons: dict[str, int] = defaultdict(int)
            for row in bad:
                reasons[row[3].split(":")[0]] += 1
            lines.append(
                f"   ({len(bad)} candidates invalid or aborted: "
                + "; ".join(f"{n}x {r}" for r, n in reasons.items())
                + ")"
            )

    lines.extend(cross_check(runs, fail_ratio))

    n_trusted = len(runs) - len(untrusted)
    complete = {k: v for k, v in ratios.items() if n_trusted and len(v) == n_trusted}
    if untrusted:
        lines.append(f"== Left out of the cross-run ranking: {', '.join(untrusted)}")
    if n_trusted > 1 and complete:
        lines.append("== Across all runs (geometric mean of predicted cost / own settings)")
        order = sorted(complete, key=lambda k: float(np.exp(np.mean(np.log(complete[k])))))
        lines.append(f"   {'candidate':52s} {'geo-mean':>8s} {'worst':>6s} {'extrap':>6s}")
        for key in order[:top]:
            v = complete[key]
            lines.append(
                f"   {format_key(key):52s} {float(np.exp(np.mean(np.log(v)))):8.2f} "
                f"{max(v):6.2f} {float(np.mean(extrap[key])):6.0%}"
            )
    if complete:
        score = {k: float(np.exp(np.mean(np.log(v)))) for k, v in complete.items()}
        best = min(score, key=score.get)
        supported_keys = [k for k in score if float(np.mean(extrap[k])) <= 0.1]
        lines.append("== Suggested next settings")
        lines.append(
            f"   best predicted:          {format_key(best)} "
            f"(cost x{score[best]:.2f}, extrapolated {np.mean(extrap[best]):.0%})"
        )
        if supported_keys:
            safe = min(supported_keys, key=score.get)
            lines.append(
                f"   best with <=10% extrap.: {format_key(safe)} "
                f"(cost x{score[safe]:.2f}, extrapolated {np.mean(extrap[safe]):.0%})"
            )
    return lines


def cross_check(runs: list[Run], fail_ratio: float) -> list[str]:
    """Replay each run's settings on every other run with the same configuration."""
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.signature].append(run)
    lines = []
    for group in groups.values():
        for data_run, target in itertools.permutations(group, 2):
            horizon = min(data_run.t_reached, target.t_reached)
            res = replay_run(data_run, target.params, fail_ratio, horizon=horizon)
            actual = [a for a in target.attempts if a.t < horizon - T_TOL]
            text, worst = _compare(res, actual)
            if not lines:
                lines.append("== Cross-check: settings of one run replayed on another's data")
            lines.append(
                f"   {target.label} settings on {data_run.label} data (to "
                f"t={horizon:.4g}): predicted/actual {text}"
                + (f"  off by {worst:.0%}" if worst > SELF_CHECK_TOL else "  ok")
            )
    return lines


def write_csv(runs: list[Run], path: Path) -> None:
    names = [f.name for f in fields(Attempt)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *names])
        for run in runs:
            for a in run.attempts:
                writer.writerow([run.label, *(getattr(a, n) for n in names)])


def plot_runs(runs: list[Run], out_dir: Path) -> list[Path]:
    """dt against tau per transition interval, with the fitted dt_crit envelope."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for run in runs:
        envelope = Envelope(run)
        intervals = sorted({a.interval for a in run.attempts if a.is_transition})
        if not intervals:
            continue
        ncol = min(3, len(intervals))
        nrow = math.ceil(len(intervals) / ncol)
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow), squeeze=False
        )
        for ax, i in zip(axes.flat, intervals):
            atts = [a for a in run.attempts if a.interval == i]
            for ok, marker, color, label in (
                (True, "o", "tab:blue", "accepted"),
                (False, "x", "tab:red", "failed"),
            ):
                sel = [a for a in atts if a.ok == ok]
                ax.plot(
                    [a.tau + TAU_OFFSET for a in sel],
                    [a.dt for a in sel],
                    marker,
                    color=color,
                    label=label,
                    ms=4,
                )
            if i in envelope.knots:
                x, _ = envelope.knots[i]
                taus = np.logspace(0, x[-1] / math.log(10), 400) - TAU_OFFSET
                ax.plot(
                    taus + TAU_OFFSET,
                    [envelope.dt_crit(i, t) for t in taus],
                    "-",
                    color="0.4",
                    lw=0.8,
                    label="dt_crit fit",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"interval {i} (t0={run.schedule[i]:.4g} s)", fontsize=9)
            ax.set_xlabel("tau + 1 [s]")
            ax.set_ylabel("dt [s]")
        for ax in list(axes.flat)[len(intervals) :]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(run.label, fontsize=9)
        fig.tight_layout()
        path = out_dir / (run.label.replace("/", "__") + ".png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run output folders")
    parser.add_argument("--top", type=int, default=10, help="candidates to list")
    parser.add_argument("--csv", type=Path, help="write the per-attempt table here")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="write dt-vs-tau figures to figures/time_stepping/",
    )
    parser.add_argument(
        "--no-replay", action="store_true", help="only print diagnostics"
    )
    args = parser.parse_args(argv)

    runs = []
    for path in args.runs:
        try:
            runs.append(load_run(path))
        except (FileNotFoundError, KeyError) as exc:
            print(f"skipping {path}: {exc!r}")
    runs = [r for r in runs if r.attempts]
    if not runs:
        raise SystemExit("no usable runs")

    costs = [fit_cost_model(r) for r in runs]
    for run, cost in zip(runs, costs):
        print("\n".join(run_report(run, cost)))

    base = runs[0].params
    if any(params_key(r.params) != params_key(base) for r in runs[1:]):
        print(
            "\nnote: runs used different time-stepper settings; rules report "
            f"relative to the first run's: {format_key(params_key(base))}"
        )
    suggestions, lines = rule_report(runs, base)
    print("\n== Rule-based diagnostics (all runs pooled)")
    print("\n".join(lines))

    if args.csv:
        write_csv(runs, args.csv)
        print(f"\nwrote {args.csv}")
    if args.plot:
        for path in plot_runs(runs, REPO_ROOT / "figures" / "time_stepping"):
            print(f"wrote {path}")

    if not args.no_replay:
        ratios = [
            np.mean([a.newton for a in r.failed])
            / np.mean([a.newton for a in r.accepted])
            for r in runs
            if r.failed and r.accepted
        ]
        fail_ratio = float(np.median(ratios)) if ratios else 7.0
        grid = candidate_grid(base, suggestions)
        print(f"\n== Replay ranking ({len(grid)} candidates per run)")
        print("\n".join(rank(runs, costs, grid, fail_ratio, args.top)))


if __name__ == "__main__":
    main()
