"""Tests for analyze_time_stepping.py: loading, rules and the scheduler replay.

A synthetic run folder in tmp_path pins down the bookkeeping: Newton-iteration
slicing, recovering the controller's iteration count, the cost-model fit, the dt_crit
envelope, and R1's inert-band detection. A replay against a hand-made constant
dt_crit pins down that the real time_steppers_coso scheduler is driven the way a
simulation drives it. A smoke test on the completed reference run is skipped when
that folder is absent.
"""

from __future__ import annotations

import json
import math
import os

import pytest

import analyze_time_stepping as ats

REFERENCE_RUN = (
    ats.REPO_ROOT
    / "Case_III_applied/faults_00_01_03_05_07/velocity_1e-06_period_1e+00"
)

SCHEDULE = [0.0, 100.0, 200.0]
IS_TRANSITION = [True, False, True]
# (start time, dt, success, Newton iterations of each subsolver call). With the
# sequential scheme's two subsolvers, the controller's count is half the calls.
ATTEMPTS = [
    (0.0, 1.0, True, [2, 1]),
    (1.0, 2.5, True, [3, 2, 1, 1]),
    (3.5, 6.25, False, [10, 10]),
    (3.5, 1.25, True, [2, 1]),
    (4.75, 3.125, True, [1, 1, 2, 1]),
    # Three outer iterations whose last pT solve took 25 Newton iterations: the
    # entry's "num_iterations" (25) is *not* what the controller saw (3).
    (7.875, 7.8125, True, [1, 25, 1, 25, 1, 25]),
    (100.0, 50.0, True, [3, 1]),
    (150.0, 50.0, True, [2, 2]),
]
# Wall-clock model used to stamp the numbered pvd files.
SECONDS_PER_NEWTON, SECONDS_PER_STEP = 2.0, 10.0


def write_run(folder, attempts=ATTEMPTS, time_stepper_params=None):
    """Write the files analyze_time_stepping reads, as a Coso run would."""
    folder.mkdir(parents=True, exist_ok=True)
    history, stats = [], {}
    for t, dt, ok, calls in attempts:
        history.extend(calls)
        stats[str(len(history) - 1)] = {
            "simulation_status": "successful" if ok else "failed",
            "time": t,
            "dt": dt,
            # As porepy writes it: the iteration count of the last subsolver call.
            "num_iterations": calls[-1],
        }
    stats["global"] = {
        "num_iterations_history": history,
        "total_num_iterations": sum(history),
    }
    times = [SCHEDULE[0]] + [t + dt for t, dt, ok, _ in attempts if ok]
    cfg = {
        "schedule": SCHEDULE,
        "is_transition": IS_TRANSITION,
        "dt": 1.0,
        "model_params": {"file_name": "example_3"},
    }
    if time_stepper_params is not None:
        cfg["time_stepper_params"] = time_stepper_params
    for name, data in (
        ("solver_statistics.json", stats),
        ("times.json", {"time": times, "dt": []}),
        ("run_config.json", cfg),
    ):
        (folder / name).write_text(json.dumps(data))

    def stamp(j, mtime):
        pvd = folder / f"example_3_{j:06d}.pvd"
        pvd.write_text("")
        os.utime(pvd, (mtime, mtime))

    mtime, pending, j = 1.0e9, 0, 0
    stamp(j, mtime)
    for _, _, ok, calls in attempts:
        pending += sum(calls)
        if ok:
            j += 1
            mtime += SECONDS_PER_NEWTON * pending + SECONDS_PER_STEP
            pending = 0
            stamp(j, mtime)
    return folder


@pytest.fixture
def run(tmp_path):
    return ats.load_run(write_run(tmp_path / "run"))


def test_newton_slicing_and_intervals(run):
    assert len(run.attempts) == len(ATTEMPTS)
    assert sum(a.newton for a in run.attempts) == run.total_newton_reported
    failed = run.attempts[2]
    assert not failed.ok and failed.newton == 20
    # The failed attempt's cost is charged to the step that finally succeeded.
    assert list(run.newton_per_step[:3]) == [3, 7, 23]
    bin_step = run.attempts[6]
    assert bin_step.interval == 1 and not bin_step.is_transition
    assert run.attempts[7].tau == pytest.approx(50.0)
    assert run.finished and run.params_assumed and run.solver_scheme == "sequential"


def test_controller_count_is_outer_iterations(run):
    assert [a.outer for a in run.attempts] == [1, 2, 1, 1, 2, 3, 1, 1]
    assert run.attempts[5].newton == 78


def test_monolithic_count_is_newton_iterations(tmp_path):
    run = ats.load_run(write_run(tmp_path / "Case_III_monolithic" / "run"))
    assert run.solver_scheme == "monolithic"
    assert run.attempts[5].outer == 25


def test_recorded_params_are_used(tmp_path):
    params = {"iter_optimal_range": [2, 2], "recomp_factor": 0.4}
    run = ats.load_run(write_run(tmp_path / "run", time_stepper_params=params))
    assert not run.params_assumed
    assert run.params["iter_optimal_range"] == (2, 2)
    assert run.params["recomp_factor"] == 0.4
    # Unrecorded keys fall back to the legacy values.
    assert run.params["iter_relax_factors"] == (0.6, 2.5)


def test_cost_model_recovers_wall_clock_law(run):
    cost = ats.fit_cost_model(run)
    assert cost.a == pytest.approx(SECONDS_PER_NEWTON, rel=1e-6)
    assert cost.b == pytest.approx(SECONDS_PER_STEP, rel=1e-6)
    assert cost.overhead == pytest.approx(5.0, rel=1e-6)


def test_envelope(run):
    envelope = ats.Envelope(run)
    # One failure (tau=3.5, dt=6.25), retried successfully with dt=1.25 from the
    # same time: the knot is their geometric mean, held constant before the knot and
    # unbounded after it.
    crit = math.sqrt(1.25 * 6.25)
    assert envelope.dt_crit(0, 0.0) == pytest.approx(crit)
    assert envelope.dt_crit(0, 3.5) == pytest.approx(crit)
    assert envelope.dt_crit(0, 10.0) == math.inf
    assert envelope.dt_crit(1, 0.0) == math.inf
    # dt_crit never drops below the observed accepted step covering tau.
    assert envelope.covering_dt(0, 2.0) == 2.5
    assert envelope.covering_dt(0, 3.5) == 1.25
    # Every observed attempt is classified as it actually went.
    for a in run.attempts:
        assert (a.dt <= envelope.dt_crit(a.interval, a.tau)) == a.ok
    # dt=3 at tau=5 lies inside the accepted step (tau=4.75, dt=3.125); dt=4 does not
    # and nothing accepted near tau=5 was that large.
    assert envelope.supported(0, 5.0, 3.0, ok=True)
    assert not envelope.supported(0, 5.0, 4.0, ok=True)
    # A failure at dt=7 is backed by the failure at (3.5, 6.25); at tau=0.5 it is not.
    assert envelope.supported(0, 3.5, 7.0, ok=False)
    assert not envelope.supported(0, 0.5, 7.0, ok=False)


def test_outer_predictor(run):
    outer = ats.OuterPredictor(run)
    # Exact on the observed trajectory, including the rare 3.
    assert outer(0, 7.875, 7.8125) == 3
    # Far from any observation: the run's most common count, not the nearest outlier.
    assert outer(0, 7.875, 1e-3) == 1
    assert not outer.near(0, 7.875, 1e-3)


def test_rules_flag_inert_band(run):
    suggestions, lines = ats.rule_report([run], run.params)
    assert "INERT" in "\n".join(lines)  # iter_min=10, but no accepted step exceeds 3


def test_suggest_band():
    # Growing after 1 iteration failed 5/55, after 2 iterations 20/79; 3 was seen
    # once and is ignored.
    assert ats.suggest_band({1: (55, 5), 2: (79, 20), 3: (1, 1)}) == (2, 2)
    # Growth safe after 1 and 2 -> grow below 3, hold at 3; 4 fails too often.
    assert ats.suggest_band({1: (10, 0), 2: (10, 1), 3: (10, 3), 4: (10, 6)}) == (3, 3)
    assert ats.suggest_band({1: (10, 0), 2: (10, 1), 3: (10, 6), 4: (10, 2)}) == (3, 4)
    assert ats.suggest_band({1: (2, 0)}) is None


def test_retry_evidence(tmp_path):
    attempts = [
        (0.0, 1.0, True, [1, 1]),
        (1.0, 2.0, True, [1, 1]),
        (3.0, 4.0, False, [9, 9]),  # grew from 2 -> evidence 2/4
        (3.0, 0.8, True, [1, 1]),
        (3.8, 2.0, False, [9, 9]),  # grew from 0.8; two failures, retry 0.08 works
        (3.8, 0.4, False, [9, 9]),
        (3.8, 0.08, True, [1, 1]),
    ]
    run = ats.load_run(write_run(tmp_path / "run", attempts=attempts))
    ratios, n_seq, n_multi = ats.retry_evidence(run)
    assert (n_seq, n_multi) == (2, 1)
    assert ratios == pytest.approx([0.5, 0.4])


def test_cycle_length():
    # The reference settings: 0.2 cut, x2.5 growth -> fails again after 2 steps.
    assert ats.cycle_length(2.5, 0.2) == 2
    assert ats.cycle_length(2.0, 0.5) == 1


def _replay(params, dt_crit, outer, horizon):
    return ats.replay(
        ats.np.array([0.0, 100.0]),
        ats.np.array([True, True]),
        1.0,
        params,
        dt_crit=lambda i, tau: dt_crit,
        outer=lambda i, tau, dt: outer,
        newton=lambda ok, count: 1.0 if ok else 10.0,
        horizon=horizon,
    )


REPLAY_PARAMS = {
    "iter_optimal_range": (10, 15),
    "iter_relax_factors": (0.6, 2.0),
    "recomp_factor": 0.5,
    "max_attempts": 10,
    "dt_start_transition": 1.0,
    "dt_max_divisor": 2.0,
    "dt_min": 1e-2,
}


def test_replay_reproduces_grow_fail_cycle():
    # dt_crit = 5 and an inert band: dt doubles 1 -> 2 -> 4, 8 fails, the retry at 4
    # succeeds, growth tries 8 again, ... -- the limit cycle seen in real runs.
    res = _replay(REPLAY_PARAMS, dt_crit=5.0, outer=1, horizon=15.0)
    assert res.attempts == [
        (0.0, 1.0, True),
        (1.0, 2.0, True),
        (3.0, 4.0, True),
        (7.0, 8.0, False),
        (7.0, 4.0, True),
        (11.0, 8.0, False),
        (11.0, 4.0, True),
    ]
    assert (res.accepted, res.failed, res.newton) == (5, 2, 25.0)
    assert res.aborted is None


def test_replay_band_holds_growth():
    # With band (2, 2) and 2 iterations per step, dt is held at dt_start.
    params = dict(REPLAY_PARAMS, iter_optimal_range=(2, 2))
    res = _replay(params, dt_crit=5.0, outer=2, horizon=5.0)
    assert [dt for _, dt, _ in res.attempts] == [1.0] * 5
    assert res.failed == 0


def test_replay_reports_abort():
    # dt_crit below dt_min: every retry fails until the scheduler gives up.
    res = _replay(REPLAY_PARAMS, dt_crit=1e-3, outer=1, horizon=10.0)
    assert res.aborted is not None and res.accepted == 0


@pytest.mark.skipif(not REFERENCE_RUN.exists(), reason="reference run not present")
def test_reference_run_smoke():
    run = ats.load_run(REFERENCE_RUN)
    assert run.finished
    assert sum(a.newton for a in run.attempts) == run.total_newton_reported == 4481
    assert (len(run.accepted), len(run.failed)) == (162, 28)
    assert ats.fit_cost_model(run).r2 > 0.99
    suggestions, lines = ats.rule_report([run], run.params)
    assert "INERT" in "\n".join(lines)
    # The replay of the run's own settings reproduces the run step by step.
    res = ats.replay_run(run, run.params, fail_ratio=7.0)
    replayed = [(round(t, 3), round(dt, 3), ok) for t, dt, ok in res.attempts]
    actual = [(round(a.t, 3), round(a.dt, 3), a.ok) for a in run.attempts]
    assert replayed == actual
