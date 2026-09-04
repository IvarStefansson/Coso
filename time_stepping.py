from __future__ import annotations

import numpy as np
import porepy as pp
from porepy.time_stepper import (
    TargetNonlinearIterations,
    TimeInterval,
    TimeScheduler,
    TimeStepper,
    assemble_default_time_scheduler,
)


def build_time_stepper(
    schedule,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    iter_optimal_range: tuple[int, int] = (4, 7),
    iter_relax_factors: tuple[float, float] = (0.7, 1.3),
    recomp_factor: float = 0.5,
    max_attempts: int = 10,
    constant_dt: bool = False,
    dt_snap: float = 1e-6,
) -> TimeStepper:
    """Uniform (non-per-interval) time stepper.

    Direct replacement for a plain ``pp.TimeManager(...)`` call: ``dt_min_max`` ->
    ``dt_min``/``dt_max``, ``iter_optimal_range`` unchanged, ``iter_relax_factors``
    (decrease, increase) -> same order, ``recomp_factor`` -> retry_factor,
    ``recomp_max`` -> ``max_attempts`` (now a single global retry cap, not reset
    per interval). ``iter_max`` is dropped: the time-stepper pipeline never reads
    it, and Newton's iteration cap is controlled independently via
    ``nl_params.py``'s ``nl_max_iterations``.
    """
    scheduler = assemble_default_time_scheduler(
        schedule=schedule,
        dt_init=dt_init,
        constant_dt=constant_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        nonlinear_iter_optimal_range=iter_optimal_range,
        nonlinear_iter_relax_factors=iter_relax_factors,
        nonlinear_iter_retry_factor=recomp_factor,
        atol=dt_snap,
    )
    return TimeStepper(scheduler=scheduler, max_attempts=max_attempts)


def time_steppers_coso(
    schedule: np.ndarray,
    dt: float,
    iter_optimal_range: tuple[int, int] = (10, 15),
    iter_relax_factors: tuple[float, float] = (0.6, 2.5),
    recomp_factor: float = 0.2,
    max_attempts: int = 10,
    is_transition: np.ndarray | None = None,
    dt_start_transition: float = 2.0,
    dt_start_first: float | None = None,
) -> tuple[TimeStepper, TimeStepper]:
    """Per-interval time stepper for run_example_2/3's production schedule.

    Replacement for ``time_managers()``/``CosoTimeManager``: builds one
    :class:`TimeInterval` per schedule interval, each with its own ``dt_max``
    (fixing the previous bug where only interval 0's ``dt_max`` was ever
    applied to the whole simulation), and its own
    :class:`TargetNonlinearIterations` constraint.

    ``dt`` (the uniform ``dt_start``) is a reasonable seed for most intervals, but a
    comparison of a real run's ``times.json`` against the schedule
    (``create_schedule``'s ``is_transition`` mask) showed two very different
    situations hiding behind that one number:

    - At the schedule points where the well BC actually changes (Neumann/Dirichlet
      switch, ramp start/end -- ``is_transition[i] is True``), ``dt=1e3`` was
      routinely too large: the solver burned several failed Newton solves shrinking
      it via ``recomp_factor`` before finding a working step size (down to ~1s in
      the worst observed case).
    - At schedule points that only exist as ``num_bins_per_interval`` bin edges for
      post-processing (``is_transition[i] is False``), the physics does not change
      at all, yet ``dt`` was still reset all the way down to ``1e3`` and had to
      re-grow from scratch -- costing ~10 wasted converging steps to climb back to
      the multi-million-second scale the previous interval had already reached.

    Passing ``is_transition`` (aligned with ``schedule``, from ``create_schedule``)
    lets each interval pick a ``dt_start`` matching which of these two situations it
    is in: ``dt_start_transition`` (small) at real transitions, and this interval's
    own ``dt_max`` (i.e. start at the target ceiling right away) at bin-only edges.
    If ``is_transition`` is ``None``, every interval keeps using the uniform ``dt``
    as before.

    ``dt_start_first`` overrides the first interval's ``dt_start`` only, leaving the
    rest untouched; see the comment at its use below.
    """
    # Target reduction factor per interval boundary crossing. This sets the
    # maximum time step size for each interval as a fraction of the interval
    # length. Tailor per interval here if needed.
    ks = np.full(schedule.shape[0] - 1, 2.0)
    interval_lengths = np.diff(schedule)
    decrease_factor, increase_factor = iter_relax_factors

    if is_transition is None:
        dt_starts = np.full(schedule.shape[0] - 1, dt)
    else:
        dt_starts = np.where(
            is_transition[:-1], dt_start_transition, interval_lengths / ks
        )

    if dt_start_first is not None:
        # Used when resuming mid-schedule (see RESTART_FROM_CRASH in
        # run_example_3.py): the first interval then starts at an arbitrary accepted
        # step rather than a real schedule point, so its auto-picked dt_start (half
        # the remaining interval) bears no relation to the dt the interrupted run
        # had adaptively grown to. Passing that original dt reproduces the same step
        # instead of burning retries shrinking a far-too-large first guess.
        dt_starts[0] = dt_start_first

    intervals = [
        TimeInterval.create(
            t_start=t_start,
            dt_start=dt_start,
            dt_min=1e-2,
            dt_max=length / k,
            constraints=[
                TargetNonlinearIterations(
                    dt_min=1e-2,
                    iter_min=iter_optimal_range[0],
                    iter_max=iter_optimal_range[1],
                    increase_factor=increase_factor,
                    decrease_factor=decrease_factor,
                    retry_factor=recomp_factor,
                )
            ],
            name=f"interval_{i}",
        )
        for i, (t_start, length, k, dt_start) in enumerate(
            zip(schedule[:-1], interval_lengths, ks, dt_starts)
        )
    ]
    time_stepper = TimeStepper(
        scheduler=TimeScheduler(intervals=intervals, t_end=schedule[-1], dt_snap=1e-5),
        max_attempts=max_attempts,
    )

    # 5e10 s ~ 1580 years, which is safely below the production period for all cases.
    dt_init = 5e9
    time_stepper_init = build_time_stepper(
        # Small interval at the end for checking steady-state convergence after
        # initialization, cf, check_initialization_converged().
        schedule=[0, 3 * dt_init, 3.1 * dt_init],
        dt_init=dt_init,
        dt_min=1,
        dt_max=2 * dt_init,
        iter_optimal_range=(5, 12),
        constant_dt=False,
    )
    return time_stepper, time_stepper_init


def reset_time_io() -> None:
    """Clear the process-wide ``TimeIO`` singleton's accumulated history.

    ``porepy.time_stepper.scheduler.TimeIO`` is a process-lifetime singleton:
    every ``TimeScheduler``/``TimeSchedulerConstantDt`` grabs the same
    instance, and each successful time step appends to its
    ``exported_times``/``exported_dt`` lists, so writing ``times.json`` at any
    point dumps the *entire* accumulated history since process start. There is
    no reset hook and no per-scheduler injection point. Call this immediately
    before every ``ModelRunner(...).run()`` in a script that constructs more
    than one model (e.g. an init run followed by a main run, repeated over a
    sweep), or a run's ``times.json`` will contain entries left over from
    every prior run in the same process.
    """
    io = pp.time_stepper.scheduler.TimeIO()
    io.exported_times.clear()
    io.exported_dt.clear()
