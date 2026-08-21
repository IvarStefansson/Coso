"""Analyze a run's solver_statistics.json for actual retry counts.

Complements the schedule-vs-times.json analysis used to design time_steppers_coso's
per-interval dt_start tailoring (see time_stepping.py): that analysis could only
*infer* retries indirectly, since times.json only records successful steps. Once a
run sets params["solver_statistics_file_name"] (wired up in run_example_3.py), the
resulting solver_statistics.json records every nonlinear-solve attempt, including
failed ones -- model.before_nonlinear_loop() increments the statistics index once per
NewtonSolver.solve() call, i.e. once per attempt, not once per successful time step.
This lets us directly verify whether the dt_start tailoring reduced retries at
transition boundaries, instead of inferring it from how far below dt_start the first
accepted dt landed.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_attempts(solver_statistics_path: Path | str) -> list[dict]:
    """Return the per-attempt entries from a solver_statistics.json, in order.

    Skips the "global" summary key. Entries are already in attempt order since the
    statistics index increases monotonically with each NewtonSolver.solve() call.
    """
    with open(solver_statistics_path) as f:
        data = json.load(f)
    indices = sorted((k for k in data if k != "global"), key=int)
    return [data[k] for k in indices]


def summarize_retries(solver_statistics_path: Path | str) -> list[dict]:
    """Group consecutive attempts into per-time-step records.

    Returns one dict per *successful* time step (plus a trailing record for any
    attempts left over if the run ended mid-step, e.g. it was killed), each with:
        - "attempts": number of solve attempts for this time step (1 = no retries).
        - "num_iterations": iteration count of the attempt that succeeded (or the
          last attempt made, if the step never converged).
        - "converged": whether the final attempt in the group succeeded.
    A time step needing N failed attempts before succeeding shows up as one record
    with ``attempts == N + 1``, not N+1 separate entries -- this is what you want
    when asking "how many retries did this step cost", matching the granularity of
    the schedule-interval comparison against times.json.
    """
    attempts = load_attempts(solver_statistics_path)
    records = []
    current_group: list[dict] = []
    for attempt in attempts:
        current_group.append(attempt)
        status = attempt.get("solver_status")
        if status == "successful":
            records.append(
                {
                    "attempts": len(current_group),
                    "num_iterations": attempt.get("num_iterations"),
                    "converged": True,
                }
            )
            current_group = []
    if current_group:
        # Run ended (e.g. killed) mid-step: attempts made but never converged.
        records.append(
            {
                "attempts": len(current_group),
                "num_iterations": current_group[-1].get("num_iterations"),
                "converged": False,
            }
        )
    return records


def print_retry_summary(solver_statistics_path: Path | str) -> None:
    """Print a compact per-time-step retry report."""
    records = summarize_retries(solver_statistics_path)
    total_attempts = sum(r["attempts"] for r in records)
    total_retries = sum(r["attempts"] - 1 for r in records)
    print(
        f"{len(records)} time steps, {total_attempts} total solve attempts, "
        f"{total_retries} were retries after a failed attempt"
    )
    for i, r in enumerate(records):
        if r["attempts"] > 1 or not r["converged"]:
            flag = "" if r["converged"] else "  <- run ended without converging"
            print(
                f"  step #{i}: {r['attempts']} attempts "
                f"(final num_iterations={r['num_iterations']}){flag}"
            )


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("solver_statistics.json")
    print_retry_summary(path)
