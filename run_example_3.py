import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
import porepy as pp
from porepy.applications.test_utils.models import add_mixin

from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
    SmoothWellTransitions,
)
from geometry import (
    ConstraintsCapcrockAndReservoirDepth,
    FaultPlaneGeometry,
)
from porepy.models.contact_mechanics import (
    RadialReturnTangentialContactMechanicsEquation,
)
from material_parameters import granodiorite_values
from physical_model import (
    HeterogeneousFrictionCoefficient,
    HeterogeneousPermeabilitySpecification,
    PhysicalModel,
    HagenPoiseuilleWellPermeability,
    CosoBackgroundValues,
)
from diff_tpfa import DarcysLawAdEverywhere
from time_stepping import reset_time_io, time_steppers_coso

try:
    import pp_solvers
except ImportError:
    pp_solvers = None

import copy
import logging
import os
import sys
import time
from datetime import datetime

from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
)
from porepy.examples.geothermal_reservoir import WellBoundaryConditions
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

from exporting import CosoExporter, GeometryExporting
from initial_conditions import CopyInitialCondition
from solution_strategy import SolutionStrategy
from nl_params import set_nonlinear_solver
from wells import _WellDataBase

REPO_ROOT = Path(__file__).resolve().parent

LOG_TO_FILE = False
log_dir = "example_3_logs"
FINAL_TIME = 2 * pp.YEAR
SHUT_IN_DURATION = 3 * pp.DAY
PRODUCTION_WELL = "2 Production well"
NUM_BINS_PER_INTERVAL = 3
USE_ITERATIVE_SOLVER = True
USE_CONSTRAINTS = (
    False  # If False, omit ConstraintsCapcrockAndReservoirDepth from the model stack
)
RESTART_FROM_CRASH = True
# Debugging aid: resume both the initialization and main models from their own
# previously saved output (vtu/pvd/times.json) under `folder_name_init`/`folder_name`,
# instead of re-simulating from t=0. Built to fast-forward to the HYPRE
# BoomerAMGCreateS heap-corruption crash (time step 38, Newton iter 20 of the
# "velocity_1e-06_period_1e+00" case) in minutes instead of hours -- see
# `linear_system_dumps/` (pp_solvers.IterativeLinearSolver debug_dump_dir, wired in
# nl_params.py) for capturing the triggering matrix once reproduced. The
# initialization model is loaded straight to its converged end state (no re-run of
# its own short equilibration loop); the main model resumes time-stepping from
# whatever schedule checkpoint its restart source last completed, writing new output
# to a separate "_restart_debug" folder so the original crash evidence is untouched.
RESTART_DRY_RUN = False
# If RESTART_FROM_CRASH: stop right after loading state for both models (before any
# Newton solve) and print the restored time -- verifies the restart actually landed
# on the expected checkpoint before committing to a real (multi-minute) run.
RESTART_STEPS_BACK = 1
# How many saved states to rewind past the last one, for the main model. 0 resumes at
# the last saved state, i.e. straight into the step that crashed -- which reproduces
# the crash fastest but proves nothing about whether the restored state is sound (a
# corrupted state can fail too). 1 (default) resumes one step earlier, so the run must
# first re-simulate a step whose outcome is already known from the source run's
# times.json (compare the reported t/dt against it) before reaching the crashing step.
if USE_ITERATIVE_SOLVER:
    if "pp_solvers" not in sys.modules:
        raise ImportError(
            "pp_solvers module is required for iterative solvers. Please install pp_solvers"
            + " or set use_iterative_solver to False."
        )
    from solver_configurations import (
        linear_solver_params,
        linear_solver_selector_params,
    )


if LOG_TO_FILE:
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_example_3_{timestamp}.log")

    # Configure root logger to direct ALL logging to the file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add file handler to root logger
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)


class _BaseModel(
    FractureDeformationExporting,
    GeometryExporting,
    CosoExporter,
    FaultPlaneGeometry,
    HagenPoiseuilleWellPermeability,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    _WellDataBase,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    RadialReturnTangentialContactMechanicsEquation,
    PhysicalModel,
):
    """Model for the Coso geothermal reservoir."""

    @property
    def time_step_indices(self) -> np.ndarray:
        # porepy's default is [0] only (most recent converged step). We need index
        # 1 (the step before that) too:
        #  - check_initialization_converged() reads it directly to measure how much
        #    the solution changed on the last initialization step.
        #  - CosoExporter.collect_data()'s fracture displacement-jump increment
        #    reads it because by the time collect_data() runs (from
        #    after_time_step_convergence, called after update_time_step_solution()
        #    has already shifted the new step into index 0), index 0 no longer
        #    holds a "previous" value to diff against -- only index 1 does.
        return np.array([0, 1])

    def create_plot_title(self) -> str:
        """Generate a formatted plot title from folder name and simulation parameters.

        Extracts simulation metadata from folder name (strike angle, fracture index,
        well status) and formats it as a human-readable title. Fracture indices are
        incremented by 1 for display purposes (0-indexed internally, 1-indexed for
        display).

        Returns:
            str | None: Formatted plot title with well status, strike angle, and
            fracture info.

        """
        fn = self.params["plot_title"]
        return fn


if USE_CONSTRAINTS:

    class BaseModel(ConstraintsCapcrockAndReservoirDepth, _BaseModel):
        """Main model for the Coso geothermal reservoir."""
else:

    class BaseModel(_BaseModel):
        """Main model for the Coso geothermal reservoir."""


class InitializationModel(
    CosoBackgroundValues,
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    BaseModel,
):
    """Initialization model for the Coso geothermal reservoir."""


class MainModel(
    CosoBackgroundValues,
    SmoothWellTransitions,
    NeumannWellBCsFromSchedule,
    CopyInitialCondition,
    WellBoundaryConditions,
    CosoBoundaryConditionsDisplacement,
    HeterogeneousPermeabilitySpecification,
    HeterogeneousFrictionCoefficient,
    BaseModel,
):
    """Main model for the Coso geothermal reservoir."""


def create_schedule(
    production_period: float,
    shut_in_duration: float = 3 * pp.DAY,
    final_time: float = 10 * pp.YEAR,
    num_bins_per_interval: int | None = None,
    transition_duration: float = 0.0,
    initial_shutin_duration: float = 0.0,
    with_production: bool = True,
) -> tuple[np.ndarray, Sequence[tuple[float, float]], np.ndarray]:
    """Build a simulation time schedule with alternating production and shut-in periods.

    Each cycle consists of a production sub-interval of length ``production_period -
    shut_in_duration`` followed by a shut-in sub-interval of length
    ``shut_in_duration``.  The schedule always starts at t=0 and ends at the last cycle
    boundary that is needed to cover ``final_time``.

    Shut-in periods correspond to intervals where Neumann (zero-flux) well boundary
    conditions are applied instead of prescribed pressures.  The returned
    ``neumann_intervals`` list marks exactly these shut-in windows so the boundary
    condition mixin can switch behaviour.

    If ``num_bins_per_interval`` is given, equally-spaced bin-boundary times are
    inserted into the schedule for *both* the production and shut-in sub-intervals of
    every cycle.  This ensures that the time integrator always outputs a state at each
    bin edge, so that post-processing (slip-rate / flux binning) can read off values at
    exactly the right times without interpolation.

    Parameters:
        production_period: Duration of one full cycle (production + shut-in) in seconds.
        shut_in_duration: Duration of the shut-in portion at the end of each cycle in
            seconds (default: 3 days).
        final_time: Total simulation end time in seconds (default: 10 years).  The
            number of cycles is rounded up so the schedule covers at least this long.
        num_bins_per_interval: Number of equal bins per sub-interval (production *and*
            shut-in).  When given, ``num_bins_per_interval + 1`` evenly-spaced boundary
            times are added to the schedule for each sub-interval.
        initial_shutin_duration: Duration of an optional zero-flux shut-in period at
            the very start of the simulation (seconds, default 0 = disabled).  When
            non-zero, Neumann BCs are active on all well grids for
            ``(0, initial_shutin_duration - transition_duration]``, then the existing
            production-ramp machinery linearly ramps the well-head pressure from the
            cached no-flow pressure to the scheduled target over the final
            ``transition_duration`` seconds of the initial shut-in.  Must be strictly
            greater than ``transition_duration`` when ``transition_duration > 0``.

    Returns:
        tuple: A tuple containing:
            - schedule (np.ndarray): Sorted array of time checkpoints (seconds),
              starting at 0 and ending at the last cycle boundary.
            - neumann_intervals (list[tuple[float, float]]): One ``(start, end)``
              tuple per shut-in period, used to activate zero-flux well BCs.
            - is_transition (np.ndarray): Boolean mask aligned with ``schedule``,
              True where the checkpoint is a "mandatory" one (BC type switch or
              ramp start/end) and False where it is a purely cosmetic bin-boundary
              inserted by ``num_bins_per_interval`` for post-processing. Time steps
              starting right after a mandatory checkpoint tend to be numerically
              stiff (BC discontinuity), while steps after a bin-only checkpoint
              continue the same smooth physics as before it. Intended for
              :func:`time_stepping.time_steppers_coso` to pick a small ``dt_start``
              only where the physics actually changes.
    """
    if not with_production:
        # No need to add transition checkpoints if there is no production, since the BC
        # type never changes.
        transition_duration = 0.0
    if transition_duration > shut_in_duration:
        raise ValueError(
            f"transition_duration ({transition_duration:.3g} s) must not exceed "
            f"shut_in_duration ({shut_in_duration:.3g} s). The shut-in ramp and the "
            "production ramp would otherwise overlap."
        )

    # Number of complete cycles required to reach final_time.
    n = int(np.ceil(final_time / production_period))

    # For each cycle i (1-indexed), add the shut-in start (= production end) and the
    # cycle end.  These are the mandatory checkpoint times that delimit each sub-
    # interval boundary.  t=0 is prepended separately below. If a transition duration is
    # set, also insert the time of full shut-in (shut-in start + transition duration)
    # and the time of production ramp start (cycle end - transition duration) for each
    # cycle. This ensures that the time integrator always steps at these critical times
    # where the BC type or target value changes, so the transition is well-resolved and
    # consistent across runs. The production ramp start is placed exactly at the end of
    # the transition window that follows and the Neumann interval end (= cycle end −
    # transition_duration) so the Neumann→Dirichlet switch always starts at an exact
    # schedule boundary.
    checkpoints = []
    for i in np.arange(1, n + 1):
        checkpoints.append(i * production_period - shut_in_duration)
        if transition_duration > 0.0:
            checkpoints.append(
                i * production_period - shut_in_duration + transition_duration
            )
            checkpoints.append(i * production_period - transition_duration)
        checkpoints.append(i * production_period)
    schedule = np.array(checkpoints)

    # Each Neumann interval covers the pure zero-flux part of the shut-in window:
    #   start = shut-in start  (= i*T - d_shutin)
    #   end   = transition start (= i*T - d_trans)
    # The final ``transition_duration`` seconds of each shut-in period are NOT included
    # in the Neumann interval; they form the production ramp window during which the BC
    # type is already Dirichlet but the well-head pressure is ramped from the last
    # reservoir pressure toward the scheduled target.  When transition_duration == 0,
    # end == i*T and the full shut-in window is Neumann (legacy behaviour).
    neumann_intervals = [
        (
            i * production_period - shut_in_duration,
            i * production_period - transition_duration,
        )
        for i in np.arange(1, n + 1)
    ]

    # Prepend t=0 so the time manager starts from the initial condition.
    schedule = np.insert(schedule, 0, 0.0)

    # If requested, prepend an initial shut-in Neumann interval and insert its mandatory
    # checkpoints into the schedule. The pure zero-flux period runs from t=0 to
    # t=(initial_shutin_duration - transition_duration); the final transition_duration
    # seconds of the initial shut-in are the production ramp window, handled
    # automatically by _production_start_times() / _transition_progress().
    if initial_shutin_duration > 0.0:
        if transition_duration > 0.0 and initial_shutin_duration <= transition_duration:
            raise ValueError(
                f"initial_shutin_duration ({initial_shutin_duration:.3g} s) must be "
                f"strictly greater than transition_duration ({transition_duration:.3g} s) "
                "so that at least one converged zero-flux step populates the pressure cache."
            )
        ramp_start = initial_shutin_duration - transition_duration
        extra: list[float] = [initial_shutin_duration]
        if transition_duration > 0.0:
            extra.append(ramp_start)
        schedule = np.unique(np.concatenate([schedule, extra]))
        neumann_intervals = [(0.0, ramp_start)] + neumann_intervals

    # All checkpoints up to this point are "mandatory": they mark an actual BC type
    # switch or ramp start/end. Anything added below (bin boundaries) is purely for
    # post-processing and does not correspond to a change in physics.
    mandatory_schedule = schedule

    if num_bins_per_interval is not None:
        bin_times = []
        for i in np.arange(1, n + 1):
            # Production sub-interval: from the end of the previous cycle to the
            # start of this cycle's shut-in.
            prod_start = (i - 1) * production_period
            prod_end = i * production_period - shut_in_duration
            bin_times.append(
                np.linspace(prod_start, prod_end, num_bins_per_interval + 1)
            )
            # Shut-in sub-interval: from shut-in start to cycle end.
            shut_start = prod_end
            shut_end = i * production_period
            bin_times.append(
                np.linspace(shut_start, shut_end, num_bins_per_interval + 1)
            )
        # Merge bin boundaries with the existing mandatory checkpoints and deduplicate.
        schedule = np.unique(np.concatenate([schedule] + bin_times))

    is_transition = np.isin(schedule, mandatory_schedule)

    return schedule, neumann_intervals, is_transition


def _latest_numbered_pvd(
    folder: Path,
    file_name: str,
    steps_back: int = 0,
    before_mtime: float | None = None,
) -> Path:
    """The numbered pvd file holding the final state of the run that wrote
    ``folder``'s current times.json, optionally rewound ``steps_back`` steps.

    Output folders are never cleared between runs, so they accumulate debris from
    several runs at once -- and it bites in *both* directions, so neither "highest
    index" nor "newest mtime" is safe:

    - A shorter run leaves behind higher-numbered files from a previous, longer
      one, possibly on a different mesh. The crashed main run reached ``_000038``
      while ``_000176`` was still lying around; restoring from that one trips a
      shape-mismatch assertion inside `Exporter.import_state_from_vtu`.
    - A *later* run can overwrite a *lower*-numbered file. In the initialization
      folder, ``_000003`` was rewritten hours after ``_000004``, by a run whose
      state carried a 1e9 pressure error -- restoring from it silently poisons the
      initialization state (and everything derived from it) rather than failing.

    So anchor on times.json instead: its final write and the final state's pvd
    write happen within the same `save_data_time_step` call, seconds apart, which
    picks out the right file regardless of index or of what else is in the folder.
    """
    numbered_pvds = list(folder.glob(f"{file_name}_" + "[0-9]" * 6 + ".pvd"))
    if not numbered_pvds:
        raise FileNotFoundError(
            f"No numbered pvd files matching '{file_name}_NNNNNN.pvd' found in "
            f"{folder} -- nothing to restart from."
        )
    if before_mtime is not None:
        # Restrict to files written before a given moment -- used to pin the
        # initialization state to the run that actually seeded the main run being
        # restarted, ignoring anything a *later* run wrote into the same folder.
        numbered_pvds = [
            pvd for pvd in numbered_pvds if pvd.stat().st_mtime < before_mtime
        ]
        if not numbered_pvds:
            raise FileNotFoundError(
                f"No numbered pvd files in {folder} predate the requested cutoff "
                "-- nothing to restart from."
            )
        # The last state written before the cutoff, i.e. that run's final state.
        anchor = max(numbered_pvds, key=lambda pvd: pvd.stat().st_mtime)
    else:
        times_mtime = (folder / "times.json").stat().st_mtime
        anchor = min(
            numbered_pvds, key=lambda pvd: abs(pvd.stat().st_mtime - times_mtime)
        )
    if steps_back == 0:
        return anchor
    # Step back by index from the anchor, so intermediate files rewritten by other
    # runs cannot be selected on mtime.
    wanted = anchor.with_name(
        f"{file_name}_{int(anchor.stem[-6:]) - steps_back:06d}.pvd"
    )
    if not wanted.exists():
        raise FileNotFoundError(
            f"{wanted} does not exist -- cannot rewind {steps_back} step(s) back "
            f"from {anchor.name}."
        )
    return wanted


def latest_restart_options(
    folder_name: str, file_name: str, steps_back: int = 0
) -> dict:
    """Build ``restart_options`` for resuming *time-stepping* from the latest saved
    checkpoint in ``folder_name`` (used for the main model).

    Uses porepy's ``pvd_file`` + ``is_mdg_pvd=True`` restart path: besides being the
    read source, ``pvd_file`` also feeds ``write_pvd_and_vtu``'s write-continuation
    (``Exporter.write_pvd(..., append=True, from_pvd_file=pvd_file)``, called on
    every subsequent successful step), which requires a valid ``pvd_file`` entry
    regardless of how state was loaded -- so this is the only restart_options shape
    safe to use for a model that will keep exporting afterward. Confirmed correct
    (by cross-referencing against the crashed run's own log) for this main-run
    folder structure: the numbered pvd counter and times.json length agree there.

    Parameters:
        folder_name: A previous run's output folder (contains the numbered pvd/vtu
            files and times.json).
        file_name: The base file name used for exporting (e.g. "example_3").

    Returns:
        A dict suitable for ``model_params["restart_options"]``.

    """
    latest_pvd = _latest_numbered_pvd(Path(folder_name), file_name, steps_back)
    times_file = latest_pvd.parent / "times.json"
    with open(times_file) as f:
        num_exported = len(json.load(f)["time"])
    pvd_index = int(latest_pvd.stem[-6:])
    if pvd_index != num_exported - 1 - steps_back:
        raise ValueError(
            f"{latest_pvd.name}'s own counter ({pvd_index}) is not the target "
            f"times.json index ({num_exported - 1 - steps_back}) in {folder_name}. "
            "porepy's "
            "pvd-based restart derives the time index from the filename, so it "
            "would read the wrong entry (or run off the end of the history). Use "
            "latest_restart_options_state_only() instead if this model does not "
            "need to keep exporting."
        )
    return {
        "restart": True,
        "pvd_file": latest_pvd,
        "is_mdg_pvd": True,
        "times_file": times_file,
    }


def latest_restart_options_state_only(
    folder_name: str,
    file_name: str,
    steps_back: int = 0,
    before_mtime: float | None = None,
) -> dict:
    """Build ``restart_options`` for loading *state only* (no further time-stepping
    or exporting expected) from the latest saved checkpoint in ``folder_name`` (used
    for the initialization model, which we load straight to its converged end state
    and never run or export from again).

    Unlike :func:`latest_restart_options`, does NOT trust the numbered ("mdg") pvd
    file's own filename counter as a times.json index: confirmed (by
    cross-referencing file mtimes) that for an initialization run's folder the
    numbered-pvd counter runs one ahead of times.json's length -- its scheduler
    class doesn't log the pre-loop initial-condition export to times.json, unlike
    the main model's. So instead this resolves the vtu files to load from the
    latest pvd's own ``<DataSet file=...>`` entries (that part of the data is fine)
    but passes the correct index explicitly via porepy's ``vtu_files`` +
    ``time_index`` restart path. ``len(exported times) - 1`` is always right by
    construction: times.json is only ever appended to, once per actually completed
    and exported time step.

    Caution: the resulting dict has no ``pvd_file`` key, so the model this is used
    on must not call ``save_data_time_step``/``write_pvd_and_vtu`` afterward (it
    unconditionally reads ``restart_options["pvd_file"]`` for write-continuation,
    regardless of how state was loaded) -- including the one auto-export inside
    ``prepare_simulation()`` itself; the caller must no-op that out first (see
    RESTART_FROM_CRASH handling around ``init_model.prepare_simulation()``).

    Parameters:
        folder_name: A previous run's output folder (contains the numbered pvd/vtu
            files and times.json).
        file_name: The base file name used for exporting (e.g. "example_3").

    Returns:
        A dict suitable for ``model_params["restart_options"]``.

    """
    folder = Path(folder_name)
    latest_pvd = _latest_numbered_pvd(folder, file_name, steps_back, before_mtime)
    vtu_files = [
        latest_pvd.parent / dataset.attrib["file"]
        for dataset in ET.parse(latest_pvd).getroot().iter("DataSet")
    ]
    times_file = folder / "times.json"
    with open(times_file) as f:
        num_exported = len(json.load(f)["time"])
    return {
        "restart": True,
        "vtu_files": vtu_files,
        "time_index": num_exported - 1 - steps_back,
        "times_file": times_file,
    }


def names_from_params(
    velocity: float, period: float, with_production: bool, use_iterative_solver: bool
):
    prod_suffix = "" if with_production else "_no_production"
    simulation_name = f"velocity_{velocity:.0e}_period_{period:.0e}{prod_suffix}"
    # While debugging iterative solvers, distinguish to avoid overwriting the production
    # runs with direct solvers.
    # Anchored to REPO_ROOT (this file's own directory), not left relative -- a relative
    # folder_name resolves against whatever the launching shell's cwd happens to be,
    # which silently scatters output across the filesystem depending on where the
    # script was invoked from.
    if use_iterative_solver:
        folder_name = str(REPO_ROOT / "Case_III_applied" / simulation_name)  # applied
    else:
        folder_name = str(REPO_ROOT / "Case_III_direct_solver" / simulation_name)
    folder_name_init = folder_name + "_initialization"
    file_name = "example_3"
    prod_label = "with production" if with_production else ", no production"
    title = f"Strain rate = {velocity:.1g} 1/y, {prod_label}"
    # title = title[0].upper() + title[1:]
    return simulation_name, folder_name, folder_name_init, file_name, title


def log_summary(velocity: float, period: float, with_production: bool):
    # Log run configuration
    logger.info("=" * 80)
    logger.info("Starting new simulation with configuration:")
    logger.info(f"  Boundary velocity: {velocity:.2e} m/year")
    logger.info(f"  Production period: {period} years")
    logger.info(f"  With production: {with_production}")
    logger.info("=" * 80)


def _json_default(obj):
    """Best-effort fallback encoder for values json.dump can't handle natively.

    Used for dumping run configuration (which mixes plain Python/numpy values with
    porepy dataclass-like objects such as SolidConstants) for later reference. Never
    raises: anything not otherwise handled falls back to its repr string, so this can
    always be used safely without risking losing the whole dump over one odd value.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        return repr(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    try:
        return repr(obj)
    except Exception:
        return f"<unrepresentable {type(obj).__name__}>"


def save_run_config(folder_name: str, **config) -> None:
    """Dump a best-effort JSON snapshot of a run's configuration for later reference.

    Not a faithful reconstruction recipe (some values, e.g. callables, only survive
    as their repr), but enough to see at a glance what parameters, schedule, and
    material values a given output folder corresponds to.
    """
    path = Path(folder_name) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2, default=_json_default)


def back_compute_friction_coefficient(init_model, sds, floor_value=1e-3, eps=1e-3):
    traction = init_model.evaluate_and_scale(sds, "contact_traction", "-").reshape(
        (3, -1), order="F"
    )
    friction_coeff = (
        np.max(np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :])) + eps
    )
    return np.maximum(friction_coeff, floor_value)


def compute_friction_coefficients_each_fracture(init_model, floor_value=1e-3, eps=1e-3):
    friction_coeffs = {}
    for sd in init_model.mdg.subdomains(dim=2):
        friction_coeffs[str(sd.frac_num)] = back_compute_friction_coefficient(
            init_model, [sd], floor_value=floor_value, eps=eps
        )
    return friction_coeffs


def check_initialization_converged(
    init_model,
    rtol: float = 1e-3,
    variable_names: dict[str, str] | None = None,
) -> dict[str, float]:
    """Pragmatic stopgap check that the last initialization step was small.

    A more rigorous steady-state check is planned for porepy itself; this is a
    cheap interim guard so an unequilibrated initialization doesn't silently
    seed the (expensive) main run and the friction-coefficient back-computation.

    Compares each variable's state at the last converged step (time_step_index=0)
    against the step before it (time_step_index=1) and computes the max-abs
    change relative to the variable's own current max-abs magnitude -- avoids
    needing separately tuned absolute tolerances for pressure (Pa), temperature
    (K), and displacement (m). Contact traction and interface/well variables are
    deliberately excluded: they are Lagrange-multiplier-like quantities without a
    comparable notion of "relative change".

    Parameters:
        init_model: The initialization model, after its ModelRunner.run() call.
        rtol: Shared relative-change tolerance for all checked variables.
        variable_names: Override the default {label: variable_name} mapping
            (pressure/temperature/displacement).

    Returns:
        The computed {label: relative_change} mapping, for logging/inspection.

    Raises:
        RuntimeError: If any checked variable's relative change exceeds rtol.
    """
    if variable_names is None:
        variable_names = {
            "pressure": init_model.pressure_variable,
            "temperature": init_model.temperature_variable,
            "displacement": init_model.displacement_variable,
        }
    eq = init_model.equation_system
    relative_changes = {}
    failures = []
    for label, var_name in variable_names.items():
        current = eq.get_variable_values(variables=[var_name], time_step_index=0)
        previous = eq.get_variable_values(variables=[var_name], time_step_index=1)
        diff = float(np.max(np.abs(current - previous)))
        scale = max(float(np.max(np.abs(current))), 1e-12)
        rel = diff / scale
        relative_changes[label] = rel
        logger.info(f"Initialization last-step relative change in {label}: {rel:.3e}")
        if rel > rtol:
            failures.append(label)
    if failures:
        raise RuntimeError(
            "Initialization does not appear converged: relative change over the "
            f"last time step exceeds rtol={rtol:.1e} for {failures}. "
            f"Relative changes: {relative_changes}. Consider a longer/slower "
            "initialization schedule (time_stepper_init in time_steppers_coso) "
            "or review the artificially high initialization permeability."
        )
    return relative_changes


HETEROGENEOUS_FRICTION_COEFFICIENTS = True
BOUNDARY_VELOCITIES = [
    # 0.0,
    1.0e-6,
    # 2.0e-6,
    # 5.0e-6,
]
WITH_PRODUCTION = [
    True,
    # False,
]
PRODUCTION_PERIODS = [
    # 0.5,
    1.0,
]  # In years
transition_duration = 2 * pp.HOUR
INITIAL_SHUTIN_DURATION = (
    pp.DAY * 2  # short shut-in at t=0 to populate no-flow pressure cache
)
if __name__ == "__main__":
    tp = True
    copy_plots = False
    if LOG_TO_FILE:
        print(f"Logging to file: {log_file}")

    slip_onset_times = {}
    simulation_summaries = []
    for with_prod in WITH_PRODUCTION:
        for velocity in BOUNDARY_VELOCITIES:
            for period in PRODUCTION_PERIODS:
                # Define the time parameters
                dt = 1e3
                production_period = period * pp.YEAR
                domain_size = 8.0e3
                fracture_size = 6e2
                refinement = 1.0 if USE_ITERATIVE_SOLVER else 3.0
                if not USE_CONSTRAINTS:
                    refinement /= 2.2
                cell_size = 10e2 * refinement
                cell_size_fracture = 0.4 * fracture_size * refinement
                cell_size_min = 0.5 * cell_size_fracture
                # was 3. Ask EK about grid with constraints...

                schedule, neumann_intervals, is_transition = create_schedule(
                    production_period,
                    final_time=FINAL_TIME,
                    num_bins_per_interval=NUM_BINS_PER_INTERVAL,
                    transition_duration=transition_duration,
                    initial_shutin_duration=INITIAL_SHUTIN_DURATION,
                    with_production=with_prod,
                )
                simulation_name, folder_name, folder_name_init, file_name, title = (
                    names_from_params(velocity, period, with_prod, USE_ITERATIVE_SOLVER)
                )
                restart_options_init = None
                restart_options_main = None
                restart_dt_first = None
                if RESTART_FROM_CRASH:
                    # Resume the main model from the last checkpoint its own previous
                    # run (in `folder_name`) reached, instead of re-simulating from
                    # t=0. Slicing `schedule`/`is_transition` here (before
                    # `time_steppers_coso` builds the per-interval scheduler) is
                    # required: porepy's scheduler always starts at `schedule[0]`, it
                    # does not pick up from `model.time_data.time`.
                    restart_options_main = latest_restart_options(
                        folder_name, file_name, RESTART_STEPS_BACK
                    )
                    with open(Path(folder_name) / "times.json") as f:
                        source_times = json.load(f)
                    restart_step = len(source_times["time"]) - 1 - RESTART_STEPS_BACK
                    restart_time = source_times["time"][restart_step]
                    # The dt the source run actually used for the step out of this
                    # state, so the resumed run repeats that same step instead of
                    # guessing a first dt (the auto-picked one is half the remaining
                    # schedule interval -- ~1700x too large here, which just burns
                    # retries). Only available when rewinding: the last saved state's
                    # own next step is the one that never completed.
                    restart_dt_first = (
                        source_times["dt"][restart_step + 1]
                        if restart_step + 1 < len(source_times["dt"])
                        else None
                    )
                    # `schedule` only holds *mandatory* checkpoints (BC changes) plus
                    # bin-boundary markers for post-processing -- the adaptive stepper
                    # takes many more accepted intermediate steps between them, so
                    # `restart_time` (an arbitrary accepted step) generally will NOT
                    # be an exact entry of `schedule`. Build a new schedule: the
                    # restart time itself, followed by whichever original checkpoints
                    # still lie ahead of it. Treated as a non-transition point, so the
                    # first resumed interval's dt_start is auto-picked from its
                    # (now shorter) length -- may not exactly match the dt the
                    # original run had adaptively grown to by this point; the
                    # TargetNonlinearIterations retry logic will correct course within
                    # a step or two if it's a poor first guess.
                    still_ahead = schedule > restart_time
                    assert np.any(still_ahead), (
                        f"Restart time {restart_time} is at or past the last "
                        f"schedule checkpoint ({schedule[-1]}) -- nothing left to "
                        "simulate."
                    )
                    logger.info(
                        "RESTART_FROM_CRASH: resuming main model at t=%.6g "
                        "(saved state %d of %d, %d step(s) back; first dt=%s), "
                        "source=%s, %d schedule checkpoints remaining",
                        restart_time,
                        restart_step,
                        len(source_times["time"]) - 1,
                        RESTART_STEPS_BACK,
                        f"{restart_dt_first:.6g}" if restart_dt_first else "auto",
                        folder_name,
                        int(np.sum(still_ahead)),
                    )
                    schedule = np.concatenate(([restart_time], schedule[still_ahead]))
                    is_transition = np.concatenate(
                        ([False], is_transition[still_ahead])
                    )
                    # The initialization model already converged in a previous run;
                    # load its final state directly instead of re-running the
                    # equilibration loop.
                    # The init model is always taken at its final converged state:
                    # RESTART_STEPS_BACK is about re-simulating a known-good *main*
                    # time step, and has no meaning for a model we never re-run.
                    #
                    # Pinned to before the main run's own output: the init folder
                    # holds states from several runs, and a *later* one (whose state
                    # carried a 1e9 pressure error from the unscaled-restart bug
                    # worked around in solution_strategy.py) had overwritten both the
                    # lower-numbered files and times.json. Restoring from that
                    # silently poisons the initialization -- and everything derived
                    # from it -- instead of failing. The init phase that seeded the
                    # run we are restarting necessarily precedes that run's output.
                    restart_options_init = latest_restart_options_state_only(
                        folder_name_init,
                        file_name,
                        before_mtime=restart_options_main["pvd_file"].stat().st_mtime,
                    )
                    # Write the restarted run's new output separately so the original
                    # crash evidence (logs, linear_system_dumps/) is left untouched.
                    folder_name = folder_name + "_restart_debug"
                time_stepper, time_stepper_init = time_steppers_coso(
                    schedule,
                    dt,
                    iter_optimal_range=(10, 15),
                    is_transition=is_transition,
                    dt_start_transition=10,
                    dt_start_first=restart_dt_first if RESTART_FROM_CRASH else None,
                )
                solid_values_local = copy.deepcopy(granodiorite_values)
                solid_values_init = copy.deepcopy(solid_values_local)
                solid_values_init["permeability"] *= (
                    1e2  # More permeable for faster equilibration during initialization.
                )
                if copy_plots:
                    # Copy the plots from previous runs to the new folder for comparison.
                    dest_folder = "figures/Case_III/"
                    source = (
                        folder_name
                        + "_saved_data/well_monitoring/flow_rate_and_displacement_plot.png"
                    )
                    os.makedirs(dest_folder, exist_ok=True)
                    dest = os.path.join(
                        dest_folder, f"{simulation_name}_flow_and_disp.png"
                    )
                    if os.path.exists(source):
                        shutil.copyfile(source, dest)
                        logger.info(f"Copied plot from {source} to {dest}")

                    continue
                log_summary(velocity, period, with_prod)
                # Re-fitted c_T for T range 20-243 degC (Davatzes & Hickman gradient
                # 150/20 mK/m): depth-weighted optimal vs IAPWS-IF97 gives c_T = 9.26e-4
                # K^-1 (RMSE ~9 kg/m3) vs the water table default of 2.07e-4 K^-1 (RMSE
                # ~66 kg/m3).
                water_values = {
                    **pp.fluid_values.water,
                    "thermal_expansion": 7.8745e-04,  # Fitted to Coso T range (see physical_model.py)
                }
                model_params_init = {
                    "fault_plane_dirs": ["point_cloud_clusters"],
                    "fault_extension_config": "fault_extensions.json",
                    "exclude_faults": [
                        # "0000", # Nearly vertical, intersects fracture 5
                        # "0001", Shrunk in config to avoid interference with caprock
                        # and reservoir boundaries. Can be restored once we go to fine
                        # meshes. Now the distance from the fracture to the boundary is
                        # the limiting factor, causing too fine meshes.
                        "0002",
                        # "0003",  #  Intersects injection, extended in config
                        "0004",
                        # "0005",  # Intersects both production wells, extended in
                        # config to ensure intersection.
                        # "0006",  # Intersects 5. Favourable for shut-in?
                        "0007",  # Intersects 5. Favourable for shut-in? Tiny, though.
                        "0008",
                        "0009",
                        "0010",
                        "0011",
                        "0012",
                    ],
                    "well_sheet_names": {
                        "1 Injection well": "68-20RD",
                        "2 Production well": "16A-20",
                        "3 Production well": "16B-20",
                    },
                    "top_boundary_z": 0.0,
                    "read_well_operation_data": False,
                    "plot_title": title,
                    "domain_sizes": np.array([domain_size, domain_size, 3e3]),
                    "material_constants": {
                        "solid": pp.SolidConstants(**solid_values_init),
                        "fluid": pp.FluidComponent(**water_values),
                    },
                    "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
                    "grid_type": "simplex",
                    "meshing_arguments": {
                        "cell_size_boundary": cell_size,
                        "cell_size_fracture": cell_size_fracture,
                        "cell_size_min": cell_size_min,
                    },
                    "file_name": file_name,
                    "data_folder_name": f"{folder_name}_saved_data",
                    "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
                    "use_wells": False,
                    "reference_variable_values": pp.ReferenceVariableValues(
                        temperature=pp.Celsius_to_Kelvin(20),
                        pressure=pp.ATMOSPHERIC_PRESSURE,
                    ),
                    "fracture_file": "coords.txt",
                    "folder_name": folder_name_init,
                    # Relative to "folder_name" -- porepy's initialize_data_saving()
                    # already joins self.params["folder_name"] onto this path, so
                    # prefixing folder_name_init here again double-nests the output.
                    "solver_statistics_file_name": "solver_statistics.json",
                    "initialization": True,
                    "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
                    "fracture_params": {  # Other options are available in the geometry mixin.
                        "fracture_major_axes": np.array(
                            (
                                fracture_size / domain_size,
                                fracture_size / domain_size,
                                fracture_size / domain_size,
                            )
                        ),
                    },
                    "heterogeneous_permeability": True,
                    "darcy_flux_discretization": "tpfa" if tp else "mpfa",
                    "fourier_flux_discretization": "tpfa" if tp else "mpfa",
                    "use_ic_interpolation": True,
                    "boundary_displacement_velocity": velocity / pp.YEAR,
                    "surface_temperature": pp.Celsius_to_Kelvin(20.0),
                }
                if restart_options_init is not None:
                    model_params_init["restart_options"] = restart_options_init
                model_params = copy.deepcopy(model_params_init)
                if with_prod:
                    injection_p = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)
                    production_p = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
                    injection_t = np.full(schedule.shape, pp.Celsius_to_Kelvin(40))
                    # Approximately the expected production from temperature at depth:
                    production_t = np.full(schedule.shape, 475)
                    # Can be refined to have different schedules for each well.
                    for name in MainModel.injection_well_names.fget(None):
                        model_params[f"{name}_pressures"] = injection_p
                        model_params[f"{name}_temperatures"] = injection_t
                    for name in MainModel.production_well_names.fget(None):
                        model_params[f"{name}_pressures"] = production_p
                        model_params[f"{name}_temperatures"] = production_t

                save_run_config(
                    folder_name_init,
                    model_params=model_params_init,
                    schedule=schedule,
                    neumann_intervals=neumann_intervals,
                    is_transition=is_transition,
                    dt=dt,
                )
                init_model = InitializationModel(model_params_init)
                if RESTART_FROM_CRASH:
                    # restart_options_init has no "pvd_file" (see
                    # latest_restart_options_state_only), but write_pvd_and_vtu()
                    # unconditionally reads restart_options["pvd_file"] for its
                    # write-continuation logic whenever restart_options["restart"] is
                    # True -- regardless of which restart read-path was used. We
                    # never intend for this model to export again (its saved history
                    # is already complete and we're not re-running its time loop),
                    # so no-op the one auto-export prepare_simulation() would
                    # otherwise trigger, rather than fight porepy's restart_options
                    # schema for a write we don't want anyway.
                    init_model.save_data_time_step = lambda: None

                t0_init = time.perf_counter()
                nonlinear_solver = set_nonlinear_solver(
                    iterative_linear_solver=USE_ITERATIVE_SOLVER
                )
                init_model.prepare_simulation()
                grids = [init_model.mdg.subdomains(dim=1)[0]]
                energy_balance = init_model.energy_balance_equation(grids)
                variables, _ = (
                    init_model.equation_system.variable_indexer.filter_by_tags(
                        model=init_model,
                        tags=[
                            pp.solvers.VariableTag(
                                name="temperature",
                                defined_on=pp.solvers.OnLowerDimensions(),
                            )
                        ],
                    )
                )
                var_indexer = init_model.equation_system.variable_indexer.construct_restricted_indexer(
                    [variables[0]]
                )
                op = init_model.volume_integral(
                    integrand=pp.ad.Scalar(1), grids=grids, dim=1
                )
                op = init_model.solid_internal_energy(grids)
                op = init_model.fluid_internal_energy(grids)
                ad = init_model.equation_system.evaluate(
                    op, derivative=True, variable_indexer=var_indexer
                )
                if RESTART_FROM_CRASH:
                    # init_model.prepare_simulation() above already restored the
                    # converged end state from restart_options_init; no need to
                    # re-run the (short) equilibration loop. The convergence check
                    # below needs time_step_index=1, which restart does not
                    # populate -- skip it, trusting the original run's validation.
                    t1_init = time.perf_counter()
                    logger.info(
                        "RESTART_FROM_CRASH: init model restored to time=%.6g "
                        "(source=%s); skipped re-running its equilibration loop "
                        "and its convergence check.",
                        init_model.time_data.time,
                        folder_name_init,
                    )
                else:
                    reset_time_io()
                    pp.ModelRunner(
                        init_model,
                        nonlinear_solver=nonlinear_solver,
                        params={"prepare_simulation": False},
                        time_stepper=time_stepper_init,
                    ).run()
                    t1_init = time.perf_counter()
                    check_initialization_converged(init_model)
                # Analyze the initialization results to set friction coefficient
                sds = init_model.mdg.subdomains(dim=2)
                minimum_friction = 0.2  # Physical lower bound for granodiorite SOURCE
                if HETEROGENEOUS_FRICTION_COEFFICIENTS:
                    friction_coeffs = compute_friction_coefficients_each_fracture(
                        init_model, floor_value=minimum_friction, eps=5e-3
                    )
                    model_params["friction_coefficients"] = friction_coeffs
                    logger.info("=" * 80)
                    logger.info(
                        "Determined per-fracture friction coefficients from "
                        f"initialization: {friction_coeffs}"
                    )
                    logger.info("=" * 80)
                    # Used only as a fallback (e.g. by super().friction_coefficient()
                    # for any subdomain not covered by 'friction_coefficients'), so it
                    # should still reflect the back-computed values rather than the raw
                    # material default.
                    solid_values_local["friction_coefficient"] = max(
                        friction_coeffs.values()
                    )
                else:
                    friction_coeff = back_compute_friction_coefficient(
                        init_model, sds, floor_value=minimum_friction
                    )
                    logger.info("=" * 80)
                    logger.info(
                        "Determined friction coefficient from initialization: "
                        + f"{friction_coeff:.4f}"
                    )
                    logger.info("=" * 80)

                    solid_values_local["friction_coefficient"] = friction_coeff
                model_params.update(
                    {
                        "heterogeneous_friction_coefficient": HETEROGENEOUS_FRICTION_COEFFICIENTS,
                        "file_name": file_name,
                        "folder_name": folder_name,
                        "initialization": False,
                        "use_wells": with_prod,
                        "reference_from_initial": True,
                        "material_constants": {
                            "solid": pp.SolidConstants(**solid_values_local),
                            "fluid": pp.FluidComponent(**water_values),
                        },
                        "neumann_intervals": neumann_intervals,
                        "well_transition_duration": transition_duration,
                        # Relative to "folder_name" -- see the matching comment on the
                        # init model_params above.
                        "solver_statistics_file_name": "solver_statistics.json",
                    }
                )
                # model_params was deepcopy'd from model_params_init, which may carry
                # the *initialization* model's restart_options (pointing at
                # folder_name_init) -- the main model needs its own, distinct one.
                if restart_options_main is not None:
                    model_params["restart_options"] = restart_options_main
                else:
                    model_params.pop("restart_options", None)
                model_class = MainModel

                save_run_config(
                    folder_name,
                    model_params=model_params,
                    schedule=schedule,
                    neumann_intervals=neumann_intervals,
                    is_transition=is_transition,
                    dt=dt,
                    friction_coefficient=friction_coeff
                    if not HETEROGENEOUS_FRICTION_COEFFICIENTS
                    else friction_coeffs,
                )
                model = MainModel(model_params)  # Load from initialization from file
                model.initialization_model = init_model
                # solver_params.update(
                #     {
                #         "local_line_search": 1,
                #         "global_line_search": 1,
                #     }
                # )
                t0_main = time.perf_counter()
                nonlinear_solver = set_nonlinear_solver(
                    iterative_linear_solver=USE_ITERATIVE_SOLVER
                )
                reset_time_io()
                # Construct the runner separately from .run(): ModelRunner.__init__
                # seeds model.time_data from the scheduler (schedule included) and
                # only then calls prepare_simulation(), which is where a restart's
                # reset_state_from_file() runs. Preparing the model by hand before
                # this would leave it on SolutionStrategy's placeholder [0, 1]
                # schedule while boundary conditions are built. Splitting the call
                # in two lets us inspect the restored state before any Newton solve.
                model_runner = pp.ModelRunner(
                    model, nonlinear_solver=nonlinear_solver, time_stepper=time_stepper
                )
                if RESTART_FROM_CRASH:
                    logger.info(
                        "RESTART_FROM_CRASH: main model restored to time=%.6g "
                        "(source pvd=%s)",
                        model.time_data.time,
                        restart_options_main["pvd_file"],
                    )
                    if RESTART_DRY_RUN:
                        logger.info(
                            "RESTART_DRY_RUN: stopping before any Newton solve."
                        )
                        continue
                model_runner.run()
                t1_main = time.perf_counter()
                simulation_summaries.append(
                    {
                        "name": simulation_name,
                        "init_steps": init_model.time_data.time_index_successful,
                        "init_wall_time": t1_init - t0_init,
                        "main_steps": model.time_data.time_index_successful,
                        "main_wall_time": t1_main - t0_main,
                    }
                )
                model.save_results()

    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    for s in simulation_summaries:
        total_steps = s["init_steps"] + s["main_steps"]
        total_time = s["init_wall_time"] + s["main_wall_time"]
        print(f"\nSimulation: {s['name']}")
        print(
            f"  Initialization : {s['init_steps']:4d} time steps, "
            f"{s['init_wall_time']:8.1f} s ({s['init_wall_time'] / 60:.1f} min)"
        )
        print(
            f"  Main run       : {s['main_steps']:4d} time steps, "
            f"{s['main_wall_time']:8.1f} s ({s['main_wall_time'] / 60:.1f} min)"
        )
        print(
            f"  Total          : {total_steps:4d} time steps, "
            f"{total_time:8.1f} s ({total_time / 60:.1f} min)"
        )
    print("=" * 80)
