import copy
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import porepy as pp

from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
    SmoothWellTransitions,
)
from geometry import FaultPlaneGeometry
from material_parameters import granodiorite_values
from physical_model import (
    HeterogeneousFrictionCoefficient,
    HeterogeneousPermeabilitySpecification,
    PhysicalModel,
    HagenPoiseuilleWellPermeability,
    CosoBackgroundValues,
)
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
from porepy.models.contact_mechanics import (
    RadialReturnTangentialContactMechanicsEquation,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

from exporting import CosoExporter, GeometryExporting
from initial_conditions import CopyInitialCondition
from nl_params import set_nonlinear_solver
from solution_strategy import SolutionStrategy
from time_stepping import reset_time_io, time_steppers_coso
from wells import _WellDataBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_ROOT = Path(__file__).resolve().parent
FINAL_TIME = 2 * pp.YEAR
SHUT_IN_DURATION = 3 * pp.DAY
NUM_BINS_PER_INTERVAL = 3
SOLVER_SCHEME = "monolithic"
TIME_STEPPER_PARAMS = {
    "iter_optimal_range": (10, 15),
    "iter_relax_factors": (0.6, 2.5),
    "recomp_factor": 0.2,
    "max_attempts": 10,
    "dt_start_transition": 10,
    "dt_max_divisor": 2.0,
    "dt_min": 1e-2,
}

# Single run: strain rate [m/year], production period [years], transition [s].
VELOCITY = 1.0e-6
PERIOD = 1.0
TRANSITION_DURATION = 2 * pp.HOUR
INITIAL_SHUTIN_DURATION = 2 * pp.DAY
MINIMUM_FRICTION = 0.2
EXCLUDE_FAULTS = ["0002", "0004", "0006", "0007", "0008", "0009", "0010", "0011", "0012"]


class BaseModel(
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
        # Index 1 (the step before the latest) is read by
        # check_initialization_converged() and by CosoExporter.collect_data() for the
        # fracture displacement-jump increment.
        return np.array([0, 1])

    def create_plot_title(self) -> str:
        return self.params["plot_title"]


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


if __name__ == "__main__":
    dt = 1e3
    production_period = PERIOD * pp.YEAR
    domain_size = 8.0e3
    fracture_size = 6e2
    refinement = 1.0 / 2.2 * 0.7  # iterative solver, no constraints, fine run
    cell_size = 10e2 * refinement
    cell_size_fracture = 0.4 * fracture_size * refinement
    cell_size_min = 0.5 * cell_size_fracture

    schedule, neumann_intervals, is_transition = create_schedule(
        production_period,
        final_time=FINAL_TIME,
        num_bins_per_interval=NUM_BINS_PER_INTERVAL,
        transition_duration=TRANSITION_DURATION,
        initial_shutin_duration=INITIAL_SHUTIN_DURATION,
        with_production=True,
    )
    folder_name = str(REPO_ROOT / f"Case_III_{SOLVER_SCHEME}" / "single_run")
    folder_name_init = folder_name + "_initialization"
    file_name = "example_3"
    title = f"Strain rate = {VELOCITY:.1g} 1/y, with production"

    time_stepper, time_stepper_init = time_steppers_coso(
        schedule, dt, is_transition=is_transition, dt_start_first=None,
        **TIME_STEPPER_PARAMS,
    )
    solid_values_local = copy.deepcopy(granodiorite_values)
    solid_values_init = copy.deepcopy(solid_values_local)
    solid_values_init["permeability"] *= 1e2  # Faster equilibration in initialization.

    # Re-fitted thermal expansion for the Coso T range (see physical_model.py).
    water_values = {**pp.fluid_values.water, "thermal_expansion": 7.8745e-04}
    model_params_init = {
        "fault_plane_dirs": ["point_cloud_clusters"],
        "fault_extension_config": "fault_extensions.json",
        "exclude_faults": EXCLUDE_FAULTS,
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
        "adaptive_indicator_scaling": 1,
        "use_wells": False,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=pp.Celsius_to_Kelvin(20),
            pressure=pp.ATMOSPHERIC_PRESSURE,
        ),
        "fracture_file": "coords.txt",
        "folder_name": folder_name_init,
        "solver_statistics_file_name": "solver_statistics.json",
        "initialization": True,
        "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
        "fracture_params": {
            "fracture_major_axes": np.full(3, fracture_size / domain_size),
        },
        "heterogeneous_permeability": True,
        "darcy_flux_discretization": "tpfa",
        "fourier_flux_discretization": "tpfa",
        "use_ic_interpolation": True,
        "boundary_displacement_velocity": VELOCITY / pp.YEAR,
        "surface_temperature": pp.Celsius_to_Kelvin(20.0),
    }
    model_params = copy.deepcopy(model_params_init)
    model_params_init["time_manager"] = time_stepper_init.time_manager
    model_params["time_manager"] = time_stepper.time_manager

    injection_p = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)
    production_p = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
    injection_t = np.full(schedule.shape, pp.Celsius_to_Kelvin(40))
    production_t = np.full(schedule.shape, 475)  # Expected production T at depth.
    for name in MainModel.injection_well_names.fget(None):
        model_params[f"{name}_pressures"] = injection_p
        model_params[f"{name}_temperatures"] = injection_t
    for name in MainModel.production_well_names.fget(None):
        model_params[f"{name}_pressures"] = production_p
        model_params[f"{name}_temperatures"] = production_t

    # Initialization run
    init_model = InitializationModel(model_params_init)
    nonlinear_solver = set_nonlinear_solver(
        iterative_linear_solver=True, scheme=SOLVER_SCHEME
    )
    init_model.prepare_simulation()
    reset_time_io()
    t0_init = time.perf_counter()
    pp.ModelRunner(
        init_model,
        nonlinear_solver=nonlinear_solver,
        params={"prepare_simulation": False},
        time_stepper=time_stepper_init,
    ).run()
    t1_init = time.perf_counter()
    check_initialization_converged(init_model)

    # Per-fracture friction coefficients back-computed from the initialization.
    friction_coeffs = compute_friction_coefficients_each_fracture(
        init_model, floor_value=MINIMUM_FRICTION, eps=5e-3
    )
    model_params["friction_coefficients"] = friction_coeffs
    logger.info(f"Friction coefficients from initialization: {friction_coeffs}")
    # Fallback for subdomains not covered by 'friction_coefficients'.
    solid_values_local["friction_coefficient"] = max(friction_coeffs.values())
    model_params.update(
        {
            "heterogeneous_friction_coefficient": True,
            "folder_name": folder_name,
            "initialization": False,
            "use_wells": True,
            "reference_from_initial": True,
            "material_constants": {
                "solid": pp.SolidConstants(**solid_values_local),
                "fluid": pp.FluidComponent(**water_values),
            },
            "neumann_intervals": neumann_intervals,
            "well_transition_duration": TRANSITION_DURATION,
        }
    )

    # Main run
    model = MainModel(model_params)
    model.initialization_model = init_model
    nonlinear_solver = set_nonlinear_solver(
        iterative_linear_solver=True, scheme=SOLVER_SCHEME
    )
    reset_time_io()
    t0_main = time.perf_counter()
    pp.ModelRunner(
        model, nonlinear_solver=nonlinear_solver, time_stepper=time_stepper
    ).run()
    t1_main = time.perf_counter()
    model.save_results()
    print(
        f"Init: {init_model.time_manager.time_index} steps, {t1_init - t0_init:.1f} s; "
        f"main: {model.time_manager.time_index} steps, {t1_main - t0_main:.1f} s"
    )
