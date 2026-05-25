import shutil
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
    ConceptualGeometryTwoFractures,
    ConstraintsCapcrockAndReservoirDepth,
)

from material_parameters import granodiorite_values
from physical_model import (
    HeterogeneousPermeabilitySpecification,
    PhysicalModel,
    HagenPoiseuilleWellPermeability,
)
from diff_tpfa import DarcysLawAdEverywhere
from time_manager import CosoTimeManager as TimeManager

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
from solver_configurations import linear_solver_params, solver_params
from wells import WellDataConceptual

LOG_TO_FILE = False
log_dir = "example_2_logs"
FINAL_TIME = 2 * pp.YEAR
SHUT_IN_DURATION = 3 * pp.DAY
PRODUCTION_WELL = "2 Production well"

if LOG_TO_FILE:
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_example_2_{timestamp}.log")

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


class BaseModel(
    FractureDeformationExporting,
    GeometryExporting,
    CosoExporter,
    # ConstraintsCapcrockAndReservoirDepth,
    ConceptualGeometryTwoFractures,
    HeterogeneousPermeabilitySpecification,
    HagenPoiseuilleWellPermeability,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    WellDataConceptual,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    PhysicalModel,
):
    """Model for the Coso geothermal reservoir."""

    def create_plot_title(self) -> str:
        """Generate a formatted plot title from folder name and simulation parameters.

        Extracts simulation metadata from folder name (strike angle, fracture index,
        well status) and formats it as a human-readable title. Fracture indices are
        incremented by 1 for display purposes (0-indexed internally, 1-indexed for
        display).

        Returns:
            str | None: Formatted plot title with well status, strike angle, and
            fracture info.

        Example:
            Input folder_name: "case_II_with_wells_strike_35_tilted_fracture_0"
            Output: "with wells strike 35 tilted fracture, Tilted, Strike: 1"
        """
        fn = self.params["plot_title"]
        return fn


class InitializationModel(
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    BaseModel,
):
    """Initialization model for the Coso geothermal reservoir."""


class MainModel(
    SmoothWellTransitions,
    NeumannWellBCsFromSchedule,
    CopyInitialCondition,
    WellBoundaryConditions,
    CosoBoundaryConditionsDisplacement,
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
) -> tuple[np.ndarray, Sequence[tuple[float, float]]]:
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

    return schedule, neumann_intervals


def names_from_params(velocity: float, period: float, with_production: bool):
    prod_suffix = "" if with_production else "_no_production"
    simulation_name = f"velocity_{velocity:.0e}_period_{period:.0e}{prod_suffix}"
    folder_name = "Case_II/" + simulation_name
    folder_name_init = folder_name + "_initialization"
    file_name = "example_2"
    prod_label = "with production" if with_production else ", no production"
    title = f"Strain rate = {velocity:.1g} m/y, {prod_label}"
    # title = title[0].upper() + title[1:]
    return simulation_name, folder_name, folder_name_init, file_name, title


def time_managers(schedule: np.ndarray, dt: float, production_period: float):
    # Target reduction factor per interval boundary crossing.  This is used to compute
    # the maximum time step size for each interval as a fraction of the interval length.
    # Can be tailored based on type/length of interval.
    ks = np.full(schedule.shape[0] - 1, 2.0)
    dt_min_max = [(1e-2, dt_max / k) for dt_max, k in zip(np.diff(schedule), ks)]
    time_manager = TimeManager(
        schedule=schedule,
        dt_init=dt,
        dt_min_max=dt_min_max,
        iter_max=20,
        iter_optimal_range=(5, 12),
        iter_relax_factors=(0.6, 2.5),
        recomp_factor=0.2,
        recomp_max=10,
        rtol=1e-20,  # Low rtol due to large time values.
        atol=1e-5,
    )
    # 5e9 s ≈ 158 years, which is safely below the production period for all cases.
    dt_init = 5e9
    time_manager_init = TimeManager(
        schedule=[0, 3 * dt_init],
        dt_init=dt_init,
        dt_min_max=(1, 2 * dt_init),
        iter_max=20,
        iter_optimal_range=(5, 12),
        constant_dt=False,
    )
    return time_manager, time_manager_init


def log_summary(velocity: float, period: float, with_production: bool):
    # Log run configuration
    logger.info("=" * 80)
    logger.info("Starting new simulation with configuration:")
    logger.info(f"  Boundary velocity: {velocity:.2e} m/year")
    logger.info(f"  Production period: {period} years")
    logger.info(f"  With production: {with_production}")
    logger.info("=" * 80)


NUM_BINS_PER_INTERVAL = 4
use_iterative_solver = True
if use_iterative_solver and "pp_solvers" not in sys.modules:
    raise ImportError(
        "pp_solvers module is required for iterative solvers. Please install pp_solvers"
        + " or set use_iterative_solver to False."
    )

boundary_velocities = [
    0.0,
    # 1.0e-6,
    # 2.0e-6,
    # 5.0e-6,
]
with_production = [
    True,
    # False,
]
production_periods = [
    # 0.5,
    1.0,
]  # In years
transition_duration = 2 * pp.HOUR
INITIAL_SHUTIN_DURATION = (
    pp.DAY / 2  # short shut-in at t=0 to populate no-flow pressure cache
)
if __name__ == "__main__":
    tp = True
    copy_plots = False
    if LOG_TO_FILE:
        print(f"Logging to file: {log_file}")

    slip_onset_times = {}
    simulation_summaries = []
    for with_prod in with_production:
        for velocity in boundary_velocities:
            for period in production_periods:
                # Define the time parameters
                dt = 1e3
                production_period = period * pp.YEAR
                domain_size = 4.0e3
                fracture_size = 5e2
                refinement = 0.65 if use_iterative_solver else 0.65
                cell_size = 10e2 * refinement
                cell_size_fracture = 0.6 * fracture_size * refinement

                schedule, neumann_intervals = create_schedule(
                    production_period,
                    final_time=FINAL_TIME,
                    num_bins_per_interval=NUM_BINS_PER_INTERVAL,
                    transition_duration=transition_duration,
                    initial_shutin_duration=INITIAL_SHUTIN_DURATION,
                    with_production=with_prod,
                )
                time_manager, time_manager_init = time_managers(
                    schedule, dt, production_period
                )
                solid_values_local = copy.deepcopy(granodiorite_values)
                solid_values_init = copy.deepcopy(solid_values_local)
                simulation_name, folder_name, folder_name_init, file_name, title = (
                    names_from_params(velocity, period, with_prod)
                )
                if copy_plots:
                    # Copy the plots from previous runs to the new folder for comparison.
                    dest_folder = "figures/Case_II/"
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
                model_params_init = {
                    "plot_title": title,
                    "domain_sizes": np.full(3, domain_size),
                    "material_constants": {
                        "solid": pp.SolidConstants(**solid_values_init),
                        "fluid": pp.FluidComponent(**pp.fluid_values.water),
                    },
                    "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
                    "time_manager": time_manager_init,
                    "grid_type": "simplex",
                    "meshing_arguments": {
                        "cell_size": cell_size,
                        "cell_size_fracture": cell_size_fracture,
                    },
                    "file_name": file_name,
                    "data_folder_name": f"{folder_name}_saved_data",
                    "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
                    "use_wells": False,
                    "reference_variable_values": pp.ReferenceVariableValues(
                        temperature=pp.Celsius_to_Kelvin(50)
                    ),
                    "thermal_gradient": 7e-2,
                    "fracture_file": "coords.txt",
                    "folder_name": folder_name_init,
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
                    "surface_temperature": pp.Celsius_to_Kelvin(30.0),
                }
                model_params = copy.deepcopy(model_params_init)
                if with_prod:
                    injection_p = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)
                    production_p = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
                    injection_t = np.full(schedule.shape, pp.Celsius_to_Kelvin(10))
                    production_t = np.full(schedule.shape, 373.15)
                    # Can be refined to have different schedules for each well.
                    for name in MainModel.injection_well_names.fget(None):
                        model_params[f"{name}_pressures"] = injection_p
                        model_params[f"{name}_temperatures"] = injection_t
                    for name in MainModel.production_well_names.fget(None):
                        model_params[f"{name}_pressures"] = production_p
                        model_params[f"{name}_temperatures"] = production_t

                initialization_class = InitializationModel
                if use_iterative_solver:
                    initialization_class = add_mixin(
                        pp_solvers.IterativeSolverMixin, InitializationModel
                    )
                    model_params_init["linear_solver"] = {
                        "preconditioner_factory": pp_solvers.thm_factory
                    }

                    model_params_init["linear_solver"]["options"] = linear_solver_params

                init_model = initialization_class(model_params_init)

                t0_init = time.perf_counter()
                pp.run_time_dependent_model(init_model, solver_params)
                t1_init = time.perf_counter()
                # Analyze the initialization results to set friction coefficient
                sds = init_model.mdg.subdomains(dim=2)
                traction = init_model.evaluate_and_scale(
                    sds, "contact_traction", "-"
                ).reshape((3, -1), order="F")
                friction_coeff = (
                    np.max(
                        np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :])
                    )
                    + 0.001
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
                        "time_manager": time_manager,
                        "file_name": file_name,
                        "folder_name": folder_name,
                        "initialization": False,
                        "use_wells": with_prod,
                        "reference_from_initial": True,
                        "material_constants": {
                            "solid": pp.SolidConstants(**solid_values_local),
                            "fluid": pp.FluidComponent(**pp.fluid_values.water),
                        },
                        "neumann_intervals": neumann_intervals,
                        "well_transition_duration": transition_duration,
                        "production_well_y_endpoint": 1e3,
                    }
                )
                model_class = MainModel
                if use_iterative_solver:
                    model_class = add_mixin(pp_solvers.IterativeSolverMixin, MainModel)
                    model_params["linear_solver"] = {
                        "preconditioner_factory": pp_solvers.thm_factory
                    }
                    model_params["linear_solver"]["options"] = linear_solver_params

                model = model_class(model_params)  # Load from initialization from file
                model.initialization_model = init_model
                solver_params.update(
                    {
                        "local_line_search": 1,
                        "global_line_search": 1,
                    }
                )
                t0_main = time.perf_counter()
                pp.run_time_dependent_model(model, solver_params)
                t1_main = time.perf_counter()
                slip_onset_times[simulation_name] = model.slip_onset_times()
                simulation_summaries.append(
                    {
                        "name": simulation_name,
                        "init_steps": init_model.time_manager.time_index,
                        "init_wall_time": t1_init - t0_init,
                        "main_steps": model.time_manager.time_index,
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
