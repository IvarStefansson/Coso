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
    ConstraintsCapcrockAndReservoirDepth,
    FaultPlaneGeometry,
)
from porepy.models.contact_mechanics import (
    RadialReturnTangentialContactMechanicsEquation,
)
from material_parameters import granodiorite_values
from physical_model import (
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
from nl_params import solver_params, set_nonlinear_solver
from wells import _WellDataBase

LOG_TO_FILE = False
log_dir = "example_3_logs"
FINAL_TIME = 2 * pp.YEAR
SHUT_IN_DURATION = 3 * pp.DAY
PRODUCTION_WELL = "2 Production well"
NUM_BINS_PER_INTERVAL = 3
USE_ITERATIVE_SOLVER = True
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


class BaseModel(
    FractureDeformationExporting,
    GeometryExporting,
    CosoExporter,
    # ConstraintsCapcrockAndReservoirDepth,
    FaultPlaneGeometry,
    HagenPoiseuilleWellPermeability,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    _WellDataBase,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    # RadialReturnTangentialContactMechanicsEquation,
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

        """
        fn = self.params["plot_title"]
        return fn


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


def names_from_params(
    velocity: float, period: float, with_production: bool, use_iterative_solver: bool
):
    prod_suffix = "" if with_production else "_no_production"
    simulation_name = f"velocity_{velocity:.0e}_period_{period:.0e}{prod_suffix}"
    # While debugging iterative solvers, distinguish to avoid overwriting the production
    # runs with direct solvers.
    if use_iterative_solver:
        folder_name = "Case_III_applied/" + simulation_name  # applied
    else:
        folder_name = "Case_III_direct_solver/" + simulation_name
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


boundary_velocities = [
    0.0,
    1.0e-6,
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
    pp.DAY * 2  # short shut-in at t=0 to populate no-flow pressure cache
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
                domain_size = 8.0e3
                fracture_size = 6e2
                refinement = 2.9 if USE_ITERATIVE_SOLVER else 1.5
                cell_size = 10e2 * refinement
                cell_size_fracture = 0.4 * fracture_size * refinement

                schedule, neumann_intervals = create_schedule(
                    production_period,
                    final_time=FINAL_TIME,
                    num_bins_per_interval=NUM_BINS_PER_INTERVAL,
                    transition_duration=transition_duration,
                    initial_shutin_duration=INITIAL_SHUTIN_DURATION,
                    with_production=with_prod,
                )
                time_stepper, time_stepper_init = time_steppers_coso(
                    schedule, dt, iter_optimal_range=(10, 15)
                )
                solid_values_local = copy.deepcopy(granodiorite_values)
                solid_values_init = copy.deepcopy(solid_values_local)
                solid_values_init["permeability"] *= (
                    1e3  # More permeable for faster equilibration during initialization.
                )
                simulation_name, folder_name, folder_name_init, file_name, title = (
                    names_from_params(velocity, period, with_prod, USE_ITERATIVE_SOLVER)
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
                        "0002",
                        # "0003",  #  Intersects injection, extended in config
                        "0004",
                        "0006",  #
                        "0007",
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
                        "cell_size": cell_size,
                        "cell_size_fracture": cell_size_fracture,
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
                    "heterogeneous_permeability": False,
                    "darcy_flux_discretization": "tpfa" if tp else "mpfa",
                    "fourier_flux_discretization": "tpfa" if tp else "mpfa",
                    "use_ic_interpolation": True,
                    "boundary_displacement_velocity": velocity / pp.YEAR,
                    "surface_temperature": pp.Celsius_to_Kelvin(20.0),
                }
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

                init_model = InitializationModel(model_params_init)

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
                reset_time_io()
                pp.ModelRunner(
                    init_model,
                    nonlinear_solver=nonlinear_solver,
                    params={"prepare_simulation": False},
                    time_stepper=time_stepper_init,
                ).run()
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
                    }
                )
                model_class = MainModel

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
                pp.ModelRunner(
                    model, nonlinear_solver=nonlinear_solver, time_stepper=time_stepper
                ).run()
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
