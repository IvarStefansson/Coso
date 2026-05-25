import re
import shutil
from typing import Sequence

import numpy as np
import porepy as pp
from porepy.applications.test_utils.models import add_mixin

from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
)
from geometry import ConceptualGeometry
from material_parameters import granodiorite_values
from physical_model import HeterogeneousPermeabilitySpecification, PhysicalModel

try:
    import pp_solvers
except ImportError:
    pp_solvers = None

import copy
import logging
import os
import sys
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
from porepy.viz.data_saving_model_mixin import (
    FractureDeformationExporting,
    IterationExporting,
    ResidualExporting,
)

from exporting import CosoExporter, GeometryExporting, summarize_slip_onset_times
from initial_conditions import CopyInitialCondition
from solution_strategy import SolutionStrategy
from solver_configurations import linear_solver_params, solver_params
from wells import WellDataConceptual

# Suppress multiprocessing semaphore warnings (common when debugging)
# warnings.filterwarnings("ignore", ".*semaphore.*resource_tracker.*")

LOG_TO_FILE = False
log_dir = "example_4_logs"


if LOG_TO_FILE:
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_example_4_{timestamp}.log")

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
    # IterationExporting,
    # ResidualExporting,
    GeometryExporting,
    CosoExporter,
    ConceptualGeometry,
    # pp.poromechanics.TpsaPoromechanicsMixin,
    # diff_tpfa.DarcysLawAdEverywhere,
    HeterogeneousPermeabilitySpecification,
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

        Extracts simulation metadata from folder name (strike angle, fracture index, well status)
        and formats it as a human-readable title. Fracture indices are incremented by 1 for
        display purposes (0-indexed internally, 1-indexed for display).

        Returns:
            str | None: Formatted plot title with well status, strike angle, and fracture info.

        Example:
            Input folder_name: "case_II_with_wells_strike_35_tilted_fracture_0"
            Output: "with wells strike 35 tilted fracture, Tilted, Strike: 1"
        """
        fn = self.params["plot_title"]
        return fn


# inds = slice(0, 1) if isinstance(PhysicalModel, pp.Thermoporomechanics) else slice(0)
# for method, dof in zip(diff_tpfa.methods[inds], diff_tpfa.dofs[inds]):
#     diff_tpfa.override_methods(BaseModel, method, dof, diff_tpfa.sort_criterion)


class InitializationModel(
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    # pp.constitutive_laws.ConstantPorosity,
    BaseModel,
):
    """Initialization model for the Coso geothermal reservoir."""


class MainModel(
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
    final_time: float = 6 * pp.YEAR,
    num_bins_per_interval: int | None = None,
) -> tuple[np.ndarray, Sequence[tuple[float, float]]]:
    """Create a time schedule for the simulation based on production period and time step.

    Generates a time schedule that includes both the production period and additional
    time steps to ensure proper resolution of the simulation. The schedule starts at 0,
    includes the production period, and extends beyond it with additional steps
    determined by the time step.

    Parameters:
        production_period: The total production period in years.
        shut_in_duration: The duration of shut-in periods in years (default is 2 days).
        final_time: The total simulation time in years (default is 4 years).
        num_bins_per_interval: If given, bin boundaries for both production and shut-in
            sub-intervals are inserted into the schedule so that simulation outputs
            align exactly with bin edges used in post-processing plots.

    Returns:
        tuple: A tuple containing:
            - schedule (np.ndarray): An array of time points for the simulation.
            - neumann_intervals (list of tuples): A list of time intervals where Neumann
              boundary conditions are applied.
    """
    # Calculate the number of production periods needed to cover the final time
    n = int(np.ceil(final_time / production_period))
    # Create a schedule with production periods and shut-in durations.
    schedule = np.hstack(
        [
            [i * production_period - shut_in_duration, i * production_period]
            for i in np.arange(1, n + 1)
        ]
    )
    # Create a list of Neumann intervals corresponding to the shut-in periods.
    neumann_intervals = [
        (i * production_period - shut_in_duration, i * production_period)
        for i in np.arange(1, n + 1)
    ]
    # Add the initial time step at 0 to the schedule.
    schedule = np.insert(schedule, 0, 0.0)

    if num_bins_per_interval is not None:
        # Insert bin boundaries for each production and each shut-in sub-interval so
        # that simulation checkpoints align exactly with the bin edges used for
        # plotting slip rates and production rates.
        bin_times = []
        for i in np.arange(1, n + 1):
            prod_start = (i - 1) * production_period
            prod_end = i * production_period - shut_in_duration
            bin_times.append(
                np.linspace(prod_start, prod_end, num_bins_per_interval + 1)
            )
            shut_start = prod_end
            shut_end = i * production_period
            bin_times.append(
                np.linspace(shut_start, shut_end, num_bins_per_interval + 1)
            )
        schedule = np.unique(np.concatenate([schedule] + bin_times))

    return schedule, neumann_intervals


def names_from_params(
    velocity: float, period: float, well_name: str, thermal_expansion: float
):
    simulation_name = (
        f"velocity_{velocity:.0e}_period_{period:.0e}_"
        + well_name
        + f"_thermal_exp_{thermal_expansion:.0e}"
    )
    folder_name = "Case_I/" + simulation_name
    folder_name_init = folder_name + "_initialization"
    file_name = "example_4"
    title = (
        well_name.replace("_", " ") + f", Boundary velocity = {velocity:.1e} m/y, "
        # + f"Thermal exp. = {thermal_expansion:.1e} 1/K"
    )
    title = title[0].upper() + title[1:]
    return simulation_name, folder_name, folder_name_init, file_name, title


def time_managers(schedule: np.ndarray, dt: float, production_period: float):
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt,
        dt_min_max=(1e-2, max(dt, production_period / 5)),
        iter_max=20,
        iter_optimal_range=(5, 12),
        iter_relax_factors=(0.6, 2.0),
        recomp_factor=0.3,
        recomp_max=10,
    )
    dt_init = 5e9  # * pp.YEAR
    time_manager_init = pp.TimeManager(
        [0, 5 * dt_init],
        dt_init=dt_init,
        dt_min_max=(1, 2 * dt_init),
        iter_max=20,
        iter_optimal_range=(5, 12),
        constant_dt=False,
    )
    return time_manager, time_manager_init


def log_summary(
    well_name,
    thermal_expansion,
    velocity,
    period,
):
    # Log run configuration
    logger.info("=" * 80)
    logger.info("Starting new simulation with configuration:")
    logger.info(f"  Well configuration: {well_name}")
    logger.info(f"  Boundary velocity: {velocity:.2e} m/year")
    logger.info(f"  Production period: {period} years")
    logger.info(f"  Thermal expansion coefficient: {thermal_expansion:.2e} 1/K")
    logger.info("=" * 80)


use_iterative_solver = False
if use_iterative_solver and "pp_solvers" not in sys.modules:
    raise ImportError(
        "pp_solvers module is required for iterative solvers. Please install pp_solvers"
        + " or set use_iterative_solver to False."
    )

cases = (("long_well", 1e3), ("short_well", 2e3))
boundary_velocities = [
    0.0,
    1.0e-6,
    2.0e-6,
    # 5.0e-6,
]
thermal_expansions = [
    granodiorite_values["thermal_expansion"],
    0.0,
]

production_periods = [
    0.5,
    1.0,
]  # In years
SHUT_IN_DURATION = 3 * pp.DAY
FINAL_TIME = 6 * pp.YEAR
PRODUCTION_WELL = "2 Production well"
if __name__ == "__main__":
    tp = True
    copy_plots = False
    if LOG_TO_FILE:
        print(f"Logging to file: {log_file}")
        clean_logs = False
        if clean_logs:
            for f in os.listdir(log_dir):
                if re.match(r"run_example_4_\d{8}_\d{6}\.log", f):
                    os.remove(os.path.join(log_dir, f))

    slip_onset_times = {}
    for velocity in boundary_velocities:
        for well_name, well_endpoint in cases:
            for period in production_periods:
                for thermal_expansion in thermal_expansions:
                    # Define the time parameters
                    dt = 1e2
                    production_period = period * pp.YEAR
                    domain_size = 4.0e3
                    fracture_size = 5e2
                    refinement = 0.3 if use_iterative_solver else 1.0
                    cell_size = 11e2 * refinement
                    cell_size_fracture = 0.6 * fracture_size * refinement

                    schedule, neumann_intervals = create_schedule(
                        production_period, num_bins_per_interval=5
                    )
                    time_manager, time_manager_init = time_managers(
                        schedule, dt, production_period
                    )
                    solid_values_local = copy.deepcopy(granodiorite_values)
                    solid_values_local["thermal_expansion"] = thermal_expansion
                    solid_values_init = copy.deepcopy(solid_values_local)
                    simulation_name, folder_name, folder_name_init, file_name, title = (
                        names_from_params(
                            velocity, period, well_name, thermal_expansion
                        )
                    )
                    if copy_plots:
                        # Copy the plots from previous runs to the new folder for comparison.
                        dest_folder = "figures/Case I/"
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
                    log_summary(
                        well_name,
                        thermal_expansion,
                        velocity,
                        period,
                    )
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
                            #     pressure=pp.BAR,
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
                    }
                    model_params = copy.deepcopy(model_params_init)
                    # Reduce target pressure in favour of displacement BC as driving force?
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

                    # Create the model

                    initialization_class = InitializationModel
                    if use_iterative_solver:
                        initialization_class = add_mixin(
                            pp_solvers.IterativeSolverMixin, InitializationModel
                        )
                        model_params_init["linear_solver"] = {
                            "preconditioner_factory": pp_solvers.thm_factory
                        }

                        model_params_init["linear_solver"]["options"] = (
                            linear_solver_params
                        )

                    init_model = initialization_class(model_params_init)

                    pp.ModelRunner(init_model, solver_params).run()
                    # Analyze the initialization results to set friction coefficient
                    sds = init_model.mdg.subdomains(dim=2)
                    traction = init_model.evaluate_and_scale(
                        sds, "contact_traction", "-"
                    ).reshape((3, -1), order="F")
                    friction_coeff = (
                        np.max(
                            np.linalg.norm(traction[:-1], axis=0)
                            / np.abs(traction[-1, :])
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
                            "use_wells": True,
                            "reference_from_initial": True,
                            "material_constants": {
                                "solid": pp.SolidConstants(**solid_values_local),
                                "fluid": pp.FluidComponent(**pp.fluid_values.water),
                            },
                            "neumann_intervals": neumann_intervals,
                            "production_well_y_endpoint": well_endpoint,
                        }
                    )
                    model_class = MainModel
                    if use_iterative_solver:
                        model_class = add_mixin(
                            pp_solvers.IterativeSolverMixin, MainModel
                        )
                        model_params["linear_solver"] = {
                            "preconditioner_factory": pp_solvers.thm_factory
                        }
                        model_params["linear_solver"]["options"] = linear_solver_params

                    model = model_class(
                        model_params
                    )  # Load from initialization from file
                    model.initialization_model = init_model
                    solver_params.update(
                        {
                            "local_line_search": 1,
                            "global_line_search": 1,
                            # "nl_convergence_res_atol": 1e0,
                        }
                    )
                    pp.ModelRunner(model, solver_params).run()
                    slip_onset_times[simulation_name] = model.slip_onset_times()
