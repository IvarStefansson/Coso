import re

import porepy as pp
import numpy as np
from geometry import ConceptualGeometry
from material_parameters import granodiorite_values

from physical_model import PhysicalModel, HeterogeneousPermeabilitySpecification
from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
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
from porepy.viz.data_saving_model_mixin import (
    IterationExporting,
    FractureDeformationExporting,
    ResidualExporting,
)
from solution_strategy import SolutionStrategy
from initial_conditions import CopyInitialCondition
from exporting import CosoExporter, GeometryExporting, summarize_slip_onset_times
from wells import WellDataConceptual
from porepy.numerics.nonlinear import line_search
import diff_tpfa
import logging
import sys
import copy
import os
from datetime import datetime

LOG_TO_FILE = True
log_dir = "example_4_logs"

if LOG_TO_FILE:
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamped log filename
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
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstraintLineSearch,
    line_search.SplineInterpolationLineSearch,
    line_search.LineSearchNewtonSolver,
):
    pass


def create_schedule(
    production_period: float, dt: float, shut_in_duration: float = pp.DAY
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Create a time schedule for the simulation based on production period and time step.

    Generates a time schedule that includes both the production period and additional
    time steps to ensure proper resolution of the simulation. The schedule starts at 0,
    includes the production period, and extends beyond it with additional steps
    determined by the time step.

    Parameters:
        production_period: The total production period in years.
        dt: The desired time step size in years.
        shut_in_duration: The duration of shut-in periods in years (default is 1 day).

    Returns:
        tuple: A tuple containing:
            - schedule (np.ndarray): An array of time points for the simulation.
            - neumann_intervals (list of tuples): A list of time intervals where Neumann
              boundary conditions are applied.
    """
    schedule = np.array(
        [
            0,
            2 * dt,  # Initial time steps with closed wells
            # pp.DAY,  # Ramp up to operation
            production_period - shut_in_duration,  # First shut-in
            production_period,  # Restart operation
            2 * production_period - shut_in_duration,  # Second shut-in
            2 * production_period,  # End second shut-in and end of simulation
            3 * production_period - shut_in_duration,  # Second shut-in
            3 * production_period,  # End second shut-in and end of simulation
            4 * production_period - shut_in_duration,  # Second shut-in
            4 * production_period,  # End second shut-in and end of simulation
        ]
    )
    # Lazy way to determine if we have the initial closed well steps or not. TODO:
    # Remove hack.
    offset = 2 if schedule[1] < 2 * pp.DAY else 1
    neumann_intervals = [
        (schedule[2 * i + offset], schedule[2 * i + offset + 1])
        for i in range(schedule.size // 2 - 1)
    ]
    return schedule, neumann_intervals


if __name__ == "__main__":
    tp = True
    if LOG_TO_FILE:
        print("Logging to file: " + log_dir + "/" + log_file)
        clean_logs = False
        if clean_logs:
            for f in os.listdir(log_dir):
                if re.match(r"run_example_4_\d{8}_\d{6}\.log", f):
                    os.remove(os.path.join(log_dir, f))
    cases = (("long_well", 1e3), ("short_well", 2e3))
    boundary_velocities = [0.0, 2.0e-6, 5.0e-6]

    # boundary_velocities = [5.0, 15.0]
    production_periods = [0.5, 1.0]  # In years
    slip_onset_times = {}
    for velocity in boundary_velocities:
        for well_name, well_endpoint in cases:
            for period in production_periods:
                # Define the time parameters
                dt = 1e2
                production_period = period * pp.YEAR
                domain_size = 4.0e3
                fracture_size = 5e2
                cell_size = 10e2

                # Log run configuration
                logger.info("=" * 80)
                logger.info("Starting new simulation with configuration:")
                logger.info(
                    f"  Well configuration: {well_name} (endpoint: {well_endpoint} m)"
                )
                logger.info(f"  Boundary velocity: {velocity:.2e} m/year")
                logger.info(f"  Production period: {period} years")
                logger.info(f"  Time step (dt): {dt}")
                logger.info(f"  Domain size: {domain_size} m")
                logger.info(f"  Fracture size: {fracture_size} m")
                logger.info(f"  Cell size: {cell_size} m")
                logger.info(f"  TPFA discretization: {tp}")
                logger.info("=" * 80)

                schedule, neumann_intervals = create_schedule(production_period, dt)
                # Add the initial time step to the schedule
                # schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
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
                    [0, 2 * dt_init],
                    dt_init=dt_init,
                    dt_min_max=(1, 2 * dt_init),
                    iter_max=20,
                    iter_optimal_range=(5, 12),
                    constant_dt=False,
                )
                init_granodiorite_values = copy.deepcopy(granodiorite_values)
                simulation_name = (
                    f"velocity_{velocity:.1e}_period_{period}_" + well_name
                )
                folder_name = "Case_I/" + simulation_name
                folder_name_init = folder_name + "_initialization"
                file_name = "example_4"
                title = (
                    well_name.replace("_", " ")
                    + f", Velocity = {velocity:.1e} m/y, Production period = {period} y"
                )
                title = title[0].upper() + title[1:]
                model_params_init = {
                    "plot_title": title,
                    "domain_sizes": np.full(3, domain_size),
                    "material_constants": {
                        "solid": pp.SolidConstants(**init_granodiorite_values),
                        "fluid": pp.FluidComponent(**pp.fluid_values.water),
                    },
                    "units": pp.Units(m=1.0, kg=1.0e0, K=1.0),
                    "time_manager": time_manager_init,
                    "grid_type": "simplex",
                    "meshing_arguments": {
                        "cell_size": cell_size,
                        "cell_size_fracture": 0.7 * fracture_size,
                    },
                    "file_name": file_name,
                    "data_folder_name": f"{folder_name}_saved_data",
                    "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
                    "use_wells": False,
                    "reference_variable_values": pp.ReferenceVariableValues(
                        temperature=350.0,
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
                    "boundary_displacement_velocity_scaling": velocity / pp.YEAR,
                    "heterogeneous_permeability": True,
                    "darcy_flux_discretization": "tpfa" if tp else "mpfa",
                    "fourier_flux_discretization": "tpfa" if tp else "mpfa",
                }
                model_params = copy.deepcopy(model_params_init)
                # Reduce target pressure in favour of displacement BC as driving force?
                injection_pressures = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)

                production_pressures = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
                injection_temperatures = np.full(
                    schedule.shape, pp.Celsius_to_Kelvin(10)
                )
                production_temperatures = np.full(schedule.shape, 373.15)
                # Can be refined to have different schedules for each well.
                for name in MainModel.injection_well_names.fget(None):
                    model_params[f"{name}_pressures"] = injection_pressures
                    model_params[f"{name}_temperatures"] = injection_temperatures
                for name in MainModel.production_well_names.fget(None):
                    model_params[f"{name}_pressures"] = production_pressures
                    model_params[f"{name}_temperatures"] = production_temperatures

                # Create the model
                solver_params = {
                    "nl_convergence_res_atol": 1e-1,
                    "nl_convergence_inc_atol": 1e-4,
                    "nl_divergence_res_atol": 1e20,
                    "nl_max_iterations": 20,
                    "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
                    "local_line_search": 1,
                    "global_line_search": 0,
                    "residual_line_search_interval_size": 1e-3,
                    "constraint_violation_tolerance": 1e-3,
                    # "linear_solver": "scipy_sparse",
                }

                init_model = InitializationModel(model_params_init)

                pp.run_time_dependent_model(init_model, solver_params)
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
                    f"Determined friction coefficient from initialization: {friction_coeff:.4f}"
                )
                logger.info("=" * 80)

                granodiorite_values["friction_coefficient"] = friction_coeff
                model_params.update(
                    {
                        "time_manager": time_manager,
                        "file_name": file_name,
                        "folder_name": folder_name,
                        "initialization": False,
                        "use_wells": True,
                        "reference_from_initial": True,
                        "material_constants": {
                            "solid": pp.SolidConstants(**granodiorite_values),
                            "fluid": pp.FluidComponent(**pp.fluid_values.water),
                        },
                        "neumann_intervals": neumann_intervals,
                        "production_well_y_endpoint": well_endpoint,
                    }
                )
                model = MainModel(model_params)
                model.initialization_model = init_model
                solver_params.update(
                    {
                        "local_line_search": 1,
                        "global_line_search": 1,
                        "nl_convergence_res_atol": 1e0,
                    }
                )
                pp.run_time_dependent_model(model, solver_params)
                model.plot_well_monitoring()
                slip_onset_times[simulation_name] = model.sliding_onset_times()
    # Summarize the slip onset times in a CSV file.
    summarize_slip_onset_times(
        slip_onset_times,
        model.fracture_names(),
        boundary_velocities,
        output_file=folder_name + "/saved_data/slip_onset_times.csv",
    )
