import time
import pandas
import porepy as pp
import numpy as np
from geometry import CoolingGeometry
from material_parameters import granodiorite_values
from porepy.applications.test_utils.models import add_mixin

from physical_model import PhysicalModel
from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
)
from porepy.applications.discretizations.flux_discretization import FluxDiscretization

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
from exporting import CosoExporter, GeometryExporting
from wells import WellDataConceptual
from porepy.numerics.nonlinear import line_search
import diff_tpfa
import logging
import sys
import copy
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="coso.log",
    filemode="w",
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


class BaseModel(
    FractureDeformationExporting,
    IterationExporting,
    # ResidualExporting,
    GeometryExporting,
    CosoExporter,
    CoolingGeometry,
    FluxDiscretization,
    pp.poromechanics.TpsaPoromechanicsMixin,
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


if __name__ == "__main__":
    tp = True
    no_solid_expansion = True
    fracture_strike_angles = np.array(
        [
            # 45,
            41,
            # 35,
            # 42,
            # 38,
        ]
    )
    granodiorite_values["permeability"] = 2e-15
    granodiorite_values["residual_aperture"] = 1.0e-3
    if no_solid_expansion:
        granodiorite_values["thermal_expansion"] = 0.0

    init_granodiorite_values = copy.deepcopy(granodiorite_values)

    well_options = [
        True,
        # False,
    ]
    for strike in fracture_strike_angles:
        for has_wells in well_options:
            tic = time.time()
            if has_wells:
                suffix = "_with_wells"
            else:
                suffix = "_without_wells"
            suffix += f"_strike_{int(strike)}"
            if tp:
                suffix += "_tp"
                # MainModel = add_mixin(diff_tpfa.DarcysLawAdEverywhere, MainModel)
                # InitializationModel = add_mixin(
                #     diff_tpfa.DarcysLawAdEverywhere, InitializationModel
                # )
                # MainModel = add_mixin(  # type: ignore
                #     pp.poromechanics.TpsaPoromechanicsMixin, MainModel
                # )
                # InitializationModel = add_mixin(  # type: ignore
                #     pp.poromechanics.TpsaPoromechanicsMixin, InitializationModel
                # )
            if no_solid_expansion:
                suffix += "_no_solid_expansion"
            logger.info(f"Starting the simulation with suffix {suffix}")
            dt = 1e2
            production_period = 2 * pp.YEAR
            shut_in_duration = 2 * pp.DAY  # reduce to 1 day?
            schedule = np.array(
                [
                    0,
                    1 * production_period - shut_in_duration,  # First shut-in
                    1 * production_period,  # Restart operation
                    2 * production_period - shut_in_duration,  # Second shut-in
                    2 * production_period,  # End second shut-in
                    3 * production_period - shut_in_duration,  # Third shut-in
                    3 * production_period,
                    # 4 * production_period - shut_in_duration,  # First shut-in
                    # 4 * production_period,
                    # 5 * production_period - shut_in_duration,  # Fifth shut-in
                    # 5 * production_period,  # End fifth shut-in
                ]
            )

            if has_wells:
                neumann_intervals = [
                    (schedule[2 * i + 1], schedule[2 * i + 2])
                    for i in range(schedule.size // 2 - 1)
                ]
            else:
                # Only need first and last time step if no wells are present
                schedule = np.array([schedule[0], schedule[-1]])
                dt = 100 * pp.DAY
                neumann_intervals = []
            # schedule += injection_start_time
            # Add the initial time step to the schedule
            # schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
            time_manager = pp.TimeManager(
                schedule=schedule,
                dt_init=dt,
                dt_min_max=(1e-2, max(dt, production_period / 4)),
                iter_max=20,
                iter_optimal_range=(5, 12),
                iter_relax_factors=(0.5, 2.0),
                recomp_factor=0.1,
                recomp_max=10,
            )
            dt_init = 10e9  # * pp.YEAR
            time_manager_init = pp.TimeManager(
                [0, 2 * dt_init],
                dt_init=dt_init,
                dt_min_max=(1, 2 * dt_init),
                constant_dt=True,
            )
            fracture_size = 4e2
            cell_size = 18e2

            folder_name = "thermal" + suffix
            folder_name_init = folder_name + "_initialization"
            file_name = "example_6"
            data_folder_name = f"{folder_name}_saved_data"
            model_params_init = {
                "domain_sizes": np.array([6, 6, 4]) * 1e3,
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
                "data_folder_name": data_folder_name,
                "adaptive_indicator_scaling": True,
                "initialization": True,
                "use_wells": False,
                "reference_variable_values": pp.ReferenceVariableValues(
                    temperature=350.0,
                    #     pressure=pp.BAR,
                ),
                "thermal_gradient": 7e-2,
                "fracture_file": "coords.txt",
                "folder_name": folder_name_init,
                "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
                "fracture_params": {
                    "fracture_major_axes": np.array(
                        (0.75 * fracture_size, fracture_size, fracture_size)
                    ),
                    "strike_angles": np.deg2rad([0, strike, 45]),
                },
                "darcy_flux_discretization": "tpfa" if tp else "mpfa",
                "fourier_flux_discretization": "tpfa" if tp else "mpfa",
            }
            model_params = copy.deepcopy(model_params_init)
            # Reduce target pressure in favour of displacement BC as driving force?
            injection_pressures = np.full(schedule.shape, 8 * pp.MEGA * pp.PASCAL)

            production_pressures = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
            injection_temperatures = np.full(schedule.shape, 323.15)
            production_temperatures = np.full(schedule.shape, 373.15)
            # NOTE: The current protocols seem to give significantly more injection than
            # production by mass. This is consistent with the data in production.csv
            # and injection.csv, where the ratio is about 2.5:7, albeit with two
            # production wells.

            # Can be refined to have different schedules for each well.
            for name in MainModel.injection_well_names.fget(None):
                model_params[f"{name}_pressures"] = injection_pressures
                model_params[f"{name}_temperatures"] = injection_temperatures
            for name in MainModel.production_well_names.fget(None):
                model_params[f"{name}_pressures"] = production_pressures
                model_params[f"{name}_temperatures"] = production_temperatures

            # Create the model
            solver_params = {
                "nl_convergence_tol_res": 1e-1,
                "nl_convergence_tol": 1e-3,
                "nl_divergence_tol": 1e20,
                "max_iterations": 20,
                "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
                "local_line_search": 1,
                "global_line_search": 0,
                "residual_line_search_interval_size": 1e-3,
                "constraint_violation_tolerance": 1e-3,
            }

            init_model = InitializationModel(model_params_init)

            pp.run_time_dependent_model(init_model, solver_params)
            # Analyze the initialization results to set friction coefficient
            sds = init_model.mdg.subdomains(dim=2)
            traction = init_model.evaluate_and_scale(
                sds, "contact_traction", "-"
            ).reshape((3, -1), order="F")
            friction_coeff = (
                np.max(np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :]))
                + 0.01
            )
            granodiorite_values["friction_coefficient"] = friction_coeff
            # Save granodiorite values for main model
            df = pandas.DataFrame.from_dict(granodiorite_values, orient="index")
            output_path = Path(data_folder_name) / "granodiorite_values.csv"
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, header=False)
            model_params.update(
                {
                    "time_manager": time_manager,
                    "file_name": file_name,
                    "folder_name": folder_name,
                    "use_wells": has_wells,
                    "initialization": False,
                    "reference_from_initial": True,
                    "material_constants": {
                        "solid": pp.SolidConstants(**granodiorite_values),
                        "fluid": pp.FluidComponent(**pp.fluid_values.water),
                    },
                    "neumann_intervals": neumann_intervals,
                }
            )

            model = MainModel(model_params)  # Load from initialization from file
            model.initialization_model = init_model
            solver_params.update(
                {
                    "local_line_search": 1,
                    "global_line_search": 1,
                    "nl_convergence_tol_res": 1e0,
                }
            )
            pp.run_time_dependent_model(model, solver_params)
            model.plot_well_monitoring()
            toc = time.time()
            logger.info(
                f"Simulation with suffix {suffix} completed in {toc - tic:.2f} seconds."
            )
