import porepy as pp
import numpy as np
from geometry import ConceptualGeometry
from material_parameters import granodiorite_values

from physical_model import PhysicalModel
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
from porepy.examples.geothermal_reservoir import (
    WellBoundaryConditions,
    NeumannWellBCsFirstTimeInterval,
)
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
logging.basicConfig(level=logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaseModel(
    FractureDeformationExporting,
    IterationExporting,
    ResidualExporting,
    GeometryExporting,
    CosoExporter,
    ConceptualGeometry,
    # pp.poromechanics.TpsaPoromechanicsMixin,
    # diff_tpfa.DarcysLawAdEverywhere,
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


if __name__ == "__main__":
    fast = 1 == 11  # Set to 1 for fast run, 0 for full run
    # Define the time parameters
    logger.info("Starting the simulation")
    dt = 3e1
    # injection_start_time = 10e3
    # Include dt to make sure itis included in the time steps which are exported.
    schedule = np.array([0, 1, 10, 20, 50, 80, 111, 112, 113, 114, 115, 116]) * pp.DAY
    schedule = np.array(
        [0, 2 * dt, pp.DAY, 100 * pp.DAY, 102 * pp.DAY, 200 * pp.DAY, 202 * pp.DAY]
    )
    schedule = np.array(
        [
            0,
            2 * dt,  # Initial time steps with closed wells
            # pp.DAY,  # Ramp up to operation
            pp.YEAR,  # First shut-in
            pp.YEAR + 2 * pp.DAY,  # Restart operation
            2 * pp.YEAR,  # Second shut-in
            2 * pp.YEAR + 2 * pp.DAY,  # End second shut-in and end of simulation
        ]
    )
    neumann_intervals = [
        # (-1.0, schedule[1]),  # Close wells initially Skip this?
        (schedule[2], schedule[3]),  # Shut-in period 1
        (schedule[-2], schedule[-1]),  # Shut-in period 2
    ]
    # schedule += injection_start_time
    # Add the initial time step to the schedule
    # schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt,
        dt_min_max=(1e-2, max(dt, pp.YEAR / 4)),
        iter_max=20,
        iter_optimal_range=(5, 12),
        iter_relax_factors=(0.6, 1.8),
        recomp_factor=0.2,
        recomp_max=10,
    )
    dt_init = 2e9  # * pp.YEAR
    time_manager_init = pp.TimeManager(
        [0, 2 * dt_init],
        dt_init=dt_init,
        dt_min_max=(1, 2 * dt_init),
        constant_dt=True,
    )
    fracture_size = 6e2
    cell_size = 8e2
    if fast:
        cell_size = 2e3
    init_granodiorite_values = copy.deepcopy(granodiorite_values)
    suffix = "_long_well"
    suffix = ""
    folder_name = "conceptual" + suffix
    folder_name_init = folder_name + "_initialization"
    file_name = "example_4" + suffix
    model_params_init = {
        "domain_size": 2.0e3,
        "material_constants": {
            "solid": pp.SolidConstants(**init_granodiorite_values),
            "fluid": pp.FluidComponent(**pp.fluid_values.water),
            # "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),  # type: ignore[arg-type]
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
            temperature=300.0,
            #     pressure=pp.BAR,
        ),
        "thermal_gradient": 5e-2,  # 73,  # K/m  tåltes ikke
        "fracture_file": "coords.txt",
        "folder_name": folder_name_init,
        "initialization": True,
        "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
        # "lithostatic_stress_multipliers": np.array([1.0, 1.0, 1.0]),
        "fracture_params": {  # Other options are available in the geometry mixin.
            "fracture_major_axes": np.array(
                (fracture_size, fracture_size, 1.0 * fracture_size)
            ),
            # "num_points": np.array((9, 8)),  # Number of points to define each fracture
            # "dip_angles": np.array((np.pi / 4, np.pi / 2)),  # Slanted and vertical
        },
    }
    model_params = copy.deepcopy(model_params_init)
    # Reduce target pressure in favour of displacement BC as driving force?
    injection_pressures = np.full(schedule.shape, 10 * pp.MEGA * pp.PASCAL)
    # if issubclass(MainModel, NeumannWellBCsFirstTimeInterval):
    #     injection_pressures[0] = 2.0 * pp.MEGA * pp.PASCAL
    #     injection_pressures[1] = 3.0 * pp.MEGA * pp.PASCAL
    # else:
    injection_pressures[0] = 5 * pp.MEGA * pp.PASCAL

    production_pressures = np.full(schedule.shape, -1 * pp.MEGA * pp.PASCAL)  # Reduce
    injection_temperatures = np.full(schedule.shape, 323.15)
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
        "nl_convergence_tol_res": 1e-1,
        "nl_convergence_tol": 1e-4,
        "nl_divergence_tol": 1e20,
        "max_iterations": 30,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": 1,
        "global_line_search": 0,
        "residual_line_search_interval_size": 1e-3,
        "constraint_violation_tolerance": 1e-3,
        # "linear_solver": "scipy_sparse",
    }
    # Restart does not seem to work properly if used for a new instance. Main suspect is
    # configuration of file names/appendices.
    run_initialization = 1 == 1
    if not run_initialization:
        model_params_init["restart_options"] = {
            "restart": True,
            "pvd_file": Path(folder_name_init) / f"{file_name}.pvd",
            "times_file": Path(folder_name_init) / "times.json",
            "is_mdg_pvd": True,
        }
    init_model = InitializationModel(model_params_init)

    pp.run_time_dependent_model(init_model, solver_params)
    # Analyze the initialization results to set friction coefficient
    sds = init_model.mdg.subdomains(dim=2)
    traction = init_model.evaluate_and_scale(sds, "contact_traction", "-").reshape(
        (3, -1), order="F"
    )
    friction_coeff = (
        np.max(np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :])) + 0.01
    )
    granodiorite_values["friction_coefficient"] = friction_coeff
    model_params.update(
        {
            "time_manager": time_manager,
            "file_name": file_name,
            "folder_name": folder_name,
            "use_wells": True,
            "reference_from_initial": True,
            "material_constants": {
                "solid": pp.SolidConstants(**granodiorite_values),
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
                # "numerical": pp.NumericalConstants(characteristic_displacement=1e0),  # type: ignore[arg-type]
            },
            "neumann_intervals": neumann_intervals,
            "production_well_z_endpoint": 0.0,  # -1.5,
        }
    )
    if "long" in folder_name:
        model_params["production_well_z_endpoint"] = -1.5
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
